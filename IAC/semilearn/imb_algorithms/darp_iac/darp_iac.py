import numpy as np
from inspect import signature

from .utils import DARPPseudoLabelingHook
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument
import torch.nn as nn
import torch

import torch.nn.functional as F
def ce_loss(logits, targets, reduction='none'):
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)

class IACNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.inver_aux_classifier=nn.Sequential(nn.Linear(self.backbone.num_features, self.backbone.num_features),nn.ReLU(),nn.Linear(self.backbone.num_features, num_classes))

    def forward(self, x, **kwargs):
        results_dict = self.backbone(x, **kwargs)
        results_dict['logits_inver_aux'] = self.inver_aux_classifier(results_dict['feat'])
        return results_dict

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, 'backbone'):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix='backbone.backbone')
        else:
            matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@IMB_ALGORITHMS.register('darp_iac')
class DARP_IAC(ImbAlgorithmBase):
    def __init__(self, args, **kwargs):
        self.imb_init(warmup_epochs=args.darp_warmup_epochs, alpha=args.darp_alpha, iter_T=args.darp_iter_T, num_refine_iter=args.darp_num_refine_iter)
        super().__init__(args, **kwargs)

        # comput lb imb ratio
        self.rou_iac = 1.9
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist_iac = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)

        self.lower_bound = torch.from_numpy(0.1 / lb_class_dist)
        self.model = IACNet(self.model, num_classes=self.num_classes)
        self.ema_model = IACNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

        
    def imb_init(self, warmup_epochs=200, alpha=2.0, iter_T=10, num_refine_iter=10):
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.iter_T = iter_T
        self.num_refine_iter = num_refine_iter

    def set_hooks(self):
        super().set_hooks()

        # get ground truth distribution
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        target_disb = lb_class_dist * len(self.dataset_dict['train_ulb']) / lb_class_dist.sum()

        # reset PseudoLabelingHook hook
        self.register_hook(DARPPseudoLabelingHook(warmup_epochs=self.warmup_epochs, alpha=self.alpha, iter_T=self.iter_T,
                                                  num_refine_iter=self.num_refine_iter, dataset_len=len(self.dataset_dict['train_ulb']),
                                                  num_classes=self.num_classes, target_disb=target_disb),
                           "PseudoLabelingHook",)

    def process_batch(self, **kwargs):
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys()) + ['idx_ulb']
        return super().process_batch(input_args=input_args, **kwargs)

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

    def train_step(self, *args, **kwargs):
        self.idx_ulb = kwargs['idx_ulb']
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        if 'idx_ulb' not in input_args:
            kwargs.pop('idx_ulb')

        out_dict, log_dict = super().train_step(*args, **kwargs)
        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

        # get logits
        logits_x_lb = self.model.module.inver_aux_classifier(feats_x_lb)
        logits_x_ulb_s = self.model.module.inver_aux_classifier(feats_x_ulb_s)
        with torch.no_grad():
            logits_x_ulb_w = self.model.module.inver_aux_classifier(feats_x_ulb_w)

        # compute iac loss using logits_aux from dict
        iac_loss = self.compute_iac_loss(
            logits_x_lb=logits_x_lb,
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s
        )
        out_dict['loss'] += iac_loss
        log_dict['train/iac_loss'] = iac_loss.item()

        return out_dict, log_dict
    
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['pseudo_orig'] = self.hooks_dict['PseudoLabelingHook'].pseudo_orig
        save_dict['pseudo_refine'] = self.hooks_dict['PseudoLabelingHook'].pseudo_refine
        return save_dict
        
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['PseudoLabelingHook'].pseudo_orig = checkpoint['pseudo_orig']
        self.hooks_dict['PseudoLabelingHook'].pseudo_refine = checkpoint['pseudo_refine']

    def compute_iac_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]

        if not self.lb_class_dist_iac.is_cuda:
            self.lb_class_dist_iac = self.lb_class_dist_iac.to(y_lb.device)
        if not self.lower_bound.is_cuda:
            self.lower_bound = self.lower_bound.to(y_lb.device)

        pred_a = torch.pow(self.lb_class_dist_iac, self.rou_iac-1)
        pred_a = pred_a / pred_a.sum()

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(torch.max(torch.pow(self.lb_class_dist_iac[y_lb.type(torch.long)], self.rou_iac),
                                               self.lower_bound[y_lb.type(torch.long)]))
        iac_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(0.95).to(logits_x_ulb_w.dtype)
            mask_ulb = mask_ulb_1

        iac_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            iac_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
            iac_ulb_loss += 0.003 * (
                ce_loss(torch.log(self.compute_prob(logits_s).mean(0)).reshape(1, -1), pred_a.reshape(1, -1),
                        reduction='none')).mean()

        iac_loss = iac_lb_loss + iac_ulb_loss
        return iac_loss


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--darp_warmup_epochs', int, 200),
            SSL_Argument('--darp_alpha', float, 2.0),
            SSL_Argument('--darp_iter_T', int, 10),
            SSL_Argument('--darp_num_refine_iter', int, 10),
        ]
