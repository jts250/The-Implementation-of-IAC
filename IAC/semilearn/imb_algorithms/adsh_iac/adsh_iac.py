import copy

from semilearn.core import ImbAlgorithmBase
from semilearn.algorithms.utils import SSL_Argument
from semilearn.core.utils import get_data_loader, IMB_ALGORITHMS
from .utils import AdaptiveThresholdingHook
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

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

        self.inver_aux_classifier =nn.Linear(self.backbone.num_features, num_classes)

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


@IMB_ALGORITHMS.register('adsh_iac')
class Adsh_IAC(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(tau_1=args.adsh_tau_1)
        # super().__init__(args, **kwargs)
        super(Adsh_IAC, self).__init__(args, net_builder, tb_log, logger, **kwargs)
        assert args.algorithm == 'fixmatch', "Adsh only supports FixMatch as the base algorithm."

        self.rou_iac = 1.9

        # comput lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        self.lower_bound = torch.from_numpy(0.1 / lb_class_dist)

        # TODO: better ways
        self.model = IACNet(self.model, num_classes=self.num_classes)
        self.ema_model = IACNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

    def imb_init(self, tau_1):
        self.tau_1 = tau_1

    def set_dataset(self):
        dataset_dict = super().set_dataset()
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False
        return dataset_dict

    #'''
    def process_batch(self, **kwargs):
        # get core algorithm parameteters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, *args, **kwargs):

        out_dict, log_dict = super().train_step(*args, **kwargs)

        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

        # compute iac loss using logits_aux from dict
        # get logits
        logits_x_lb = self.model.inver_aux_classifier(feats_x_lb)
        logits_x_ulb_s = self.model.inver_aux_classifier(feats_x_ulb_s)
        with torch.no_grad():
            logits_x_ulb_w = self.model.inver_aux_classifier(feats_x_ulb_w)

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
    #'''

    def set_data_loader(self):
        loader_dict = super().set_data_loader()

        # add unlabeled evaluation data loader
        loader_dict['eval_ulb'] = get_data_loader(self.args,
                                                  self.dataset_dict['eval_ulb'],
                                                  self.args.eval_batch_size,
                                                  data_sampler=None,
                                                  shuffle=False,
                                                  num_workers=self.args.num_workers,
                                                  drop_last=False)

        return loader_dict

    def set_hooks(self):
        super().set_hooks()

        # reset hooks
        self.register_hook(AdaptiveThresholdingHook(self.num_classes, self.tau_1), "MaskingHook")

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--adsh_tau_1', float, 0.95),
        ]

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

    def compute_iac_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s):
        if not isinstance(logits_x_ulb_s, list):
            logits_x_ulb_s = [logits_x_ulb_s]

        if not self.lb_class_dist.is_cuda:
            self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
        if not self.lower_bound.is_cuda:
            self.lower_bound = self.lower_bound.to(y_lb.device)

        pred_a = torch.pow(self.lb_class_dist, self.rou_iac-1)
        pred_a = pred_a / pred_a.sum()
        mask_lb = self.bernouli_mask(torch.max(torch.pow(self.lb_class_dist[y_lb.type(torch.long)], self.rou_iac),
                                               self.lower_bound[y_lb.type(torch.long)]))

        # compute labeled abc loss
        iac_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(self.p_cutoff).to(logits_x_ulb_w.dtype)
            mask_ulb = mask_ulb_1

        iac_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            iac_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
            iac_ulb_loss += 0.003 * (
                ce_loss(torch.log(self.compute_prob(logits_s).mean(0)).reshape(1, -1), pred_a.reshape(1, -1),
                        reduction='none')).mean()

        iac_loss = iac_lb_loss + iac_ulb_loss
        return iac_loss
