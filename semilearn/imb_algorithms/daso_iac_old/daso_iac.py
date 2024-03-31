import os
import queue
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from .utils import DASOFeatureQueue, DASOPseudoLabelingHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
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
        self.inver_aux_classifier = nn.Linear(self.backbone.num_features, num_classes)

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

@IMB_ALGORITHMS.register('daso_iac')
class DASO_IAC(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(T_proto=args.daso_T_proto, T_dist=args.daso_T_dist, daso_queue_len=args.daso_queue_len,
                      interp_alpha=args.daso_interp_alpha, with_dist_aware=args.daso_with_dist_aware, assign_loss_ratio=args.daso_assign_loss_ratio,
                      num_pl_dist_iter=args.daso_num_pl_dist_iter, num_pretrain_iter=args.daso_num_pretrain_iter)
        super().__init__(args, net_builder, tb_log, logger, **kwargs)

        self.rou_iac = 1.9
        # comput lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        self.lower_bound = torch.from_numpy(0.2 / lb_class_dist)
        self.model = IACNet(self.model, num_classes=self.num_classes)
        self.ema_model = IACNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

        # get queue
        self.queue = DASOFeatureQueue(num_classes=self.num_classes, 
                                      feat_dim=self.model.num_features, 
                                      queue_length=self.daso_queue_len)
        self.similarity_fn = nn.CosineSimilarity(dim=2)


    def imb_init(self, T_proto=0.05, T_dist=1.5, 
                       daso_queue_len=256, interp_alpha=0.3, with_dist_aware=True, assign_loss_ratio=1.0, 
                       num_pl_dist_iter=100, num_pretrain_iter=5120):
        self.T_proto = T_proto
        self.T_dist = T_dist
        self.daso_queue_len = daso_queue_len
        self.interp_alpha = interp_alpha
        self.lambda_f = assign_loss_ratio
        self.with_dist_aware = with_dist_aware
        self.num_pl_dist_iter = num_pl_dist_iter
        self.num_pretrain_iter = num_pretrain_iter

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(DASOPseudoLabelingHook(num_classes=self.num_classes, T_dist=self.T_dist, with_dist_aware=self.with_dist_aware, interp_alpha=self.interp_alpha), 
                           "PseudoLabelingHook", "LOWEST")

    def process_batch(self, **kwargs):
        # get core algorithm parameteters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)


    def train_step(self, *args, **kwargs):
        # push memory queue using ema model
        self.ema.apply_shadow()
        with torch.no_grad():
            x_lb, y_lb = kwargs['x_lb'], kwargs['y_lb']
            feats_x_lb = self.model(x_lb)['feat']
            self.queue.enqueue(feats_x_lb.clone().detach(), y_lb.clone().detach())
        self.ema.restore()

        # forward through loop
        out_dict, log_dict = super().train_step(*args, **kwargs)
        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

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
        iac_loss = self.compute_iac_loss(
            logits_x_lb=logits_x_lb,
            y_lb=kwargs['y_lb'],
            logits_x_ulb_w=logits_x_ulb_w,
            logits_x_ulb_s=logits_x_ulb_s
        )
        out_dict['loss'] += iac_loss


        if self.it + 1 < self.num_pretrain_iter:
            # get core algorithm output
            return out_dict, log_dict 
        
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

        # compute semantic labels
        prototypes = self.queue.prototypes  # (K, D)

        with torch.no_grad():
            # similarity between weak features and prototypes  (B, K)
            sim_w = self.similarity_fn(feats_x_ulb_w.unsqueeze(1), prototypes.unsqueeze(0)) / self.T_proto
            prob_sim_w = sim_w.softmax(dim=1)
        self.probs_sim = prob_sim_w.detach()

        # compute soft loss
        # similarity between strong features and prototypes  (B, K)
        sim_s = self.similarity_fn(feats_x_ulb_s.unsqueeze(1), prototypes.unsqueeze(0)) / self.T_proto
        assign_loss = self.ce_loss(sim_s, prob_sim_w, reduction='mean')

        # add assign loss 
        out_dict['loss'] += self.lambda_f * assign_loss
        log_dict['train/assign_loss'] = assign_loss.item()
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        # save_dict['queue_bank'] = self.queue.bank
        save_dict['queue_prototypes'] = self.queue.prototypes.cpu()
        save_dict['pl_list'] = self.hooks_dict['PseudoLabelingHook'].pseudo_label_list
        save_dict['pl_dist'] = self.hooks_dict['PseudoLabelingHook'].pseudo_label_dist
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        # self.queue.bank = checkpoint['queue_bank'] 
        self.queue.prototypes = checkpoint['queue_prototypes'] 
        self.hooks_dict['PseudoLabelingHook'].pseudo_label_list = checkpoint['pl_list']
        self.hooks_dict['PseudoLabelingHook'].pseudo_label_dist = checkpoint['pl_dist']
        return checkpoint

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

        # compute labeled abc loss
        mask_lb = self.bernouli_mask(torch.max(torch.pow(self.lb_class_dist[y_lb.type(torch.long)], self.rou_iac),
                                               self.lower_bound[y_lb.type(torch.long)]))
        iac_lb_loss = (self.ce_loss(logits_x_lb, y_lb, reduction='none') * mask_lb).mean()

        # compute unlabeled abc loss
        with torch.no_grad():
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
            max_probs, y_ulb = torch.max(probs_x_ulb_w, dim=1)
            mask_ulb_1 = max_probs.ge(0.95).to(logits_x_ulb_w.dtype)
            mask_ulb = mask_ulb_1

        iac_ulb_loss = 0.0
        for logits_s in logits_x_ulb_s:
            iac_ulb_loss += (self.ce_loss(logits_s, y_ulb, reduction='none') * mask_ulb).mean()
            # iac_ulb_loss += 0.003 * (
            #     ce_loss(torch.log(self.compute_prob(logits_s).mean(0)).reshape(1, -1), pred_a.reshape(1, -1),
            #             reduction='none')).mean()

        iac_loss = iac_lb_loss + iac_ulb_loss
        return iac_loss

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--daso_queue_len', int, 256),
            SSL_Argument('--daso_T_proto', float, 0.05),
            SSL_Argument('--daso_T_dist', float, 1.5),
            SSL_Argument('--daso_interp_alpha', float, 0.5),
            SSL_Argument('--daso_with_dist_aware', str2bool, True),
            SSL_Argument('--daso_assign_loss_ratio', float, 1.0),
            SSL_Argument('--daso_num_pl_dist_iter', int, 100),
            SSL_Argument('--daso_num_pretrain_iter', int, 5120),
        ]        


    
    