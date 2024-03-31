import os
import copy
import torch
import numpy as np

from .utils import ProgressiveDistAlignEMAHook, CReSTCheckpointHook, CReSTLoggingHook

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import get_dataset, get_data_loader, send_model_cuda, IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from semilearn.core.hooks import ParamUpdateHook

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

class CCParamUpdateHook(ParamUpdateHook):
    """
    Parameter Update Hook

    necessary for update the model parameters
    """

    def before_train_step(self, algorithm):
        if hasattr(algorithm, 'start_run'):
            torch.cuda.synchronize()
            algorithm.start_run.record()

    # call after each train_step to update parameters
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.optimizer.step()

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()

        if hasattr(algorithm, 'end_run'):
            algorithm.end_run.record()
            torch.cuda.synchronize()
            algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.

class IACNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features
        self.inver_aux_classifier=nn.Linear(self.backbone.num_features, num_classes)

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


@IMB_ALGORITHMS.register('crest_iac')
class CReST_IAC(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.imb_init(num_gens=args.crest_num_gens, dist_align_t=args.crest_dist_align_t, pro_dist_align=args.crest_pro_dist_align, sampling_alpha=args.crest_alpha)
        super(CReST_IAC, self).__init__(args, net_builder, tb_log, logger, **kwargs)
        self.rou_iac = 1.9

        # comput lb imb ratio
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist_iac = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        self.lower_bound = torch.from_numpy(0.1 / lb_class_dist)

        # TODO: better ways
        self.model = IACNet(self.model, num_classes=self.num_classes)
        self.ema_model = IACNet(self.ema_model, num_classes=self.num_classes)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.optimizer, self.scheduler = self.set_optimizer()

    def imb_init(self, num_gens=6, dist_align_t=0.5, pro_dist_align=True, sampling_alpha=3):
        self.num_gens = num_gens
        self.dist_align_t = dist_align_t
        self.pro_dist_align = pro_dist_align
        self.sampling_alpha = sampling_alpha
        self.start_gen = 0
        self.pseudo_label_list = None
        self.best_gen = 0 
        self.best_gen_eval_acc = 0.0
    
    def set_hooks(self):
        super().set_hooks()

        #reset paramupdate
        # self.register_hook(CCParamUpdateHook(), "ParamUpdateHook", "HIGHEST")
        
        # reset checkpoint hook
        self.register_hook(CReSTCheckpointHook(), "CheckpointHook", "HIGH")
        self.register_hook(CReSTLoggingHook(), "LoggingHook", "LOW")

        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in  self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        lb_class_dist = lb_class_dist / lb_class_dist.sum()
        self.lb_class_dist = lb_class_dist

        # get ground truth distribution
        if self.pro_dist_align:
            self.register_hook(
                ProgressiveDistAlignEMAHook(num_classes=self.num_classes, p_target_type='gt', p_target=lb_class_dist), 
                "DistAlignHook")

    def get_split(self, lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list=None):
        if pseudo_label_list is not None and len(pseudo_label_list):
            data_picked = []
            targets_picked = []

            lb_class_dist = self.lb_class_dist
            sorted_class = np.argsort(lb_class_dist)[::-1]
            class_imb_ratio = lb_class_dist[sorted_class][0] / lb_class_dist[sorted_class[-1]]  # self.lb_imb_ratio
            class_imb_ratio = 1. / class_imb_ratio
            mu = np.math.pow(class_imb_ratio, 1 / (self.num_classes - 1))

            for c in sorted_class:
                num_picked = int(
                    len(pseudo_label_list[c]) *
                    np.math.pow(np.math.pow(mu, (self.num_classes - 1) - c), 1 / self.sampling_alpha))  # this is correct!!!
                idx_picked = pseudo_label_list[c][:num_picked]

                try:
                    if len(idx_picked) > 0:
                        data_picked.append(eval_ulb_data[idx_picked])
                        targets_picked.append(np.ones_like(eval_ulb_targets[idx_picked]) * c)
                        print('class {} is added {} pseudo labels'.format(c, num_picked))
                except:
                    continue
            data_picked.append(lb_data)
            targets_picked.append(lb_targets)
            lb_data = np.concatenate(data_picked, axis=0)
            lb_targets = np.concatenate(targets_picked, axis=0)
        else:
            self.print_fn('Labeled data not update')
        return lb_data, lb_targets

    def set_dataset(self, pseudo_label_list=None):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        
        # set include_lb_to_ulb to False
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, self.args.data_dir, include_lb_to_ulb=False)
        # eval_ulb
        dataset_dict['eval_ulb'] = copy.deepcopy(dataset_dict['train_ulb'])
        dataset_dict['eval_ulb'].is_ulb = False

        # add pseudo labels into lb
        lb_data, lb_targets = dataset_dict['train_lb'].data, dataset_dict['train_lb'].targets
        eval_ulb_data, eval_ulb_targets = dataset_dict['eval_ulb'].data, dataset_dict['eval_ulb'].targets
        lb_data, lb_targets = self.get_split(lb_data, lb_targets, eval_ulb_data, eval_ulb_targets, pseudo_label_list)
        dataset_dict['train_lb'].data = lb_data
        dataset_dict['train_lb'].targets = lb_targets

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}, unlabeled eval data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len, len(dataset_dict['eval_ulb'])))
        
        
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict
    
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

    def re_init(self):
        self.it = 0
        self.best_gen_eval_acc = 0.0
        self.ema = None
        
        # build dataset with pseudo label list
        self.dataset_dict = self.set_dataset(self.pseudo_label_list)

        # build model and ema_model
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        self.model = IACNet(self.model, num_classes=self.num_classes)
        self.ema_model = IACNet(self.ema_model, num_classes=self.num_classes)

        self.model = send_model_cuda(self.args, self.model)
        self.ema_model = send_model_cuda(self.args, self.ema_model)
        self.ema_model.load_state_dict(self.model.state_dict())

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build data loader
        self.loader_dict = self.set_data_loader()

    def train(self):

        # EMA Init
        self.model.train()

        for gen in range(self.start_gen, self.num_gens):
            self.gen = gen

            # before train generation
            if self.pro_dist_align:
                cur = self.gen / ( self.num_gens - 1)
                self.cur_dist_align_t = (1.0 - cur) * 1.0 + cur * self.dist_align_t
            else:
                self.cur_dist_align_t = self.dist_align_t

            # reinit every generation
            if self.gen > 0:
                self.re_init()
            
            self.call_hook("before_run")


            for epoch in range(self.start_epoch, self.epochs):
                self.epoch = epoch
                
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_epoch")
                for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                             self.loader_dict['train_ulb']):
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    # NOTE: progressive dist align will be called inside each train_step in core algorithms
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                    self.call_hook("after_train_step")
                    self.it += 1

                self.call_hook("after_train_epoch")
            
            # after train generation
            eval_dict = {'eval/best_acc': self.best_gen_eval_acc, 'eval/best_it': self.best_it}
            for key, item in eval_dict.items():
                self.print_fn(f"CReST Generation {gen}, Model result - {key} : {item}")

            self.print_fn(f"Generation {self.gen} finished, updating pseudo label list")
            ulb_logits = self.evaluate('eval_ulb', return_logits=True)['eval_ulb/logits']
            if isinstance(ulb_logits, np.ndarray):
                ulb_logits = torch.from_numpy(ulb_logits)
            ulb_score, ulb_pred = torch.max(torch.softmax(ulb_logits, dim=1), dim=1)
            self.pseudo_label_list = []
            for c in range(self.num_classes):
                idx_gather = torch.where(ulb_pred == c)[0]
                if len(idx_gather) == 0:
                    self.pseudo_label_list.append([])
                    continue
                score_gather = ulb_score[idx_gather]
                score_sorted_idx = torch.argsort(score_gather, descending=True)
                idx_gather = idx_gather[score_sorted_idx]
                self.pseudo_label_list.append(idx_gather.numpy())

            self.it = 0 
        
        self.call_hook("after_run")

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['gen'] = self.gen
        if self.pro_dist_align:
            save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
            save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.gen = checkpoint['gen']
        self.start_gen = checkpoint['gen']
        if self.pro_dist_align:
            self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
            self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--crest_num_gens', int, 6),
            SSL_Argument('--crest_dist_align_t', float, 0.5),
            SSL_Argument('--crest_pro_dist_align', str2bool, True),
            SSL_Argument('--crest_alpha', float, 3),
        ]

    def train_step(self, *args, **kwargs):

        out_dict, log_dict = super().train_step(*args, **kwargs)

        # get features
        feats_x_lb = out_dict['feat']['x_lb']
        feats_x_ulb_w = out_dict['feat']['x_ulb_w']
        feats_x_ulb_s = out_dict['feat']['x_ulb_s']
        if isinstance(feats_x_ulb_s, list):
            feats_x_ulb_s = feats_x_ulb_s[0]

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

    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()

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
            iac_ulb_loss += 0.000 * (
                ce_loss(torch.log(self.compute_prob(logits_s).mean(0)).reshape(1, -1), pred_a.reshape(1, -1),
                        reduction='none')).mean()

        iac_loss = iac_lb_loss + iac_ulb_loss
        return iac_loss

    def process_batch(self, **kwargs):
        # get core algorithm parameteters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)