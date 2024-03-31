# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import numpy as np
import torch
from semilearn.core.criterions import WCELoss
import torch.nn.functional as F
from semilearn.core.hooks import ParamUpdateHook


class DecoupleParamUpdateHook(ParamUpdateHook):
    def after_train_step(self, algorithm):
        loss = algorithm.out_dict['loss']
        # algorithm.optimizer.zero_grad()
        # update parameters
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

        # cossl teacher classifier update
        if algorithm.epoch >= algorithm.warm_epoch:
            for param in algorithm.model.parameters():
                param.requires_grad = False
            for param in algorithm.model.classifier.parameters():
                param.requires_grad = True

        if algorithm.scheduler is not None:
            algorithm.scheduler.step()
        algorithm.model.zero_grad()

        algorithm.end_run.record()
        torch.cuda.synchronize()
        algorithm.log_dict['lr'] = algorithm.optimizer.param_groups[-1]['lr']
        algorithm.log_dict['train/run_time'] = algorithm.start_run.elapsed_time(algorithm.end_run) / 1000.

@ALGORITHMS.register('decouple')
class Decouple(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        lb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_lb'].targets:
            lb_class_dist[c] += 1
        self.class_frequency = lb_class_dist
        lb_class_dist = np.array(lb_class_dist)
        self.mask_dist = torch.from_numpy(np.min(lb_class_dist) / lb_class_dist)
        self.lb_class_dist = (np.min(lb_class_dist) / lb_class_dist) #torch.from_numpy((np.min(lb_class_dist) / lb_class_dist))
        self.lb_class_dist =np.power(self.lb_class_dist,1.0)
        self.lb_class_dist = torch.from_numpy(self.lb_class_dist / np.sum(self.lb_class_dist) * len(self.lb_class_dist) )
        print(self.lb_class_dist)
        print("haha")
        self.wce_loss = WCELoss()
        self.warm_epoch = args.warm_epoch
        print(self.warm_epoch)

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if not self.lb_class_dist.is_cuda:
                self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
            if not self.mask_dist.is_cuda:
                self.mask_dist = self.mask_dist.to(x_lb.device)

            logits_x_lb = self.model(x_lb)['logits']
            if self.epoch < self.warm_epoch:
                sup_loss = self.ce_loss(logits_x_lb, y_lb.long(), reduction='mean')
            else:
                #mask_lb = self.bernouli_mask(torch.pow(self.mask_dist[y_lb.type(torch.long)], 1.0))
                #sup_loss = (self.ce_loss(logits_x_lb, y_lb.long(), reduction='none')* mask_lb).mean()
                sup_loss = self.wce_loss(logits_x_lb, y_lb.long(), weights=self.lb_class_dist, reduction='mean')

        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict

    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")
    def set_hooks(self):
        super().set_hooks()

        # reset ParamUpdateHook hook, CoSSL training code is implemented here
        self.register_hook(DecoupleParamUpdateHook(), "ParamUpdateHook", "HIGHEST")
        
    @staticmethod
    @torch.no_grad()
    def bernouli_mask(x):
        return torch.bernoulli(x.detach()).float()


#ALGORITHMS['decouple'] = Decouple