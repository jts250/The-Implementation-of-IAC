# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import numpy as np
import torch
from semilearn.core.criterions import LDAMLoss
import torch.nn.functional as F

@ALGORITHMS.register('ldam_cs')
class LDAM_cs(AlgorithmBase):
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
        self.lb_class_dist = lb_class_dist #np.array(torch.from_numpy(np.min(lb_class_dist) / lb_class_dist))

    def train_step(self, x_lb, y_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            #if not self.lb_class_dist.is_cuda:
            #    self.lb_class_dist = self.lb_class_dist.to(y_lb.device)
            idx = self.epoch // 160
            #print('hehe')
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.lb_class_dist)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.lb_class_dist)
            #per_cls_weights = torch.FloatTensor(per_cls_weights).to(y_lb.device)
            
            ldam_loss=LDAMLoss(self.lb_class_dist,np.array(per_cls_weights))

            logits_x_lb = self.model(x_lb)['logits']
            #sup_loss = self.wce_loss(logits_x_lb, y_lb.long(), weights=self.lb_class_dist, reduction='mean')
            sup_loss = ldam_loss(logits_x_lb, y_lb.long(), reduction='mean')/4.0

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