# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn

from torch.nn import functional as F
import numpy as np

def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
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

def wce_loss(logits, targets, weights, reduction='none'):
    if logits.shape != targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-F.one_hot(targets,10) * log_pred, dim=1)#10Àà
        nll_loss = (nll_loss.T * weights[targets]).T  # apply weights
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    # else:
    #     log_pred = F.log_softmax(logits, dim=-1)
    #     weighted_loss = F.nll_loss(log_pred, targets, weight=weights, reduction=reduction)
    else:
        print("fail")
        return weighted_loss


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):#m=0.5,s=30
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, reduction='none'):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        #if reduction == 'none':
        #    return F.cross_entropy(self.s * output, target, weight=self.weight, reduction=reduction)
        #else:
        #    return F.cross_entropy(self.s * output, target, weight=self.weight, reduction=reduction).mean()
        return F.cross_entropy(self.s * output, target, weight=self.weight, reduction=reduction).mean()


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        targets=targets.long()
        return ce_loss(logits, targets, reduction)

class WCELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, weights, reduction='none'):
        targets=targets.long()
        return wce_loss(logits, targets, weights, reduction)

