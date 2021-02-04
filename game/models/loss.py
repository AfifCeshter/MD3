#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch


class StateNLLLoss(torch.nn.modules.loss._Loss):
    """
    a personal negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.
    Shape:
        - y_pred: (batch, num_docs)
        - y_true: (batch, num_docs), 0 or 1
        - output: loss
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(StateNLLLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, y_pred, y_true):

        y_pred_log = torch.log(y_pred)
        valid_docs_prob_log = y_pred_log * y_true.float()
        batch_loss = -valid_docs_prob_log.sum(dim=-1)

        if self.reduction == 'none':
            return batch_loss
        elif self.reduction == 'sum':
            return batch_loss.sum()
        elif self.reduction == 'mean':
            return batch_loss.sum() / batch_loss.shape[0]
        else:
            raise ValueError(self.reduction)
