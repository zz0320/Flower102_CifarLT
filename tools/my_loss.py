# -*- coding: utf-8 -*-
"""
# @file name    :my_loss.py
# @author       :zz0320
# @data         :2022-4-10
# @brief        :重写的loss函数
"""

from abc import ABC
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing = 0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

if __name__ == '__main__':

    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    criterion = LabelSmoothLoss(0.001)
    loss = criterion(output, label)

    print("CrossEntropy:{}".format(loss))




