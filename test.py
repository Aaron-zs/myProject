#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from myDataset import *
from torch.utils.data import DataLoader
from LSTM import *
from tensorboardX import SummaryWriter
import torch

if __name__ == '__main__':
    # loss_fn = nn.BCEWithLogitsLoss()
    # a = torch.Tensor([1.0, 2.0])
    # a = a.unsqueeze(1)
    # b = torch.Tensor([0.8])
    # print(torch.cuda.device_count())

    A = torch.randint(2, (10,))
    print(A)
    B = torch.randint(2, (10,))
    print(B)
    print((A == B).sum())
