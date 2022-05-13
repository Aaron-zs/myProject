#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):

        super(BiLSTM, self).__init__()
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,bidirectional=True, dropout=0.5, batch_first=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.lstm(x)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # [b, hid_dim*2] => [b, 1]
        #hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out


class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE, hidden_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None代表将隐藏状态初始化为0
        o = self.out(r_out[:, -1, :])  # 选择最后一个时间步的r_out作为输出
        return o
