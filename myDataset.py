#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.data_1 = np.load("new_data/x_dark_light_val_first.npy")
        self.data_2 = np.load("new_data/x_dark_light_val_second.npy")
        self.data_3 = np.load("new_data/x_dark_val_area_second.npy")
        self.label = np.load("new_data/y_dark_light_val.npy")

    def __getitem__(self, index):
        img = []
        for i in range(20):
            tmp = []
            tmp.append(self.data_1[index][i])
            tmp.append(self.data_2[index][i])
            #tmp.append(self.data_3[index][i])
            img.append(tmp)
        label = 0
        if self.label[index][0] == 1:
            label = 1
        return np.array(img), label

    def __len__(self):
        return 2836

if __name__ == '__main__':
    test_dataset = MyDataset()
    end_0 = 0
    # data = np.load("new_data/x_dark_light_val.npy")
    # for i in range(len(data)):
    #     img, label =test_dataset[i]
    #     a = list(img[:, 0])
    #     b = list(img[:, 1])
    #     tmp = a + b
    #     print(np.array(tmp) == data[i])
    #     print(label)

