import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Dense(nn.Module):
    def __init__(self, x_dim, y_dim=2):
        super().__init__()

        self.dense_model = nn.Sequential(
            nn.Linear(x_dim, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 5),
            nn.ReLU(inplace=True),
            nn.Linear(5, y_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_ = self.dense_model(x)
        # print(y_.is_cuda, x.is_cuda)

        # y_pre = self.sigmoid(y_)
        y_pre = y_
        # y_pre = np.argmax(y_sigmoid, 1)

        return y_pre
