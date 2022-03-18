import torch
import torch.utils.data as data

import random
import numpy as np
import pandas as pd


class Dataloader(data.Dataset):
    def __init__(self, dataset_x, dataset_y):
        self.x_dataset = np.load(dataset_x)
        self.y_dataset = np.load(dataset_y)

        self.data_len = self.x_dataset.shape[0]

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        return self.x_dataset[item].astype(np.float32), self.y_dataset[item].astype(np.int64)

    def train_x(self):
        return self.x_dataset

    def train_y(self):
        return self.y_dataset
