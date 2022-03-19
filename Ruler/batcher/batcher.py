
import sys
from math import floor
import numpy as np
import torch
sys.path.append("..")
from attacks import PGDAttacker


class Batcher(object):
    def __init__(self, limit, args, device):
        self.batch_size = args.batch_size
        self.limit = limit
        self.attack_steps = args.attack_steps
        self.attack_lr = args.attack_lr
        self.adv_sample_limit = floor(args.adv_ratio * args.batch_size)
        self.device = device
        constraint = np.load(args.constraint_path)
        self.attacker = PGDAttacker(args.attack_eps, args.protected_attribs, constraint)

    def next_batch_unatt(self, start, x_set, y_set):
        if start + self.batch_size < self.limit:
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        else:
            self.batch_size = self.limit - 1
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        x_ori = x_ori.astype(np.float32)
        x_ori = torch.from_numpy(x_ori).to(self.device)
        y = y.astype(np.longlong)
        y = torch.from_numpy(y).to(self.device)
        return start + len(y), x_ori, y

    def next_batch_original(self, model, start, x_set, y_set):
        # print('I ma in original')
        if start + self.batch_size < self.limit:
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        else:
            self.batch_size = self.limit - 1
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        x_ori = x_ori.astype(np.float32)
        x_ori = torch.from_numpy(x_ori).to(self.device)
        y = y.astype(np.longlong)
        y = torch.from_numpy(y).to(self.device)
        end = start + self.batch_size
        cur_adv_sample_num, x_adv = self.attacker.attack_original(x_ori, y, model)
        return cur_adv_sample_num, end, x_adv, y

    def next_batch(self, model, start, x_set, y_set):
        """
        :param model:The net
        :param start:The start index of train_set
        :param x_set:The train data
        :param y_set: The label data
        :return:(end, x_batch, y_batch), end is the next index; x_batch includes x_ori and x_adv,
        batch_size equals to self.batch_size
        """

        # x_set = train_set.train_x()
        # y_set = train_set.train_y()
        if start + self.batch_size < self.limit:
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        else:
            self.batch_size = self.limit - 1
            x_ori = x_set[start:start + self.batch_size, :]
            y = y_set[start:start + self.batch_size]
        x_ori = x_ori.astype(np.float32)
        x_ori = torch.from_numpy(x_ori).to(self.device)
        y = y.astype(np.longlong)
        y = torch.from_numpy(y).to(self.device)
        adv_samples = []
        adv_y = []
        cur_batch_size = index = cur_adv_sample_num = end = 0
        flag_adv = True
        while index < self.batch_size:
            flag = False
            x_ori_sample = torch.unsqueeze(x_ori[index, :], 0)
            y_sample = torch.unsqueeze(y[index], 0)
            if flag_adv:
                flag, x_adv_sample = self.attacker.attack_sample(x_ori_sample, y_sample, model,
                                                                 self.attack_steps,
                                                                 self.attack_lr)
            if flag:
                adv_samples.extend([x_ori_sample, x_adv_sample])
                adv_y.extend([y_sample, y_sample])
                cur_batch_size += 2
                cur_adv_sample_num += 1
            else:
                adv_samples.append(x_ori_sample)
                adv_y.append(y_sample)
                cur_batch_size += 1
            if flag_adv and cur_adv_sample_num >= self.adv_sample_limit:
                flag_adv = False
            if cur_batch_size == self.batch_size:
                end = start + index + 1
                break
        x_adv = torch.cat(adv_samples).to(self.device)
        y_adv = torch.cat(adv_y).to(self.device)
        return cur_adv_sample_num, end, x_adv, y_adv
