import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import torch.nn as nn

import os.path as osp
import time
import utils_yuc as utils

import dataloaders
import models
from attacks import PGDAttacker
from batcher import Batcher

import numpy as np

from tqdm import tqdm


class Trainer:
    def __init__(self, args, device):
        self.device = device
        self.args = args
        self.save_path = args.save_path
        self.epoch = 0
        self.adv_epoch = 0
        self.cur_val_accuracy = 0
        self.train_set = dataloaders.Dataloader(args.train_x_path, args.train_y_path)
        self.limit = self.train_set.__len__()
        val_set = dataloaders.Dataloader(args.val_x_path, args.val_y_path)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

        # Create model, optimizer and scheduler
        # self.model = models.WRN(depth=32, width=10, num_classes=10)
        self.model = models.Dense(args.input_dim, args.output_dim)
        # self.model = torch.nn.DataParallel(self.model, )
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

        if self.args.model_load_pth is not None:
            self._load_from_checkpoint(self.args.model_load_pth)
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
        cudnn.benchmark = True

        constraint = np.load(args.constraint_path)
        self.attacker = PGDAttacker(args.attack_eps, args.protected_attribs, constraint)

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()
    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch']
        print('Model loaded successfully')

    def _save_checkpoint(self, epoch):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        torch.save(model_data, osp.join(self.save_path, f'checkpoint_{epoch}.pth'))

    def train(self):
        adv_num = 0
        losses = utils.AverageMeter()
        x_set = self.train_set.train_x()
        y_set = self.train_set.train_y()
        if self.args.attack_mode == 'original':
            adv_flag = True
        elif self.args.attack_mode == 'adv':
            adv_flag = False
        elif self.args.attack_mode == 'ablation':
            adv_flag = True
        while self.adv_epoch < self.args.adv_epochs:
            self.args.seed += 1
            seed = self.args.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.shuffle(x_set)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.shuffle(y_set)
            self.args.seed += 1
            self.model.train()
            correct = 0
            total = 0
            start_time = time.time()
            start_index = 0
            batcher = Batcher(self.limit, self.args, self.device)

            for _ in range(self.limit // self.args.batch_size):
                cur_batch_adv_num = 0
                if adv_flag:
                    if self.args.attack_mode == 'adv' or 'ablation':
                        cur_batch_adv_num, end_index, x_batch, y_batch = batcher.next_batch(self.model, start_index,
                                                                                            x_set, y_set)
                    elif self.args.attack_mode == 'original':
                        cur_batch_adv_num, end_index, x_batch, y_batch = batcher.next_batch_original(self.model,
                                                                                                     start_index,
                                                                                                     x_set, y_set)
                else:
                    end_index, x_batch, y_batch = batcher.next_batch_unatt(start_index, x_set, y_set)
                input_adv = x_batch.to(self.device, non_blocking=True)
                target_adv = y_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(input_adv)
                loss = self.criterion(logits, target_adv)
                loss.backward()
                self.optimizer.step()
                _, pred = torch.max(logits, dim=1)
                correct += (pred == target_adv).sum()
                total += target_adv.size(0)
                losses.update(loss.data.item(), input_adv.size(0))
                start_index = end_index
                adv_num += cur_batch_adv_num
            print("Cur adv-ratio is {}".format(adv_num / self.train_set.__len__()))
            adv_num = 0  

            self.epoch += 1
            self.adv_epoch += adv_flag
            self.lr_scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time

            acc = (float(correct) / total) * 100
            message = 'Epoch {},Attacked? {}, Time {}, Loss: {}, Train Accuracy: {}'. \
                format(self.epoch, adv_flag, batch_time, loss.item(), acc)
            self._log(message)
            self._save_checkpoint(self.epoch)

            val_acc = self.eval()
            self._log('Natural validation accuracy: {}'.format(val_acc))
            self.cur_val_accuracy = self.cur_val_accuracy if val_acc < self.cur_val_accuracy else val_acc
            if (not adv_flag) and self.cur_val_accuracy > self.args.accuracy_threshold:
                adv_flag = True  

    def eval(self):
        self.model.eval()
        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.to(self.device, non_blocking=True)
            input = input.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy

    def eval_adversarial(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.to(self.device, non_blocking=True)
            input = input.to(self.device, non_blocking=True)
            input2 = input.clone()
            input, target = self.attacker.attack_htx(input, target, self.model, self.args.attack_steps,
                                                     self.args.attack_lr,
                                                     random_init=True)
            print("Does Adv work?{}".format(input.equal(input2)))

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy
