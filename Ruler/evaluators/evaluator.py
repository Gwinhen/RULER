import numpy as np
import sys

import os

import torch

# print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'evaluators'))
from evaluate_metrics import Metrics
from evaluate_fairness import measure_discrimination

import models


class evaluator:
    def __init__(self, mode, args, model_list, device, path):
        """
        :param mode: test-acc, spd, unfairness, aod , type is str
        :param args: arguments from command line
        :param model_list : iterable, eg, range(1, 3)
        :param device : cpu or cuda
        :param path : the path directory we output our log file
        """
        self.num_gen = args.num_gen
        self.sample_round = args.sample_round
        self.model_list = model_list
        self.device = device
        self.path = path
        self.args = args
        self.model_path = args.model_path
        self.dataset = args.dataset
        self.constraint = np.load(args.constraint_path)
        self.input_len = np.load(args.train_x_path).shape[1]
        self.output_len = len(set(np.load(args.train_y_path)))
        if mode == 'unfairness':
            self.evaluation_unfairness()
        elif mode == 'test-acc':
            self.x_test = np.load(args.test_x_path)
            self.y_test = np.load(args.test_y_path)
            self.evaluation_nat_test_acc(args.dataset)
        elif mode == 'metrics':
            self._compute_metrics()
        else:
            raise ValueError("No mode is {}.Mode must be spd, aod, test-acc , dp, dpr, eo or unfairness"
                             .format(mode))

    def evaluation_unfairness(self):
        model = models.Dense(self.input_len, self.output_len)
        for model_index in self.model_list:
            ckpt = '{}/checkpoint_{}.pth'.format(self.model_path, model_index)
            print("\nThe current model is epoch {}".format(model_index))
            model.load_state_dict(torch.load(ckpt, map_location=self.device)['model'])
            measure_discrimination(self.sample_round, self.num_gen, self.input_len, model, self.constraint,
                                   self.dataset)

    def evaluation_nat_test_acc(self, dataset):
        model = models.Dense(self.input_len, self.output_len)
        x_test = torch.from_numpy(np.load('data/PGD_dataset/{}/x_test.npy'.format(dataset)).astype('float32'))
        y_test = torch.from_numpy(np.load('data/PGD_dataset/{}/y_test.npy'.format(dataset)).astype('float32'))
        for model_index in self.model_list:
            ckpt = '{}/checkpoint_{}.pth'.format(self.args.model_path, model_index)
            model.load_state_dict(torch.load(ckpt, map_location=self.device)['model'])
            with torch.no_grad():
                output = model(x_test)
                _, pred = torch.max(output, dim=1)
            correct = (pred == y_test).sum()
            accuracy = correct / len(y_test)
            msg = "Current epoch {}'s accuracy is {}".format(model_index, accuracy)
            print(msg)
            self._log(self.path, msg, 'test_acc')
        # if dataset == 'german':
        #     evaluate_german(self.device, model_path=self.args.model_path, path=self.path, _log=self._log, model_list=self.model_list)
        # elif dataset == 'adult':
        #     evaluate_adult(self.device, model_path=self.args.model_path, path=self.path, _log=self._log, model_list=self.model_list)
        # elif dataset == 'bank':
        #     evaluate_bank(self.device, model_path=self.args.model_path, path=self.path, _log=self._log, model_list=self.model_list)
        # elif dataset == 'compas':
        #     evaluate_compas(self.device, model_path=self.args.model_path, path=self.path, _log=self._log, model_list=self.model_list)
        # else:
        #     raise ValueError("No dataset is {}".format(dataset))
        return

    def _compute_metrics(self):
        model = models.Dense(self.input_len, self.output_len)
        x_test = torch.from_numpy(np.load(self.args.train_x_path).astype('float32'))
        y_true = torch.from_numpy(np.load(self.args.train_y_path).astype('float32'))
        for model_index in self.model_list:
            ckpt = '{}/checkpoint_{}.pth'.format(self.model_path, model_index)
            print("\nThe current model is epoch {}".format(model_index))
            model.load_state_dict(torch.load(ckpt, map_location=self.device)['model'])
            with torch.no_grad():
                output = model(x_test)
                _, y_pred = torch.max(output, dim=1)
            metrics = Metrics(x_true=x_test, y_true=y_true.reshape(-1, 1), y_pred=y_pred.reshape(-1, 1),
                              favorable_class=self._get_favorable_class(),
                              privileged_group=self._get_privileged_group(),
                              protected_attribs=self._get_protected_attribs()).get_metrics()
            print('SPD is {}'.format(metrics['spd']))
            print('AOD is {}'.format(metrics['aod']))
            print('E O is {}'.format(metrics['eo']))
            print('D P is {}'.format(metrics['dp']))
            print('DPR is {}'.format(metrics['dpr']))
        return

    def _log(self, path, msg, mode):
        with open('{}/{}.log'.format(path, mode), 'a') as f:
            f.write('{}\n'.format(msg))

    def _get_favorable_class(self):
        """:return int, favorable value of a given dataset"""
        if self.dataset == 'compas':
            return 0  # 0 means low recidivism rate, (as the same as Fairea)
        elif self.dataset == 'adult':
            return 1  # 1 means income >50k
        elif self.dataset == 'german':
            pass

    def _get_privileged_group(self):
        """:return int, the privileged value of protected_attribs"""
        adult = {6: 0, 7: 1}  # means {6(race): 0(white), 7(gender): 1(male)}
        german = {}  # 
        compas = {2: 1}  # means {2(race) :1(Caucasian)}
        if self.dataset == 'adult':
            return adult[self._get_protected_attribs()]
        elif self.dataset == 'german':
            return german[self._get_protected_attribs()]
        elif self.dataset == 'compas':
            return compas[self._get_protected_attribs()]
        else:
            raise ValueError('Dataset {} has no privileged_group'.format(self.dataset))

    def _get_protected_attribs(self):
        return self.args.protected_attribs
