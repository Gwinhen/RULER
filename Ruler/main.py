import argparse
import json
import numpy as np
from trainers import Trainer
import utils_yuc as utils
import os
import pathlib

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
    parser.add_argument('--cfg_path', default='config.json', type=str,
                        help='path to config file')
    parser.add_argument('--data_root', default='data', type=str,
                        help='path to dataset')
    parser.add_argument('--alg', default='train', type=str,
                        help='Algorithm to train | compas | adult | bank | german')
    parser.add_argument('--save_path',  type=str,
                        help='path to save model and log file')
    parser.add_argument('--attack_steps',type=int,
                        help='number of attack iterations (PGD-n)')
    parser.add_argument('--adv_epochs',  type=int,
                        help='the epoch we will train through adversarial')
    parser.add_argument('--mode', default='train', type=str,
                        help='mode to use| Can be train, eval, vis')
    parser.add_argument('--adv_ratio', default=None, type=float,
                        help="the number of adv_sample's up bound")
    parser.add_argument('--dataset', default=None, type=str,
                        help='the dataset where we deploy our approach')
    parser.add_argument('--accuracy_threshold', default=None, type=float,
                        help='adv-training will begin when val-accuracy surpass this threshold')
    parser.add_argument('--model_load_pth', default=None, type=str,
                        help='If we want to train based on certain model, it will be the path.')
    parser.add_argument('--protected_attribs', nargs='+', type=int,
                        help='input should be 6 or 6 10 like.')
    parser.add_argument('--gpu_id',  type=int,
                        help='which gpu we used to deploy our model')
    parser.add_argument('--attack_mode', default='adv', type=str,
                        help='attack_mode is original | adv')
    parser.add_argument('--lr', default=0.003, type=float,
                        help='the learning rate of our model')
    args = parser.parse_args()
    return args


def update_dataset_pth(cfgs):
    dataset = cfgs["dataset"]
    cfgs["val_x_path"] = "data/PGD_dataset/{}/x_val.npy".format(dataset)
    cfgs["val_y_path"] = "data/PGD_dataset/{}/y_val.npy".format(dataset)
    cfgs["train_x_path"] = "data/PGD_dataset/{}/x_train.npy".format(dataset)
    cfgs["train_y_path"] = "data/PGD_dataset/{}/y_train.npy".format(dataset)
    cfgs["test_x_path"] = "data/PGD_dataset/{}/x_test.npy".format(dataset)
    cfgs["test_y_path"] = "data/PGD_dataset/{}/y_test.npy".format(dataset)
    cfgs["constraint_path"] = "data/PGD_dataset/{}/constraint.npy".format(dataset)
    # cfgs["protected_attribs"] = np.load("data/PGD_dataset/{}/protected_attribs.npy".format(dataset))
    return cfgs


def main(args):
    # Read configs
    with open(args.cfg_path, "r") as fp:
        configs = json.load(fp)
    # print(configs)

    # Update the configs based on command line args
    arg_dict = vars(args)
    # print(arg_dict)
    for key in arg_dict:
        # if key in configs:
        if arg_dict[key] is not None:
            configs[key] = arg_dict[key]
    configs = update_dataset_pth(configs)
    configs['input_dim'] = np.load(configs['train_x_path']).shape[1]
    configs['output_dim'] = len(set(np.load(configs['train_y_path'])))
    device = torch.device(f"cuda:{configs['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(configs)
    configs = utils.ConfigMapper(configs)
    configs.attack_eps = float(configs.attack_eps) / 255

    configs.alg = args.alg
    configs.save_path = os.path.join(configs.save_path, configs.alg)
    pathlib.Path(configs.save_path).mkdir(parents=True, exist_ok=True)
    if configs.mode == 'train':
        trainer = Trainer(configs, device)
        trainer.train()
    else:
        raise ValueError('mode should be train, eval or vis')
        

if __name__ == '__main__':
    os.chdir('./Ruler')
    args = parse_args()
    main(args)

