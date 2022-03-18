import argparse
import os

import sys

sys.path.append(os.getcwd())
from evaluator import evaluator
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='adv based model, evaluator')
    parser.add_argument('--dataset', default='adult', type=str,
                        help='training data of dataset')
    parser.add_argument('--train_x_path',
                        help='path to load training x data')
    parser.add_argument('--test_x_path',
                        help='path to load test x data')
    parser.add_argument('--test_y_path',
                        help='path to load test y data')
    parser.add_argument('--constraint_path',
                        help='path to load constraint of dataset')
    parser.add_argument('--model_start', default=1, type=int,
                        help='the first index of model to evaluate')
    parser.add_argument('--model_end', default=100, type=int,
                        help='the last model we evaluate')
    parser.add_argument('--mode', default='test-acc', type=str,
                        help='metrics is test-acc | unfairness | aod | spd | DP | DPR')
    parser.add_argument('--path', default='log', type=str,
                        help='outpath of the log file, path is a directory.')
    parser.add_argument('--model_path', default='results/adult', type=str,
                        help='the path our model restored,a directory, like results/compas/train.')
    parser.add_argument('--sample_round', default=10, type=int,
                        help='we test sample_round times.')
    parser.add_argument('--num_gen', default=100, type=int,
                        help='for each ample_round we generate num_gen samples.')
    parser.add_argument('--protected_attribs', default=1, type=int,
                        help='the index of a given dataset, used to compute metrics(not including unfairness).')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='which gpu we used to deploy our model')
    args = parser.parse_args()
    return args


def update_args(args):
    args.train_x_path = "data/PGD_dataset/{}/x_train.npy".format(args.dataset)
    args.train_y_path = "data/PGD_dataset/{}/y_train.npy".format(args.dataset)
    args.test_x_path = "data/PGD_dataset/{}/x_test.npy".format(args.dataset)
    args.test_y_path = "data/PGD_dataset/{}/y_test.npy".format(args.dataset)
    args.constraint_path = "data/PGD_dataset/{}/constraint.npy".format(args.dataset)
    return args


if __name__ == "__main__":
    args = parse_args()
    args = update_args(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(args)
    evaluator = evaluator(mode=args.mode, args=args,
                          model_list=range(args.model_start, args.model_end + 1),
                          device=device, path=args.path)

