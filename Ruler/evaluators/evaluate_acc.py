import os

import torch
import numpy as np
import generation_utilities
import models


def evaluate_adult(device, model_path, path, _log, model_list):
    """
    :param device: which device
    :param model_path: we'll load model from this path
    :param path: where to write our log
    :param _log: function to output
    :param model_list: iterable, contains the model's index
    :return
    """
    input_len = 12
    model = models.Dense(input_len)
    x_test = torch.from_numpy(np.load('data/PGD_dataset/adult/x_test.npy').astype('float32'))
    y_test = torch.from_numpy(np.load('data/PGD_dataset/adult/y_test.npy').astype('float32'))
    for model_index in model_list:
        ckpt = '{}/checkpoint_{}.pth'.format(model_path, model_index)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        with torch.no_grad():
            output = model(x_test)
            _, pred = torch.max(output, dim=1)
        correct = (pred == y_test).sum()
        accuracy = correct / len(y_test)
        msg = "Current epoch {}'s accuracy is {}".format(model_index, accuracy)
        print(msg)
        _log(path, msg, 'test_acc')
    return


def evaluate_bank(device, model_path, path, _log, model_list):
    """
        :param device: which device
        :param model_path: we'll load model from this path
        :param path: where to write our log
        :param _log: function to output
        :param model_list: iterable, contains the model's index
        :return
    """
    input_len = 16
    model = models.Dense(input_len)
    x_test = torch.from_numpy(np.load('data/PGD_dataset/bank/x_test.npy').astype('float32'))
    y_test = torch.from_numpy(np.load('data/PGD_dataset/bank/y_test.npy').astype('float32'))
    for model_index in model_list:
        ckpt = '{}/checkpoint_{}.pth'.format(model_path, model_index)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        with torch.no_grad():
            output = model(x_test)
            _, pred = torch.max(output, dim=1)
        correct = (pred == y_test).sum()
        accuracy = correct / len(y_test)
        msg = "Current epoch {}'s accuracy is {}".format(model_index, accuracy)
        print(msg)
        _log(path, msg, 'test_acc')
    return


def evaluate_german(device, model_path, path, _log, model_list):
    """
        :param device: which device
        :param model_path: we'll load model from this path
        :param path: where to write our log
        :param _log: function to output
        :param model_list: iterable, contains the model's index
        :return
    """
    input_len = 24
    model = models.Dense(input_len)
    x_test = torch.from_numpy(np.load('data/PGD_dataset/german/x_test.npy').astype('float32'))
    y_test = torch.from_numpy(np.load('data/PGD_dataset/german/y_test.npy').astype('float32'))
    for model_index in model_list:
        ckpt = '{}/checkpoint_{}.pth'.format(model_path, model_index)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        with torch.no_grad():
            output = model(x_test)
            _, pred = torch.max(output, dim=1)
        correct = (pred == y_test).sum()
        accuracy = correct / len(y_test)
        msg = "Current epoch {}'s accuracy is {}".format(model_index, accuracy)
        print(msg)
        _log(path, msg, 'test_acc')
    return


def evaluate_compas(device, model_path, path, _log, model_list):
    """
        :param device: which device
        :param model_path: we'll load model from this path
        :param path: where to write our log
        :param _log: function to output
        :param model_list: iterable, contains the model's index
        :return
    """
    input_len = 12
    model = models.Dense(input_len)
    x_test = torch.from_numpy(np.load('data/PGD_dataset/compas/x_test.npy').astype('float32'))
    y_test = torch.from_numpy(np.load('data/PGD_dataset/compas/y_test.npy').astype('float32'))
    for model_index in model_list:
        ckpt = '{}/checkpoint_{}.pth'.format(model_path, model_index)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        with torch.no_grad():
            output = model(x_test)
            _, pred = torch.max(output, dim=1)
        correct = (pred == y_test).sum()
        accuracy = correct / len(y_test)
        msg = "Current epoch {}'s accuracy is {}".format(model_index, accuracy)
        print(msg)
        _log(path, msg, 'test_acc')
    return


def evaluate_common_acc(model, x_test, y_test):
    """
    :param model: a torch model loaded from a pth file
    :param x_test: npy
    :param y_test: npy
    :return: accuracy
    """
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    with torch.no_grad():
        output = model(x_test)
        _, pred = torch.max(output, dim=1)
    correct = (pred == y_test).sum()
    accuracy = correct / len(y_test)
    return accuracy
