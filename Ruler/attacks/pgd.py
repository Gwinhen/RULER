# Code for PGD attacker
import time
import torch.nn.functional as F
import os
import numpy as np
import torch
os.chdir("..")


# 函数装饰器
TIME_LIMIT = 300
CUR_TIME = 0
GEN_INSTANCE = 0
GSR = 0
TRY_TIMES = 0
INSTANCE_LIMIT = 100000000
INSTANCE = []


def print_run_time(func):
    def wrapper(*args, **kw):
        global CUR_TIME, GEN_INSTANCE, TIME_LIMIT, TRY_TIMES, GSR
        local_time = time.time()
        flag, adv_sample = func(*args, **kw)
        cost = time.time() - local_time
        # print('current Function {} run time is {}'.format(func.__name__, cost))
        CUR_TIME += cost
        GEN_INSTANCE += flag
        if flag:
            INSTANCE.append(adv_sample.cpu().numpy().tolist())
        if CUR_TIME >= TIME_LIMIT:
            import sys
            print("We generated {} adv samples in {}s".format(GEN_INSTANCE, TIME_LIMIT))
            GSR = GEN_INSTANCE / TRY_TIMES
            print("The GSR is {}".format(GSR))
            np.save(r'INSTANCES/instance_seed.npy', np.array(INSTANCE))
            sys.exit(-1)
        if GEN_INSTANCE > INSTANCE_LIMIT:
            import sys
            print("{} adv samples cost {}s".format(INSTANCE_LIMIT, CUR_TIME))
            GSR = GEN_INSTANCE / TRY_TIMES
            print("The GSR is {}".format(GSR))
            np.save(r'INSTANCES/instance_seed.npy', np.array(INSTANCE))
            sys.exit(-1)
        return flag, adv_sample
    return wrapper


class PGDAttacker(object):
    def __init__(self, attack_eps, protected_attribs, constraint):
        self.attack_eps = attack_eps
        self.protected_attribs = protected_attribs
        self.constraint = constraint

    def compute_grad(self, x, y, net):
        x_back = torch.detach(x)
        x_back.requires_grad = True
        net.zero_grad()
        logits = net(x_back)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = x_back.grad.detach()
        grad = grad.sign()
        return grad

    def compute_ypred(self, x, net):
        x_back = torch.detach(x)
        x_back.requires_grad = True
        net.zero_grad()
        logits = net(x_back)
        _, y_pred = torch.max(logits, dim=1)
        x_back.detach()
        return y_pred

    def clip(self, x, attribs):
        for attrib in attribs:
            x[:, attrib] = torch.clamp(x[:, attrib], self.constraint[attrib, 0], self.constraint[attrib, 1])
        return x

    def attack_non_protected(self, x, net, y_true):
        """
        :param x:The sample passed from lasy phase
        :param net:The model
        :param y_true:The true label
        :return:if found, return True, adv sample;else return False, ori sample
        """
        global TRY_TIMES
        TRY_TIMES += 1
        non_protected_attribs = list(set(range(x.size()[1])).difference(set(self.protected_attribs)))
        grad = self.compute_grad(x, y_true, net)
        zeros = torch.zeros(x.size(), device=x.device)
        zeros[:, non_protected_attribs] = x[:, non_protected_attribs] + grad[:, non_protected_attribs]
        # print('before non_pro is \n{}'.format(x))
        x[:, non_protected_attribs] = 0
        x_adv = x + zeros
        x_adv = self.clip(x_adv, non_protected_attribs)
        y_pred = self.compute_ypred(x, net)

        # print('after non_pro \n{}'.format(x_adv))
        # print('grad is \n{}'.format(grad))
        # print(y_true != y_pred)
        if y_true != y_pred:
            return False, x_adv.detach()
        else:
            return True, x.detach()

    def attack_protected(self, x, net, y_true, attack_steps, attack_lr):
        """
        The attack_mode 1 means attack all protected attribs at the
        same time while the attack_mode 2 means the attack is one by one
        :param attack_lr: 1, we change the attribs at the extent of 1
        :param x:The ori sample
        :param net:The model
        :param y_true:The true label
        :param attack_steps: The times we attack
        :return: if found, return True, adv sample;else return False, ori sample
        """
        global TRY_TIMES
        x_adv_sample = x.clone()
        for step in range(attack_steps):
            TRY_TIMES += 1
            grad = self.compute_grad(x, y_true, net)
            # print("step is {}, grad is {}".format(step, grad))
            zeros = torch.zeros(x_adv_sample.size(), device=x_adv_sample.device)
            zeros[:, self.protected_attribs] = x_adv_sample[:, self.protected_attribs] + (
                    attack_lr * grad[:, self.protected_attribs])
            x_adv_sample[:, self.protected_attribs] = 0

            x_adv_attrib = x_adv_sample + zeros
            x_adv_sample = self.clip(x_adv_attrib, self.protected_attribs)
            # print('x_adv is {}'.format(x_adv_sample))
            y_pred = self.compute_ypred(x_adv_sample, net)
            if y_pred != y_true:
                return True, x_adv_sample.detach()
        return False, x_adv_sample.detach()

    # @print_run_time
    def attack_sample(self, x_ori_sample, y_sample, net, attack_steps, attack_lr):
        """
        attack single sample, protected attributes and unprotected attributes are both included
        :return: if adv-sample found, return(True, x_adv_sample),else (False, x_ori_sample)
        """
        # print('x_ori is {}\n'.format(x_ori_sample))
        flag, x_adv_sample = self.attack_protected(x_ori_sample, net, y_sample, attack_steps, attack_lr)
        if flag:
            return flag, x_adv_sample
        flag, x_adv_sample = self.attack_non_protected(x_adv_sample, net, y_sample)
        # import sys
        # sys.exit(-1)
        if flag:
            return flag, x_adv_sample
        else:
            return flag, x_ori_sample

    def attack_original(self, x_ori_batch, y_batch, net):
        # print(x_ori_batch.shape)
        x_adv_sample = x_ori_batch.clone()
        grad = self.compute_grad(x_ori_batch, y_batch, net)
        # print("grad is {}".format(grad))
        x_adv_attrib = x_adv_sample + grad
        # print('shape is {}'.format(x_ori_sample.shape[1])) int 12
        x_adv_sample = self.clip(x_adv_attrib, range(0, x_ori_batch.shape[1]))
        y_pred = self.compute_ypred(x_adv_sample, net)
        return torch.sum(y_pred == y_batch), x_adv_sample.detach()


if __name__ == "__main__":
    '''test for our pgd attacker'''
    ckpt_path = r'results/adult/checkpoint_1.pth'
    import torch
    from models import Dense

    print('Loading model from {} ...'.format(ckpt_path))
    model_data = torch.load(ckpt_path, map_location='cuda:0')
    net = Dense(12, 2)
    net.load_state_dict(model_data['model'])
    print('Model loaded successfully')
    import numpy as np
    pgd = PGDAttacker(attack_eps=8, protected_attribs=[6],
                      constraint=np.load(r'data/PGD_dataset/adult/constraint.npy'))
    x_ori = torch.from_numpy(np.load(r'data/EIDIG&ADF/seeds/seeds_adult.npy'))
    x_ori = torch.unsqueeze(x_ori, 0)
    x_ori = x_ori.to(torch.float32)
    print(x_ori.shape)
    for i in range(x_ori.shape[1]):
        sample = x_ori[:, i, :]
        y_ori = pgd.compute_ypred(sample, net)
        pgd.attack_sample(sample, y_ori, net, 8, 1)
    np.save(r'INSTANCES/instance_seed.npy', np.array(INSTANCE).reshape(-1, 12))
