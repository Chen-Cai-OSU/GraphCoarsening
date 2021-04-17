# Created at 2020-05-18
# Summary: util functions to monitor training

import functools

import numpy as np
import torch
import torch.nn.functional as F

from sparsenet.util.util import banner, timefunc, summary


@timefunc
def check_laplacian(L, step, eps=1e-8):
    """ check the whether laplacian is symmetric during training.
        check there is no nan in laplacian
        :param L: output of get_laplacian_mat. torch.sparse.tensor
        :param step: iteration number
        :param eps: difference allowed for two float number considered as the same
    """

    # check if there is nan in the tensor
    Ltypes = (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)
    assert isinstance(L, Ltypes), 'Input laplacian is not sparse tensor'
    nan_check = torch.isnan(L._values())
    nan_cnt = nan_check.nonzero().shape[0]
    if nan_cnt != 0:
        u, v = L._indices()[:, nan_check.nonzero()[0]]
        u, v = u.item(), v.item()
        exit(f'Laplacian at step {step} has {nan_cnt} nan values, e.g., L({u}, {v}) = Nan.')

    # dont want to convert to dense. manual implement.
    indices, values, sym_check = L._indices(), L._values(), {}
    for i in range(indices.shape[1]):
        u, v = indices[:, i]
        u, v = u.item(), v.item()
        sym_check[(u, v)] = values[i].item()
    for i in range(indices.shape[1]):
        u, v = indices[:, i]
        u, v = u.item(), v.item()
        if (v, u) not in sym_check and abs(sym_check[(u, v)]) > eps:
            exit(f'Laplacian at step {step} is not symmetric... on ({u}, {v}), with L({u}, {v})={sym_check[(u, v)]}'
                 f' but L({v}, {u})=0.')
        if abs(sym_check[(u, v)] - sym_check[(v, u)]) > eps:
            exit(f'Laplacian at step {step} is not symmetric... on ({u}, {v}), with L({u}, {v})={sym_check[(u, v)]}'
                 f' but L({v}, {u})={sym_check[(v, u)]}.')

    print(f'Laplacian at step {step} is normal!')


class monitor():
    def __init__(self):
        pass

    @staticmethod
    def data_monitor(train_data, sub, args):
        banner('Train data')
        for (k, v) in train_data[:args.bs]:
            print(k, v, sub.g_sml.edge_index[:, k])
        print()

    @staticmethod
    def train_data_monitor(train_data, args):
        banner('Train_data first check')
        for i, (k, v) in enumerate(train_data[:args.bs]):
            if i > 5: break
            print(k, v)

    def train_monitor(self, pred, edge_weight_sml):
        summary(pred, 'pred')
        summary(edge_weight_sml, 'edge_weight_sml')


def no_grad_func(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return new_func


@no_grad_func
def monitor_param_saturation(model):
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


if __name__ == '__main__':
    # banner('This one is sym and has no nan!')
    # i = torch.LongTensor([[0, 1, 2], [0, 1, 2]])
    # v = torch.FloatTensor([3, 4, 5])
    # s1 = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
    # check_laplacian(s1, 1)

    # banner('This one is not symmetric!')
    # i = torch.LongTensor([[0, 1, 1],
    #                       [2, 0, 2]])
    # v = torch.FloatTensor([3, 4, 5])
    # s2 = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
    # check_laplacian(s2, 1)

    banner('This one has nan value!')
    i = torch.LongTensor([[0, 1, 2], [0, 1, 2]])
    v = torch.FloatTensor([3, 4, np.nan])
    s3 = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
    check_laplacian(s3, 1)
