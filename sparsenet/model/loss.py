# Created at 2020-04-17
# Summary: implement a loss function based on || x.T * L * x - x'.T * L_{sparse} * x' ||

import math
from functools import partial

import numpy as np
import torch
import torch_geometric
from scipy.sparse import csc_matrix
from torch_geometric.utils import get_laplacian, to_networkx, from_networkx

# convert the assignment to the projection mat, so we don't need to do it every time when we compute loss.
# n, r the size of L and L_sparse.
from sparsenet.util.sample import sample_N2Nlandmarks
from sparsenet.util.util import random_pygeo_graph, summary, fix_seed, banner, timefunc, pf

fix_seed()

tf = partial(timefunc, threshold=-1)


@timefunc
def get_projection_mat(n, r, Assignment):
    '''
    :param n: Size of original graph
    :param r: Size of sampled graph
    :param Assignment: The correspondence matrix returned from sample_N2Nlandmarks.
    :return: The projection matrix of (r, n).
    '''
    P = np.zeros((r, n))
    for key, value in Assignment.items():
        s = len(value)
        assert s != 0
        for v in value:
            P[key][v] = 1 / s  # important
    return torch.FloatTensor(P)


def get_sparse_projection_mat(n, r, Assignment):
    '''
        :param n: Size of original graph
        :param r: Size of sampled graph
        :param Assignment: The correspondence matrix returned from sample_N2Nlandmarks.
        :return: The projection matrix of size (r, n).
    '''
    index, val = [], []
    for key, value in Assignment.items():
        s = len(value)
        assert s != 0
        for v in value:
            index.append([key, v])
        val = val + [1 / s] * s
    i, v = torch.tensor(index).T, torch.tensor(val)
    return torch.sparse.FloatTensor(i, v, torch.Size([r, n]))


def get_sparse_C(n, r, Assignment):
    '''
        :param n: Size of original graph
        :param r: Size of sampled graph
        :param Assignment: The correspondence matrix returned from sample_N2Nlandmarks. (# todo: not really matrix but a dict)
                            key is the node for small graph, value is the set of nodes in big graph contracted to the smaller graph
        :return: The sparse c matrix (csc) of size (r, n).
    '''
    row, col, val = [], [], []
    for key, value in Assignment.items():
        s = len(value)
        assert s != 0
        row.extend([key] * s)
        col.extend(list(value))
        val = val + [1 / np.sqrt(s)] * s  # the major differeence

    row, col = np.array(row), np.array(col)
    data = np.array(val)
    return csc_matrix((data, (row, col)), shape=(r, n))
    # return torch.sparse.FloatTensor(i, v, torch.Size([r, n]))


def random_vec_loss(L, L_sparse, Projection, device='cpu', num_vec=None, debug=False):
    '''
    :param L: L is a n*n sparse Tensor.
    :param L_sparse: a r*r sparse Tensor
    :param Projection: The projection tensor (r * n)
    :param device: run on cpu or gpu
    :param num_vec: num of random vectors sampled for computing loss
    :param debug: debug mode. Will get removed later.
    :return: The loss  X.T L X - X.T Proj.T L_sparse Proj X, where X is the mat of concating random vecs.
    '''

    # todo: add more variety of random vector (loss freq/high freq)
    # todo: need to test for large L, the speed difference on cpu vs. gpu

    L = L.to(device)
    L_sparse = L_sparse.to(device)

    if debug:
        print('L', L)
        print('L_sparse', L_sparse)

    n = (Projection.shape[1])
    if num_vec == None:
        num_vec = max(1, int(math.log(n)))

    X = torch.rand(n, num_vec) - 0.5
    Projection = Projection.to(device)

    X = X / ((X ** 2).sum(0, keepdim=True)).sqrt()
    X = X.to(device)

    X_prime = torch.mm(Projection, X)
    quadL = torch.mm(X.t(), torch.sparse.mm(L, X))
    qualL_sparse = torch.mm(X_prime.t(), torch.sparse.mm(L_sparse, X_prime))
    loss = torch.sum(torch.abs(quadL - qualL_sparse))  # important: this is wrong!
    return loss


# @tf
def get_laplacian_mat(edge_index, edge_weight, num_node, normalization='sym'):  # todo: change back
    """ return a laplacian (torch.sparse.tensor)"""
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                            normalization=normalization)  # see https://bit.ly/3c70FJK for format
    return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_node, num_node]))


@tf
def energy_loss(L1, L2, assignment, device='cuda', test=False,
                n_measure=1, num_vec=None):
    """
    :param g1: pygeo graph
    :param g2: pygeo graph (smaller)
    :param assignment: a dict where key is the node in smaller graph and value is the nodes in larger graph
    :param n_measure
    :return:
    """

    if test:
        assert isinstance(g1, torch_geometric.data.data.Data)
        assert isinstance(g2, torch_geometric.data.data.Data)

        L1 = get_laplacian_mat(g1.edge_index, g1.edge_weight, g1.num_nodes)
        L2 = get_laplacian_mat(g2.edge_index, g2.edge_weight, g2.num_nodes)

    assert isinstance(L1, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)), summary(L1, 'L1')
    assert isinstance(L2, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)), summary(L2, 'L2')

    Projection = get_projection_mat(L1.shape[0], L2.shape[0], assignment)

    if n_measure == 1:
        loss = random_vec_loss(L1, L2, Projection, device=device, num_vec=num_vec)
        return loss
    else:
        losses = []
        for _ in range(n_measure):
            loss = random_vec_loss(L1, L2, Projection, device=device, num_vec=num_vec)
            losses.append(np.float(loss))
        mean, std = np.mean(losses), np.std(losses)
        return f'{pf(mean, 2)}±{pf(std, 2)}'


if __name__ == '__main__':
    # undirected 4-path
    banner('random_vec_loss test')
    L = get_laplacian_mat(torch.LongTensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]]),
                          torch.FloatTensor([1., 1., 1., 1., 1., 1.]), 4)
    # undirected 2-path (link)
    L_sparse = get_laplacian_mat(torch.LongTensor([[0, 1], [1, 0]]), torch.FloatTensor([1., 1.]), 2)
    Projection = get_projection_mat(L.shape[0], L_sparse.shape[0], {0: set([0, 1]), 1: set([2, 3])})

    losses = []
    for _ in range(1000):
        loss = random_vec_loss(L, L_sparse, Projection)
        losses.append(loss)
    summary(np.array(losses), 'losses')
    exit()

    banner('sample_N2Nlandmarks test')
    n_node, n_edge = 10, 40
    node_feat_dim, edge_feat_dim = 1, 1
    n_node_small, n_edge_small = 5, 20

    g1 = random_pygeo_graph(n_node, node_feat_dim, n_edge, edge_feat_dim)
    g1.edge_weight = 1.1 * torch.ones(n_edge)

    g2, assignment = sample_N2Nlandmarks(to_networkx(g1), n_node_small, weight_key='edge_weight')
    g2 = from_networkx(g2)
    g2.edge_weight = g2.edge_weight.type(torch.float)

    summary(g1, 'g1')
    summary(g2, 'g2')
    # exit()

    loss = energy_loss(g1, g2, assignment, device='cpu', test=True)
    print(loss)

    exit()

    print(loss)
