# Created at 2020-06-13
# Summary: torch related functions
import torch
import networkx as nx

from sparsenet.util.util import summary, tonp
import scipy as sp
from time import time
from sparsenet.util.util import pf
from functools import partial
from scipy.sparse import csr_matrix

def complexity(method, n=[1,10,100]):

    def timed(*args, **kw):
        for _n in n:
            ts = time()
            print(args, kw)
            kw['n'] = _n
            print(kw)
            result = method(*args, **kw)
            te = time()
            print(f'n: {_n}. t: {te-ts}')
        return result
    return timed

from scipy.sparse import coo_matrix
import numpy as np

def sparse_tensor2_sparse_numpyarray(sparse_tensor):
    """
    :param sparse_tensor: a COO torch.sparse.FloatTensor
    :return: a scipy.sparse.coo_matrix
    """
    if sparse_tensor.device.type == 'cuda':
        sparse_tensor = sparse_tensor.to('cpu')

    values = sparse_tensor._values().numpy()
    indices = sparse_tensor._indices()
    rows, cols = indices[0, :].numpy(), indices[1, :].numpy()
    size = sparse_tensor.size()
    scipy_sparse_mat = coo_matrix((values, (rows, cols)), shape=size, dtype=np.float)
    return scipy_sparse_mat

def sparse_matrix2sparse_tensor(ret, dev='cpu'):
    # coo sparse matrix to sparse tensor
    # https://bit.ly/30DI2u8
    values = ret.data
    indices = np.vstack((ret.row, ret.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = ret.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(dev)



# @partial(complexity, n=[1,2,3,4,5])
def sparse_mm(L, Q):
    """
    :param L: a sparse tensor
    :param Q: a sparse diagonal tensor
    :return: Q.L.Q
    """
    dev = L.device
    if dev == 'cuda':
        L = L.to('cpu')
        Q = Q.to('cpu')

    L = sparse_tensor2_sparse_numpyarray(L) # csr_matrix(L)
    Q = sparse_tensor2_sparse_numpyarray(Q) # csr_matrix(Q)

    ret = coo_matrix(Q.dot(L.dot(Q))) # coo matrix sparse
    return sparse_matrix2sparse_tensor(ret, dev=dev)

def sparse_mm2(P, D1, D2):
    """
    :param P: a sparse tensor of (n, N)
    :param D1: a sparse diagonal tensor of (N, N)
    :param D2: a sparse diagonal tensor of (n, n)
    :return: D1.P.D2 also a sparse tensor
    """

    dev = P.device
    if dev == 'cuda':
        P, D1, D2 = P.to('cpu'), D1.to('cpu'), D2.to('cpu')
    P = sparse_tensor2_sparse_numpyarray(P)
    D1 = sparse_tensor2_sparse_numpyarray(D1)
    D2 = sparse_tensor2_sparse_numpyarray(D2)
    try:
        ret = coo_matrix(D2.dot(P.dot(D1)))
    except ValueError:
        summary(P.todense(), 'P')
        summary(D1.todense(), 'D1')
        summary(D2.todense(), 'D2')
        exit()
    return sparse_matrix2sparse_tensor(ret, dev=dev)

def mm(n=10):
    g = nx.random_geometric_graph(n, 0.1)
    L = nx.laplacian_matrix(g).todense()
    L = torch.Tensor(L)
    Q = torch.diag(torch.rand(n))

    summary(L, 'L')
    summary(Q, 'Q')

    # method 1
    t0 = time()
    ret1 = sp.sparse.csr_matrix(L).dot(sp.sparse.csr_matrix(Q))
    ret1 = sp.sparse.csr_matrix(Q).dot(ret1)
    summary(ret1, 'ret1')
    t1 = time()
    print(f'method 1: {pf(t1 - t0, 2)}')

    # ret 2
    ret2 = tonp(Q).dot(tonp(L).dot(tonp(Q)))
    summary(ret2, 'ret2')
    t2 = time()
    print(f'method 2: {pf(t2-t1, 2)}')

    assert (ret2-ret1 ==0).all()
    # summary(tonp(tonp(ret2) - tonp(ret1.todense())), 'ret2-ret1')

if __name__ == '__main__':
    n = 50 #000
    g = nx.random_geometric_graph(n, 0.01)
    L = nx.laplacian_matrix(g)
    L = torch.Tensor(L)
    print(L)
    exit()

    Q = torch.diag(torch.rand(n))

    L, Q = L.to_sparse(), Q.to_sparse()
    ret = sparse_mm(L, Q)
    summary(ret, 'ret')

    exit()
    mm(n=1000)
