# Created at 2020-06-11
# Summary: cut, conductance related

from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import degree, to_networkx

from sparsenet.util.util import summary, timefunc, fix_seed, pf, random_pygeo_graph


def normalized_cut(edge_index, edge_attr, num_nodes: Optional[int] = None):
    r"""Computes the normalized cut :math:`\mathbf{e}_{i,j} \cdot
    \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)` of a weighted graph
    given by edge indices and edge attributes.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor): Edge weights or multi-dimensional edge features.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    row, col = edge_index[0], edge_index[1]
    deg = 1. / degree(col, num_nodes, edge_attr.dtype)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut


def _set(row, s, dev='cuda'):
    # t0 = time()
    i = torch.nonzero(row[..., None] == s)[:, 0]
    # t1 = time()
    # print(pf(t1-t0, 2))
    row_s = torch.zeros(row.size())
    row_s[i] = 1
    return row_s.type(torch.int8).to(dev)


@timefunc
def pyG_conductance(edge_index, edge_attr, s, t=None, dev='cuda', verbose=False):
    """
    :param edge_index:
    :param edge_attr:
    :param s: a list or a tensor
    :param t: a list or a tensor
    :return: conductance (tensor)
    """

    if t is None:
        _t = None
        tmp = torch.unique(edge_index).tolist()
        t = list(set(tmp) - set(s))

    s, t = torch.tensor(s).to(dev), torch.tensor(t).to(dev)
    edge_index, edge_attr = edge_index.to(dev), edge_attr.to(dev)
    row, col = edge_index[0], edge_index[1]
    del edge_index

    # row_s = torch.sum(row[..., None] == s, axis=1) # memory intensive
    row_s = _set(row, s, dev=dev)
    # col_s = torch.sum(col[..., None] == s, axis=1)
    col_s = _set(col, s, dev=dev)
    # summary(row_s - row_s_, 'row_s - row_s_')
    # summary(col_s - col_s_, 'col_s - col_s_')

    vol_s = torch.sum(torch.mul(edge_attr, row_s + col_s))

    # row_t = torch.sum(row[..., None] == t, axis=1)
    row_t = _set(row, t, dev=dev) if _t is not None else (1 - row_s).to(dev)
    # col_t = torch.sum(col[..., None] == t, axis=1)
    col_t = _set(col, t, dev=dev) if _t is not None else (1 - col_s).to(dev)
    vol_t = torch.sum(torch.mul(edge_attr, row_t + col_t))

    indices = torch.nonzero((row_s & col_t) | (row_t & col_s))
    cut = torch.sum(edge_attr[indices])

    # print(f'cut: {cut}. vol_s: {vol_s}. vol_t: {vol_t}')
    if verbose:
        print(f'cut: {cut}. vol_s: {vol_s}. vol_t: {vol_t}, conductance: {cut / max(1, min(vol_s, vol_t))}')
    return cut / max(1, min(vol_s, vol_t))  # make sure it's at least 1. This is needed for large reduction ratio.


import argparse

parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--dataset', type=str, default='ws', help='dataset for egographs')
parser.add_argument('--sample', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed()
    n_node, n_edge = 320, 5000
    N = 1
    idx = 0

    # kwargs = {'dataset': args.dataset, 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
    # pyGs = EgoGraphs(**kwargs)
    # pyGs =  syth_graphs(type=args.dataset, n=2, size=1000) #
    pyG = random_pygeo_graph(n_node, 1, n_edge, 1)
    pyGs = [pyG] * 5

    for pyG in pyGs[:1]:
        # pyG = pyGs[idx]
        for _ in range(5):
            print(pyG)
            pyG.edge_weight = pyG.edge_weight * 1
            # summary(pyG, 'pyG')
            nxG = to_networkx(pyG, edge_attrs=['edge_weight'],
                              to_undirected=True)  # important: directed/non-directed makes difference for cuts

            pyG_cut = normalized_cut(pyG.edge_index, pyG.edge_weight, pyG.num_nodes, )
            s = np.random.choice(range(pyG.num_nodes), int(pyG.num_nodes / 2.0), replace=False).tolist()
            if args.sample:
                s, t = s[:len(s) // 2], s[len(s) // 2:]
            else:
                s, t = s, None

            summary(np.array(s), 's')
            c = pyG_conductance(pyG.edge_index, pyG.edge_weight, s=s, t=None, verbose=True, dev='cuda')

            nxcut = nx.cut_size(nxG, s, T=t, weight='edge_weight')
            volume_S = nx.algorithms.volume(nxG, s, weight='edge_weight')
            c_ = nx.conductance(nxG, s, T=t, weight='edge_weight')
            print(nxcut, volume_S, pf(c, 3))
            print()

            assert c == c_, f'c: {c}. c_: {c_}'
    exit()

    nx_cut, nx_conductance = [], []
    for u, v in nxG.edges:
        cut = nx.normalized_cut_size(nxG, [u], [v], weight='edge_weight')
        conductance = nx.conductance(nxG, [u], [v], weight='edge_weight')
        conductance_ = pyG_conductance(pyG.edge_index, pyG.edge_weight, [u], [v], )
        assert conductance == conductance_, f'nx: {conductance}. pyG: {conductance_}'

        nx_cut.append(cut)
        nx_conductance.append(conductance)

    summary(np.array(nx_conductance), 'nx_conductance')

    exit()
    summary(pyG_cut.numpy(), 'pyG_cut')
    summary(np.array(nx_cut), 'nx_cut')
