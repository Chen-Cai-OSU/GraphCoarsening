# Created at 2020-05-08
# Summary: some metric to evaluate the quality of sparsified graph
import argparse
import itertools
import sys
from functools import partial
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import linalg as LA
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx

from sparsenet.util.data import data_loader
from sparsenet.util.gsp_util import gsp2pyg
from sparsenet.util.sample import sample_N2Nlandmarks
from sparsenet.util.util import timefunc, tonp, summary, fix_seed


def random_sysm(n=10):
    """ random symmetric matrix"""
    m = np.random.random((n, n))
    return m + m.T


def random_laplacian(n=10):
    g = nx.random_geometric_graph(n, 0.5)
    return nx.laplacian_matrix(g).toarray()


@timefunc
def eig(L):
    L = tonp(L)
    try:
        assert np.allclose(L, L.T)
    except AssertionError:
        summary(L - L.T, 'L - L.T')
        plt.imshow(L)  # f'L is not symmetric. Diff is {np.max(L- L.T)}'
        plt.colorbar()
        plt.title('L-L.T')
        plt.show()
        sys.exit('Not sym matrix')

    w, v = LA.eigh(L)
    return w, v


@timefunc
def effective_resistance(n=100):
    g = nx.random_geometric_graph(n, 0.5)
    print(f'num of edges: {len(g.edges)}')
    for u, v in g.edges:
        d = nx.resistance_distance(g, u, v)  # todo: this is very slow


class metric_discrepancy:
    """ look at the discrency of various metric such as
    cuts, effective resistance, max flow, conductance...
    """

    def __init__(self, G, G_prime, Assignment, N=10):
        self._setup(G, G_prime, Assignment)

    def _setup(self, G, G_prime, Assignment, weight_key='edge_weight'):
        fix_seed()
        if isinstance(G, Data):
            G = to_networkx(G, edge_attrs=[weight_key])
            if nx.is_directed(G): G = G.to_undirected()

        if isinstance(G_prime, Data):
            assert G_prime.is_directed() == False
            G_prime = to_networkx(G_prime, edge_attrs=[weight_key])
            if nx.is_directed(G_prime): G_prime = G_prime.to_undirected()

        assert nx.is_connected(G), f'Input graph must be connected.'
        assert nx.is_connected(G_prime), f'Sparsified graph must be connected.'

        self.G = G
        self.G_prime = G_prime
        self.weight_key = weight_key

        self.G_edges = list(G.edges)
        self.G_prime_edges = list(G_prime.edges())
        self.V = list(self.G.nodes)
        self.V_prime = list(self.G_prime.nodes)

        self.asgmt = Assignment  # key is for nodes in sparsified graph.
        self.inv_asgmt = {v: [key] for (key, value) in Assignment.items() for v in
                          value}  # key is the nodes in large graph.

    def __volume_debug(self):
        def volume_sum(G, S, T, weight=None):
            v1 = nx.algorithms.volume(G, S, weight=weight)
            v2 = nx.algorithms.volume(G, T, weight=weight)
            return min(v1, v2)

        return volume_sum

    def __select_func(self, method):
        """ return a function """
        if method == 'cut_size':
            self.method = nx.cut_size
        if method == 'volume':
            self.method = self.__volume_debug()
        elif method == 'conductance':
            self.method = partial(nx.algorithms.conductance, weight=self.weight_key)
        elif method == 'normalized_cut_size':
            self.method = partial(nx.algorithms.normalized_cut_size, weight=self.weight_key)
        elif method == 'resistance':
            self.method = self.wrapper(method)  # nx.algorithms.resistance_distance
        elif method == 'flow':
            self.method = self.wrapper(method)

    def wrapper(self, method):

        def modified_resistance_distance(G, nodeA, nodeB, weight=None):
            assert isinstance(nodeA, (list, set)), print(f'nodeA is {nodeA}')
            assert isinstance(nodeB, (list, set)), print(f'nodeB is {nodeB}')
            assert len(nodeA) == 1 and len(nodeB) == 1, f'Expect list of length 1. Got NodeA {nodeA} and NodeB {nodeB}'

            nodeA, nodeB = nodeA[0], nodeB[0]

            if nodeA == nodeB:
                warn(f'nodeA {nodeA} and nodeB {nodeB} has to be different for {method}. Set to be 0.')
                return 0
            else:
                if method == 'resistance':
                    return nx.algorithms.resistance_distance(G, nodeA, nodeB, weight=self.weight_key)
                elif method == 'flow':
                    flowv, _ = nx.algorithms.flow.maximum_flow(G, nodeA, nodeB, capacity=self.weight_key)
                    return flowv
                else:
                    raise NotImplementedError

        return modified_resistance_distance

    def sample(self, N, method='prime_edges', **kwargs):

        if method == 'prime_disjoint_sets':
            node_list = list(self.G_prime.nodes())
            n_sample = kwargs.get('n_sample', 5)
            assert len(node_list) > 2 * n_sample
            pairs, prime_pairs = [], []
            for i in range(N):
                s_t = np.random.choice(node_list, 2 * n_sample, replace=False)
                set_sprime = s_t[:n_sample]
                set_tprime = s_t[-n_sample:]
                set_s = [list(self.asgmt[s]) for s in set_sprime]
                set_t = [list(self.asgmt[t]) for t in set_tprime]
                set_s = list(itertools.chain.from_iterable(set_s))
                set_t = list(itertools.chain.from_iterable(set_t))
                pairs.append((set_s, set_t))
                prime_pairs.append((set_sprime, set_tprime))
                # print(f'set_s: {set_s}')
                # print(f'set_sprime: {set_sprime}')

        elif method == 'prime_pairs':
            n_pairs = len(self.G_prime) * (len(self.G_prime) - 1) * 0.5
            if N > n_pairs:
                print(f'Method {method}: N {N} is larger than number of pairs in G.')
                N = min(N, n_pairs)

            node_list = list(self.G_prime.nodes())
            # easy implementation. todo: change later
            x = np.random.choice(node_list, 2 * N, replace=True)
            y = np.random.choice(node_list, 2 * N, replace=True)
            pairs_ = [(x[i], y[i]) for i in range(2 * N) if x[i] != y[i]]
            pairs, prime_pairs = [], []
            for (u, v) in pairs_[:N]:
                sp, tp = u, v
                prime_pairs.append(([sp], [tp]))
                s, t = self.asgmt[sp], self.asgmt[tp]
                pair = (s, t)
                pairs.append(pair)

        elif method == 'prime_edges':
            if N > self.G_prime.number_of_edges():
                print(f'Method {method}: N {N} is larger than number of G prime edges.')
                N = min(N, self.G_prime.number_of_edges())

            random_pairs = np.random.choice(self.G_prime.number_of_edges(), N, replace=False)
            prime_pairs_ = [self.G_prime_edges[idx] for idx in random_pairs]
            pairs, prime_pairs = [], []
            for (u, v) in prime_pairs_:
                sp, tp = self.V_prime[u], self.V_prime[v]
                prime_pairs.append(([sp], [tp]))

                s, t = self.asgmt[sp], self.asgmt[tp]
                pair = (s, t)
                pairs.append(pair)

        elif method == 'pairs':
            n_pairs = len(self.G) * (len(self.G) - 1) * 0.5
            if N > n_pairs:
                print(f'Method {method}: N {N} is larger than number of pairs in G.')
                N = min(N, n_pairs)

            node_list = list(self.G.nodes())
            # easy implementation. todo: change later
            x = np.random.choice(node_list, 2 * N, replace=True)
            y = np.random.choice(node_list, 2 * N, replace=True)
            pairs_ = [(x[i], y[i]) for i in range(2 * N) if x[i] != y[i]]
            pairs, prime_pairs = [], []
            for (u, v) in pairs_[:N]:
                s, t = u, v
                pairs.append(([s], [t]))
                sp, tp = self.inv_asgmt[s], self.inv_asgmt[t]
                pair = (sp, tp)
                prime_pairs.append(pair)

        elif method == 'edges':
            if N > self.G.number_of_edges():
                print(f'Method {method}: N {N} is larger than number of G edges.')
                N = min(N, self.G.number_of_edges())

            random_pairs = np.random.choice(self.G.number_of_edges(), N, replace=False)
            pairs_ = [self.G_edges[idx] for idx in random_pairs]
            pairs, prime_pairs = [], []
            for (u, v) in pairs_:
                s, t = self.V[u], self.V[v]
                pairs.append(([s], [t]))

                s, t = self.inv_asgmt[s], self.inv_asgmt[t]
                pair = (s, t)
                prime_pairs.append(pair)

        else:
            NotImplementedError

        self.prime_pairs = prime_pairs
        self.pairs = pairs
        self.N = N

    @timefunc
    def discrepancy(self, method='cut_size', verbose=False):
        relative_loss = []
        self.__select_func(method=method)
        for i in range(self.N):
            s, t = self.pairs[i]
            sp, tp = self.prime_pairs[i]

            q = self.method(self.G, s, t, weight=self.weight_key)
            q_prime = self.method(self.G_prime, sp, tp, weight=self.weight_key)
            if verbose: print(q, q_prime)

            try:
                error = abs(q - q_prime) / q
            except:
                error = -1
                # assert q!=0, f'q: {q}. q_prime: {q_prime}'
            relative_loss.append(error)

        summary(np.array(relative_loss), f'{method} Error', highlight=True)
        # return relative_loss / self.N


@timefunc
def nodepair_min_cut(G, G_prime, Assignment, N=10, weight_key='edge_weight', reproducible=True):
    """
    We sample N pair of connected nodes in G, find the correspondence in G_prime.
    For each pair (s, t) in G, and (s', t') in G_prime.
    Compute the "relative loss" of min_cut defined as |min_cut(s, t) - min_cut(s', t')| / min_cut(s, t).
    Return the averaged min_cut loss over all N pairs.
    :param G: The original (weighted) graph in networkx type.
    :param G_prime: The sparsigied (weighted) graph, networkx graph too.
    :param Assignment: The mapping.
    :param weight_key: key for edge weight, default value is 'edge_weight'
    :return: "relative loss" for min_cut averaged over N pairs.
    """
    if reproducible: fix_seed()
    if isinstance(G, Data):
        G = to_networkx(G, edge_attrs=[weight_key])
        if nx.is_directed(G): G = G.to_undirected()

    if isinstance(G_prime, Data):
        assert G_prime.is_directed() == False
        G_prime = to_networkx(G_prime, edge_attrs=[weight_key])
        if nx.is_directed(G_prime): G_prime = G_prime.to_undirected()

    assert nx.is_connected(G), f'Input graph must be connected.'
    V_prime = list(G_prime.nodes)

    G_prime_edges = list(G_prime.edges())
    random_pairs = np.random.choice(G_prime.number_of_edges(), N, replace=False)
    random_pairs = [G_prime_edges[idx] for idx in random_pairs]

    relative_cut_loss = 0.0
    for (u, v) in random_pairs:
        sp, tp = V_prime[u], V_prime[v]
        s, t = Assignment[sp], Assignment[tp]
        sp, tp = [sp], [tp]

        gcut, gpcut = nx.cut_size(G, s, t, weight=weight_key), nx.cut_size(G_prime, sp, tp, weight=weight_key)
        # print(gcut, gpcut)
        relative_cut_loss += abs(gcut - gpcut) / gcut
    # print(relative_cut_loss)
    return relative_cut_loss / N


parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--n_layer', type=int, default=3, help='number of layer')
parser.add_argument('--emb_dim', type=int, default=50, help='embedding dimension')
parser.add_argument('--ratio', type=float, default=0.5, help='reduction ratio')
parser.add_argument('--n_vec', type=int, default=100, help='number of random vector')
parser.add_argument('--force_pos', action='store_true', help='Force the output of GNN to be positive')
parser.add_argument('--dataset', type=str, default='random_geo', help='the name of dataset')

# optim
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--n_epoch', type=int, default=50, help='')
parser.add_argument('--bs', type=int, default=600, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_bottomk', type=int, default=40, help='Number of Bottom K eigenvector')

# debug
parser.add_argument('--debug', action='store_true', help='debug. Smaller graph')
parser.add_argument('--lap', type=str, default='none', help='Laplacian type', choices=[None, 'sym', 'rw', 'none'])
parser.add_argument('--viz', action='store_true', help='visualization of weights of sparsified graph ')
parser.add_argument('--train_idx', type=int, default=0, help='train index of the shape data. Do not change.')
parser.add_argument('--test_idx', type=int, default=0, help='test index of the shape data. Do not change.')
parser.add_argument('--lap_check', action='store_true', help='check the laplacian is normal during training')

parser.add_argument('--loukas_quality', action='store_true', help='Compute the coarsening quality of loukas method')
parser.add_argument('--train_indices', type=str, default='0,',
                    help='train indices of the dataset')  # https://bit.ly/3dtJtPn
parser.add_argument('--test_indices', type=str, default='0,', help='test indices of the dataset')
parser.add_argument('--strategy', type=str, default='DK', help='coarsening strategy', choices=['DK', 'loukas'])
parser.add_argument('--method', type=str, default='variation_edges', help='Loukas methods',
                    choices=['variation_neighborhoods', 'variation_edges', 'variation_cliques',
                             'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'variation_neighborhood'])

if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed()
    n_node, n_edge = 320, 5000
    N = 1

    g, _ = data_loader(args, dataset=args.dataset).load(args, 'train')  # random_pygeo_graph(n_node, 1, n_edge, 1)
    # g = input_check(g, eig=False)

    if args.strategy == 'loukas':
        loukas_kwargs = {'r': args.ratio, 'method': args.method, 'loukas_quality': args.loukas_quality}
        converter = gsp2pyg(g, **loukas_kwargs)
        g_sml, assignment = converter.pyg_sml, converter.assignment
        n = g_sml.num_nodes
    elif args.strategy == 'DK':
        g_sml, assignment = sample_N2Nlandmarks(g, 100)
        n = len(g_sml)
    else:
        raise NotImplementedError

    MD = metric_discrepancy(g, g_sml, assignment)
    MD.sample(N, method='prime_disjoint_sets', n_sample=int(n * 0.1))
    MD.discrepancy(method='conductance', verbose=True)
    MD.discrepancy(method='normalized_cut_size', verbose=True)
    MD.discrepancy(method='volume', verbose=True)
    MD.discrepancy(method='cut_size', verbose=True)
    # MD.discrepancy(method='cut_size')
    exit()

    MD.sample(N, method='pairs')
    MD.discrepancy(method='resistance')
    # MD.sample(N, method='pairs')
    # MD.discrepancy(method='flow')

    exit()

    for N in range(100, 1000, 100):
        cut_loss = nodepair_min_cut(g, G_prime, Assignment, N=N, weight_key='edge_weight', reproducible=True)
        print(cut_loss)
        print('-' * 20)

    exit()
    for n in range(100, 5000, 100):
        m = random_laplacian(n)  # random_sysm(n)
        print(n)
        eig(m)
