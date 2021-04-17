# Created at 2020-06-02
# Summary: pygsp util

import collections
from time import time
from warnings import warn

import networkx as nx
import scipy
import torch
import torch_geometric
from graph_coarsening.coarsening_utils import *
from pygsp import graphs
from torch_geometric.data.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from sparsenet.util.util import summary, tonp, np2set, timefunc, red, pyg2gsp, update_dict


def assert_proj_matrix(C):
    proj_error = sp.sparse.linalg.norm(((C.T).dot(C)) ** 2 - ((C.T).dot(C)), ord='fro')
    assert proj_error < 1e-5, f'proj error {proj_error} larger than {1e-5}.'


def ba_graph(N=400):
    # return a gsp graph
    G = graphs.BarabasiAlbert(N)
    if not hasattr(G, 'coords'):
        try:
            graph = nx.from_scipy_sparse_matrix(G.W)
            pos = nx.nx_agraph.graphviz_layout(graph, prog='neato')
            G.set_coordinates(np.array(list(pos.values())))
        except ImportError:
            G.set_coordinates()
        G.compute_fourier_basis()  # this is for convenience (not really needed by coarsening)
    return G


class gsp2pyg(object):
    def __init__(self, g, **loukas_args):
        assert isinstance(g, torch_geometric.data.data.Data)
        self.origin_pyg = g
        self.gspG = self.pyg2gsp(g)
        self.pyg = self.gsp2pyg(self.gspG)
        self.loukas_method(**loukas_args)
        self.pyg_sml = self.gsp2pyg(self.gspG_sml)
        self.assignment = self.process()
        self._set_pos()

    def _set_pos(self):
        """ set pos for smaller pyg graph """
        # todo: this is for what?

        if 'pos' not in self.origin_pyg.keys:
            return

        n = self.pyg_sml.num_nodes
        d = self.origin_pyg.pos.size(1)
        pos = torch.zeros((n, d))
        for k, v in self.assignment.items():
            v = list(v)
            pos[k, :] = torch.mean(self.origin_pyg.pos[v], 0)
        self.pyg_sml.pos = pos

    def pyg2gsp(self, g):
        return pyg2gsp(g)

    def gsp2pyg(self, g):
        """ works only for g with uniform weights """
        from sparsenet.util.data import input_check
        edge_index, edge_weight = from_scipy_sparse_matrix(g.W)
        edge_weight = edge_weight.type(torch.FloatTensor)

        summary(edge_weight, 'edge_weight in gsp2pyg', highlight=True)
        pyG = Data(edge_index=edge_index, edge_weight=edge_weight,
                   edge_attr=torch.flatten(edge_weight))  # important: set edge_attr to be edge_weight
        pyG_check = input_check(pyG, size_check=False, eig=False)  # need to comment out for Cora
        try:
            assert g.N == pyG_check.num_nodes
            return pyG_check
        except AssertionError:
            print(f'AssertionError! gsp Graph size is {g.N} but pyG size is {pyG_check.num_nodes}. '
                  f'{red("Return pyG instead of pyG_check.")}')
            return pyG

    @timefunc
    def loukas_method(self, **kwargs):
        """ api to call loukas's code.
            modified from looukas's code.
        This function provides a common interface for coarsening algorithms that contract subgraphs

        Parameters
        ----------
        G : pygsp Graph
        K : int
            The size of the subspace we are interested in preserving.
        r : float between (0,1)
            The desired reduction defined as 1 - n/N.
        method : String
            ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']

        Returns
        -------
        C : np.array of size n x N
            The coarsening matrix.
        Gc : pygsp Graph
            The smaller graph.
        Call : list of np.arrays
            Coarsening matrices for each level
        Gall : list of (n_levels+1) pygsp Graphs
            All graphs involved in the multilevel coarsening

        Example
        -------
        C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
        """

        t0 = time()
        default_kwargs = {'K': 40, 'r': 0.5, 'method': 'variation_edges', 'max_levels': 20}
        loukas_quality = kwargs.get('loukas_quality', False)
        kwargs.pop('loukas_quality', None)
        kwargs = update_dict(kwargs, default_kwargs)
        print(f'{red("kwargs for coarsen function")}: {kwargs}\n')
        G = self.gspG
        K = kwargs['K']

        # precompute spectrum needed for metrics
        if loukas_quality:
            if False:  # K_all[-1] > N / 2:
                pass  # [Uk, lk] = eig(G.L)
            else:
                offset = 2 * max(G.dw)
                T = offset * sp.sparse.eye(G.N, format='csc') - G.L
                lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which='LM', tol=1e-6)
                lk = (offset - lk)[::-1]
                Uk = Uk[:, ::-1]
                kwargs['Uk'] = Uk
                kwargs['lk'] = lk
        t1 = time()

        C, Gc, Call, Gall = coarsen(self.gspG, **kwargs)

        if loukas_quality:
            metrics = coarsening_quality(G, C, kmax=K, Uk=Uk[:, :K], lk=lk[:K])
            for k in metrics:
                summary(metrics[k], k, highlight=True)
        else:
            print(red('No coarsening_quality.'))

        _check_loukas(self, G, Gc, C)
        t2 = time()

        P = C.power(2)
        self.P = P  # save memory
        self.gspG_sml = Gc
        self.C = C
        t3 = time()

        print(f'Compute Eigenvalue: {int(t1 - t0)}')
        print(f'Coarsen + Metric: {int(t2 - t1)}')
        print(f'Misc: {int(t3 - t2)}')

    def process(self):
        # convert coarsening matrix to assignment / projection to make life easy
        sml_idx, big_idx = self.P.nonzero()
        sml_idx, big_idx = sml_idx.astype(int), big_idx.astype(int)

        n = len(sml_idx)
        assignment = {}
        for i in range(n):
            assignment[sml_idx[i]] = set()

        for i in range(n):
            k, v = sml_idx[i], big_idx[i]
            assignment[k].add(v)
        del self.P
        return assignment

    def _check_loukas(self, G, Gc, C):
        # verify matrix. change to scipy multiplication.

        Q = C.dot(np.ones((C.shape[1], 1))).reshape(-1)
        Q = scipy.sparse.diags([Q], [0])
        QC = Q.dot(C)
        Lc_Q = QC.dot((G.L).dot(QC.T))
        diff = Lc_Q - Gc.L
        if np.max(diff) > 0.1:
            warn('Lc_Q - Gc.L is not close enough.')
            del Lc_Q


if __name__ == '__main__':
    from sparsenet.util.util import random_pygeo_graph

    pyg = random_pygeo_graph(100, 1, 4000, 1)

    converter = gsp2pyg(pyg, loukas_quality=False)
    gsp = converter.gspG
    pyG1 = converter.gsp2pyg(gsp)
    summary(pyG1, 'pyG1 ')

    exit()
    pyg = random_pygeo_graph(10, 1, 40, 1)
    converter = gsp2pyg(pyg)
    g_sml, assignment = converter.pyg_sml, converter.assignment
    summary(g_sml, 'g_sml')
    print(assignment)
    exit()

    G = ba_graph(400)
    method = 'variation_edges'  # 'variation_neighborhood'
    r = 0.6  # the extend of dimensionality reduction (r=0 means no reduction)
    k = 5
    kmax = int(3 * k)

    C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method)
    assert_proj_matrix(C)
    print(type(C))
    P = C.power(2)

    assert isinstance(P, scipy.sparse.csc.csc_matrix)
    P = tonp(P)
    n1, n2 = C.shape
    ret = P.dot(np.ones((n2,)))
    print(collections.Counter(ret))
    print(np2set(ret))
    summary(ret, 'check')
