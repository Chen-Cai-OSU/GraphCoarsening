# Created at 2020-05-09
# Summary: evaluate sparsification algorithm

import torch
import torch_geometric
from torch_geometric.utils import get_laplacian, from_networkx

from sparsenet.model.loss import get_laplacian_mat, get_projection_mat, random_vec_loss
from sparsenet.util.data import shape_data
from sparsenet.util.sample import sample_N2Nlandmarks
from sparsenet.util.util import summary, maybe_edge_weight, random_pygeo_graph, fix_seed, pf


class evaluator:
    def __init__(self, g, laplacian = 'sym'):
        """ original graph """
        assert isinstance(g, torch_geometric.data.data.Data)
        self.g = g
        self.n_node = g.num_nodes
        self.laplacian = laplacian
        self.L1 = self._get_laplacian(g)

    def _get_smaller_graph(self, n_node_sml):
        g = self.g
        g_sml, assignment = sample_N2Nlandmarks(g, n_node_sml, weight_key='edge_weight')
        g_sml = from_networkx(g_sml)
        return g_sml, assignment

    def BL0(self, n_node_sml, verbose = False):
        """ baseline 0 """
        g_sml, assignment = self._get_smaller_graph(n_node_sml)
        L2 = self._get_laplacian(g_sml) # get_laplacian_mat(g_sml.edge_index, g_sml.edge_weight, g_sml.num_nodes)
        Projection = get_projection_mat(self.n_node, n_node_sml, assignment)
        loss = random_vec_loss(self.L1, L2, Projection)

        if verbose:
            summary(g_sml, 'g_sml')
            print(loss)
            summary(self.L1, 'L1')
            summary(L2, 'L2')

        print(f'BL0: n_node_sml={n_node_sml}, loss: {pf(loss.item())}.')
        return

    def BL1(self, n_node_sml):
        pass

    def _get_laplacian(self, g):
        """
        :param g: torch_geometric.data.data.Data
        :return: Ll: sparse tensor
        """
        L = get_laplacian_mat(g.edge_index, g.edge_weight, g.num_nodes, normalization=self.laplacian)
        return L


if __name__ == '__main__':
    fix_seed()
    # n_node, n_edge = 320, 5000
    # nfeat_dim = 42
    # efeat_dim = 20
    # n_node_sml = 150

    g =  shape_data(n=1, _k=10)[0] # random_pygeo_graph(n_node, nfeat_dim, n_edge, efeat_dim, device='cpu') #
    g.edge_weight = 10 * torch.rand(g.edge_index.size(1), device=g.edge_index.device) # set the edge weight of original graph to be random
    summary(g, 'g')

    E = evaluator(g, laplacian=None)
    g_sml, _ = E._get_smaller_graph(100)
    summary(g_sml, 'g_sml')

    for n_node_sml in range(160, 2000, 100):
        E.BL0(n_node_sml)
