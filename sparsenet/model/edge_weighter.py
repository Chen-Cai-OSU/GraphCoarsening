# Created at 2020-05-04
# Summary: a simple model to assign the weights for edges

import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
import torch.nn.functional as F

from sparsenet.model.loss import random_vec_loss
from sparsenet.util.util import summary, random_edge_index, fix_seed, pf


def random_laplacian(n = 10):
    """ a random laplacian used for quick test """
    l = torch.rand((n,n))
    l += l.T
    return l

def random_loss(l1, l2=None, device='cuda'):
    if l2 == None:
        l2 = torch.zeros(l1.size()).to(device)
    return F.l1_loss(l1, l2)


class edge_weighter(torch.nn.Module):
    def __init__(self, emb_dim):
        super(edge_weighter, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1))

    def forward(self, g):
        """ x1, x2 is the embedding of the two ends of edge.
            both are tesnors of shape (n_dim, bs)
        """

        # one dim output, saved in edge_weight attribute of pygeo graph
        x_i = torch.index_select(g.node_feat, 0, g.edge_index[0])  # torch of shape (size_small_graph, node_dim)
        x_j = torch.index_select(g.node_feat, 0, g.edge_index[1])
        weight = self.mlp(x_i + x_j)  # tensor of [n_edge_smaller, 1]

        return weight.view(-1)

if __name__ == '__main__':
    fix_seed()
    size_of_small_graph = 30
    n_edge = 500
    node_dim = 10
    edge_feat_dim = 10

    g_small = Data(x=torch.rand(size_of_small_graph, node_dim),
                   edge_index=random_edge_index(n_edge, size_of_small_graph),
                   edge_attr=torch.rand(n_edge, edge_feat_dim).type(torch.LongTensor))
    summary(g_small, 'g_small')

    edge_index, edge_weight = get_laplacian(g_small.edge_index, normalization='sym')
    summary(edge_index, 'edge_index')
    summary(edge_weight, 'edge_weight')
    # random

    x_i = torch.index_select(g_small.x, 0, edge_index[0]) # todo: check x_i,x_j
    x_j = torch.index_select(g_small.x, 0, edge_index[1])
    summary(x_i, 'x_i')
    summary(x_j, 'x_j')

    model = edge_weighter(node_dim)
    weight = model(x_i, x_j)
    summary(weight, 'weight')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        weight = model(x_i, x_j)
        loss = random_loss(weight, l2=None)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}: {pf(loss.item(), 4)}')
    exit()


    # from edge_weight need to construct a laplacian
    l1 = random_laplacian(3)
    l2 = random_laplacian(2)
    Assignment = {0: set([1, 2]), 1: set([0])}
    loss = random_vec_loss(l1.numpy(), l2.numpy(), Assignment)
    print(loss)

    exit()
    x1, x2 = torch.rand(32, 100), torch.rand(32, 100)
    model = edge_weighter(100)
    out = model(x1, x2)
    summary(out)