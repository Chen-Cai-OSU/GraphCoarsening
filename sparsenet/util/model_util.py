# Created at 2020-05-06
# Summary: utils for sparsenet.

import copy
import os.path as osp
import random

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import from_networkx

from sparsenet.model.eval import trainer, tester  # train, set_train_data
from sparsenet.util.util import banner, red, pf, subset_graph, random_edge_index, summary, tonp, fix_seed, timefunc


def gen_subgraphs(g, assignment, hop=-1):
    """
    Given a big graph g and assignment, generate a collection of subgraphs of g, each corresponding to
    a node in the sparsified graph

    :param g: a pygeo graph
    :param assignment:
    :param hop: if -1, use assignment. Otherwise, sample a k-hop subgraph
    :return: g_subs: a dict where key is node of smaller graph and value is the pygeo_graph (subgraph of g)
    """

    from torch_geometric.utils.subgraph import k_hop_subgraph

    g_subs = dict()
    for k, indices in assignment.items():
        if hop != -1:
            indices, _, _ = k_hop_subgraph(k, hop, g.edge_index)

        g_sub = subset_graph(g, indices)
        # reindex g_sub

        g_subs[k] = g_sub
    return g_subs


class random_pyG():
    # todo: refactor. 1.
    def __init__(self):
        fix_seed()

    def random_pygeo_graph(self, n_node, node_feat_dim, n_edge, edge_feat_dim, device='cpu'):
        """ random DIRECTED pyG graph """
        g = Data(x=torch.rand(n_node, node_feat_dim),
                 edge_index=random_edge_index(n_edge, n_node),
                 edge_attr=torch.rand(n_edge, edge_feat_dim).type(torch.LongTensor))

        g = g.to(device)
        return g

    def reorder_edge_index(self, edge_index):
        """ for an edge index, reorder it so that each column
            [1, 2].T and [2,1].T are contiguious. A pretty useful subroutine.
        """
        edge_index = tonp(edge_index)
        n = edge_index.shape[1]
        assert edge_index.shape[0] == 2
        assert n % 2 == 0, 'Edge number should be even.'
        tmp = edge_index.T.tolist()
        tmp = [tuple(k) for k in tmp]

        edge2idx = dict(zip(tmp, range(n)))
        idx2edge = dict(zip(range(n), tmp))

        new_index = []
        for i in range(n):
            u, v = idx2edge[i]
            j = edge2idx[(v, u)]
            if (i not in new_index) and (j not in new_index):
                new_index.append(i)
                new_index.append(j)
        return new_index

    def reorder_pyG(self, g):
        new_index = self.reorder_edge_index(g.edge_index)

        new_edge_index = g.edge_index[:, new_index]
        new_edge_weight = g.edge_weight[new_index]

        new_g = Data(edge_index=new_edge_index, edge_weight=new_edge_weight)
        return new_g

    @timefunc
    def process_nx_graph(self, g, add_weight=False, uniform_weight=False):
        """ given a nx graph. add random edge_weight"""

        if isinstance(g, Data):
            print('already a pygeo graph')
            g = self.reorder_pyG(g)
            return g

        if add_weight:
            for u, v in g.edges:
                g[u][v]['edge_weight'] = np.random.random()

        if uniform_weight:
            for u, v in g.edges:
                g[u][v]['edge_weight'] = 0.1

        g = from_networkx(g)
        g = self.reorder_pyG(g)
        return g

    def random_rm_edges(self, g, n=1):
        """given a nx graph, random remove some edges """
        g_copy = copy.deepcopy(g)
        edges = list(g.edges)
        assert n < len(edges)
        chosen_edges = random.sample(edges, k=n)
        # for edge in chosen_edges:
        #     g.remove_edge(edge[0], edge[1])
        # g = g.remove_edge(edge[1], edge[0])

        g_copy.remove_edges_from(chosen_edges)
        return g_copy

    def increase_random_edge_w(self, g, n=1, w=1000):
        """ randomly increase the weight of edge. change the weight to w """

        g = copy.deepcopy(g)
        n_edge = g.edge_index.size(1) // 2
        assert n < n_edge, f'n {n} has to be smaler than {n_edge}.'
        change_edges = random.sample(range(n_edge), k=n)
        new_weight = g.edge_weight
        for idx in change_edges:
            new_weight[2 * idx] = w
            new_weight[2 * idx + 1] = w

        new_index = g.edge_index
        new_g = Data(edge_index=new_index, edge_weight=new_weight)
        return new_g

    def rm_pyG_edges(self, g, n=1):
        n_edge = g.edge_index.size(1) // 2
        retain_edges = random.sample(range(n_edge), k=n_edge - n)
        indices = []
        for idx in retain_edges:
            indices.append(2 * idx)
            indices.append(2 * idx + 1)

        new_edge_index = g.edge_index[:, indices]
        new_edge_weight = g.edge_weight[indices]
        new_g = Data(edge_index=new_edge_index, edge_weight=new_edge_weight)
        return new_g

    def get_planetoid(self, dataset='cora'):
        path = osp.join('/home/cai.507/Documents/DeepLearning/sparsifier/sparsenet', 'data', dataset)
        dataset = Planetoid(path, dataset, T.TargetIndegree())
        n_edge = dataset.data.edge_index.size(1)
        g = Data(edge_index=dataset.data.edge_index, edge_weight=torch.ones(n_edge))
        assert g.is_directed() == False
        # g = g.coalesce()
        return g


class ModelEvaluator():
    def __init__(self, model, dataset_loader, dev, optimizer):
        self.dev = dev
        self.optimizer = optimizer
        self.model = model
        self.dataset_loader = dataset_loader

    def set_modelpath(self, path):
        self.modelpath = path

    def train(self, idx, TR, model, args):
        """ train the model for one graph """
        TR.set_train_data(args, self.dataset_loader)
        TR.train(model, self.optimizer, args, verbose=False)
        TR.delete_train_data(idx)
        return model

    def validate(self, idx, val_indices, TE, model, args):
        val_score = self.val_score
        val_score[idx] = {'n_gen': [], 'impr_ratio': [], 'eigen_ratio': []}
        for idx_ in val_indices:
            args.test_idx = idx_
            args.cur_idx = idx_
            TE.set_test_data(args, self.dataset_loader)
            n_gen, impr_ratio, eigen_ratio = TE.eval(model, args, verbose=False)

            val_score[idx]['n_gen'].append(n_gen)
            val_score[idx]['impr_ratio'].append(impr_ratio)
            val_score[idx]['eigen_ratio'].append(eigen_ratio)

        banner(f'{args.dataset}: finish validating graph {val_indices}.')

        cur_impr_ratio = np.mean(val_score[idx]['impr_ratio'])
        cur_eigen_ratio = np.mean(val_score[idx]['eigen_ratio'])
        print(cur_eigen_ratio, self.best_eigen_ratio)
        self.val_score[idx] = val_score[idx]
        return cur_impr_ratio, cur_eigen_ratio

    def save(self, idx, model, mode='eigen-ratio'):
        """ save model for training graph idx """
        assert mode in ['eigen-ratio', 'improve-ratio']
        f = f'checkpoint-best-{mode}.pkl'

        if mode == 'eigen-ratio':
            torch.save(model.state_dict(), self.modelpath + f)
            print(red(f'Save model for train idx {idx}. Best-eigen-ratio is {pf(self.best_eigen_ratio, 2)}.'))
        elif model == 'improve-ratio':
            torch.save(model.state_dict(), self.modelpath + f)
            print(red(f'Save model for train idx {idx}. Best-improve-ratio is {pf(self.best_impr_ratio, 2)}.'))

    def find_best_model(self, model, train_indices, val_indices, args):
        """ save the best model on validation dataset """

        self.TR = trainer(dev=self.dev)
        self.TE = tester(dev=self.dev)

        self.val_score = {}
        self.best_n_gen = -1e10
        self.best_impr_ratio = -1e30
        self.best_eigen_ratio = -1e30
        self.train_indices = train_indices
        self.val_indices = val_indices

        for idx in self.train_indices:
            args.train_idx = idx
            args.cur_idx = idx

            model = self.train(idx, self.TR, model, args)
            cur_impr_ratio, cur_eigen_ratio = self.validate(idx, val_indices, self.TE, model, args)

            # save the model if it works well on val data
            if cur_eigen_ratio > self.best_eigen_ratio:
                self.best_eigen_ratio = cur_eigen_ratio
                self.save(idx, model, mode='eigen-ratio')

            if cur_impr_ratio > self.best_impr_ratio:
                self.best_impr_ratio = cur_impr_ratio
                self.save(idx, model, mode='improve-ratio')
        return model, args

    def test_model(self, model, test_indices, AP, args):
        model_name = AP.set_model_name()

        model.load_state_dict(torch.load(self.modelpath + model_name))

        for idx_ in test_indices:
            args.test_idx = idx_
            args.cur_idx = idx_
            self.TE.set_test_data(args, self.dataset_loader)
            self.TE.eval(model, args, verbose=False)
            banner(f'{args.dataset}: finish testing graph {idx_}.')


if __name__ == '__main__':
    n_node = 100
    helper = random_pyG()

    g = nx.random_geometric_graph(n_node, 0.2, seed=42)
    g = helper.process_nx_graph(g, add_weight=True)

    g_drop = helper.rm_pyG_edges(g, n=100)
    g_reweight = helper.increase_random_edge_w(g, n=100, w=1000)

    summary(g, 'g')
    summary(g_drop, 'g_drop')
    summary(g_reweight, 'g_reweight ')
    exit()

    edge_index = np.array([[1, 3, 4, 2], [2, 4, 3, 1]])

    n_node, n_edge = 4, 3
    nfeat_dim, efeat_dim = 1, 1
    generator = random_pyG()
    new_index = generator.reorder_edge_index(edge_index)
    # print(new_index)

    g = nx.random_geometric_graph(10, .3)
    g = generator.process_nx_graph(g)
    summary(g, 'nx_test')
    print(g)
    print(g.edge_index)
    print(g.edge_weight)
