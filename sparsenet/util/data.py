# Created at 2020-05-04
# Summary: data related functions.

import os
import os.path as osp
from time import time
from warnings import warn

import networkx as nx
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from graph_coarsening import graph_lib
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh
from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.datasets import ShapeNet, Planetoid, CoraFull, Coauthor, \
    Amazon, Reddit, FAUST, DynamicFAUST, Flickr
from torch_geometric.datasets.yelp import Yelp
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import from_networkx, degree
from torch_geometric.utils.subgraph import k_hop_subgraph
from torch_geometric.utils.undirected import to_undirected

from sparsenet.model.loss import get_laplacian_mat
from sparsenet.util.graph_util import subgraphs
from sparsenet.util.name_util import ego_graphs
from sparsenet.util.util import red, timefunc, largest_cc, num_comp, fix_seed, banner, dict2name, summary, \
    random_pygeo_graph, sparse_tensor2_sparse_numpyarray

data_dir = os.path.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
egograph_dir = os.path.join(data_dir, 'egographs')
fix_seed()


class test_parallel:
    """
    http://qingkaikong.blogspot.com/2016/12/python-parallel-method-in-class.html
    """

    def __init__(self):
        pass

    def run(self, cmd):
        assert isinstance(cmd, str)
        os.system(cmd)

    def run_parallel(self, cmds, n_jobs=1):
        Parallel(n_jobs=n_jobs)(delayed(unwrap_self)(cmd) for cmd in cmds)


def unwrap_self(cmd):
    return test_parallel().run(cmd)


@timefunc
def shape_data(n, _k=5, name='ShapeNet'):
    """ from shape dataset, generate a few graphs. One shape has one graph (pytorch.geometric format).
    :param n: number of graphs
    :param k: k-NN graph
    """

    dir = os.path.join(osp.dirname(osp.realpath(__file__)), '..')
    if name == 'ShapeNet':
        data = ShapeNet(root=os.path.join(dir, 'data/shape_net'),
                        pre_transform=T.KNNGraph(k=_k))
    elif name == 'FAUST':
        data = FAUST(root=os.path.join(dir, 'data/faust'), pre_transform=torch_geometric.transforms.FaceToEdge())
    elif name == 'DynamicFAUST':
        data = DynamicFAUST(root=os.path.join(dir, 'data/dynamic_faust'),
                            pre_transform=torch_geometric.transforms.FaceToEdge)

    samples = [i for i in range(data.len())]
    rets = [data[i] for i in samples[:n] if num_comp(data[i]) == 1]  # filter out graph of more than 1 component

    for i, ret in enumerate(rets):
        n_comp = num_comp(ret)
        if n_comp != 1: raise Exception(f'{i}th graph has {n_comp} components.')

    rets = [input_check(graph) for graph in rets]
    rets = [g for g in rets if g != None]
    return rets


def handle_num_nodes(g):
    """ for some dataset, the num_of_nodes is [100] instead of 100.
        this function handles this case
    """
    assert isinstance(g, Data)
    if isinstance(g.num_nodes, list):
        assert len(g.num_nodes) == 1
        g.num_nodes = g.num_nodes[0]
    elif isinstance(g.num_nodes, int):
        pass
    else:
        f'g.num_nodes is {g.num_nodes}'
        raise NotImplementedError
    return g

class MiscDataset(object):
    def __init__(self):
        pass

    @staticmethod
    def hybrid_graphs(name='amazons'):
        g = []
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        if name == 'amazons':
            g.append(Amazon(osp.join(path, 'photo'), name='photo', pre_transform=input_check).data)
            g.append(Amazon(osp.join(path, 'computers'), name='computers', pre_transform=input_check).data)
        elif name == 'coauthors':
            for dataset in ['Coauthor-CS', 'Coauthor-physics']:
                kwargs = {"hop": 3, "size": 20, "dataset": dataset, 's_low': 5000, 's_high': 10000}
                data = EgoGraphs(**kwargs)
                g.append(data[0])
            [handle_num_nodes(g_) for g_ in g]
        return g


def egographs(hop=4, size=5, dataset='PubMed',
              s_low=200, s_high=5000,
              sample='ego', n_vec=210, w_len=5000,
              include_self=True):
    """
    :param hop: hop size to sample subgraphs
    :param size: number of subgraphs
    :param dataset: name of dataset
    :param s_low: lower bound for the size of subgraphs
    :param s_high: upper bound for the size of subgraphs
    :param sample: ego or rw
    :param n_vec: num of eigenvector precomputed
    :param w_len: num of random walk length
    :param include_self: if True, include the eigendecomposition of origian graph
    :return: a list of PyG graphs
    """

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    assert dataset in ego_graphs
    if dataset == 'PubMed':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        g = Planetoid(path, dataset).data
    elif dataset == 'CoraFull':
        g = CoraFull(path).data
    elif dataset == 'Coauthor-physics':
        g = Coauthor(path, name='physics').data
    elif dataset == 'Coauthor-CS':
        g = Coauthor(path, name='CS').data
    elif dataset == 'Amazon-photo':
        g = Amazon(path, name='photo').data
    elif dataset == 'Amazon-computers':
        g = Amazon(path, name='computers').data
    elif dataset == 'yelp':
        g = Yelp(path, transform=None, pre_transform=None).data
    elif dataset == 'reddit':
        g = Reddit(path, transform=None, pre_transform=None).data
    elif dataset == 'flickr':
        g = Flickr(path, transform=None, pre_transform=None).data
    else:
        raise NotImplementedError

    graphs = []

    if sample == 'ego':
        indices = np.random.choice(range(g.num_nodes), replace=False, size=size)
        for idx in indices:
            print(f'Processing egograph at {idx}')
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(int(idx), hop, g.edge_index)
            g_sub = Data(edge_index=edge_index)
            n_sub = g_sub.num_nodes
            if n_sub < s_high and n_sub > s_low:
                g_sub = input_check(g_sub, size_check=False)
                graphs.append(g_sub)
            else:
                print(f'Skip egograph at {idx} due to size mismatch ({n_sub}). s_low {s_low}, s_high {s_high}.')

    elif sample == 'rw':
        from torch_geometric.data import GraphSAINTRandomWalkSampler
        g.edge_attr_ = g.edge_attr
        data = g  # deepcopy(g)
        data.edge_attr_ = g.edge_attr
        row, col = data.edge_index
        data.edge_attr = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
        w_len = 2000 if g.num_nodes < 10000 else w_len
        loader = GraphSAINTRandomWalkSampler(data, batch_size=1, walk_length=w_len,
                                             num_steps=size, sample_coverage=50,
                                             save_dir=None,
                                             num_workers=0)
        # parallel version
        gs = [g for g in loader][1:]
        for i, g in enumerate(gs):
            g.x, g.edge_attr = None, None
            gs[i] = g
        graphs = Parallel(n_jobs=1)(
            delayed(input_check)(g, size_check=False, k=n_vec) for g in gs)  # important: change back

        # serial version
        # for i, g_sub in enumerate(loader):
        #     if i == 0: continue
        #     g_sub.edge_attr = None
        #     g_sub.x = None
        #     graphs.append(input_check(g_sub, size_check=False, k=n_vec))
    else:
        raise NotImplementedError

    g.edge_attr = None
    if include_self:
        g = input_check(g, k=n_vec)
        summary(g, dataset)
        print([g] + graphs)
        return [g] + graphs  # add origianl graph as test
    else:
        print(graphs)
        return graphs


class EgoGraphs(InMemoryDataset):

    # @profile
    def __init__(self, transform=None, pre_transform=None, **kwargs):
        """ kwargs for function egographs """
        self.egograph_kwargs = kwargs
        root = os.path.join(egograph_dir, kwargs['dataset'])

        super(EgoGraphs, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return dict2name(self.egograph_kwargs)
        # return # ['wiki-vote_hop_3_size_5.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = egographs(**self.egograph_kwargs)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@timefunc
def precompute_eig(g, k=100):
    """ given a pyg graph, compute its eigenvalue and eigenvector for sym and None laplacian """
    k = min(k, g.num_nodes)
    threshold = 890000  # 89000 # important: change back to 89000
    laps = [None, ] if g.num_nodes > threshold else [None, 'sym']
    for lap in laps:  # didn't add rw since it's not symmetric
        t0 = time()
        L = get_laplacian_mat(g.edge_index, g.edge_weight, g.num_nodes, normalization=lap)
        scipy_lap = sparse_tensor2_sparse_numpyarray(L)
        kwargs = {'k': k, 'which': 'SM', 'tol': 1e-3}

        try:
            vals, vecs = eigsh(scipy_lap, **kwargs)
            vals, vecs = torch.FloatTensor(vals.real), torch.FloatTensor(vecs.real)
        except:
            msg = f'Fail to compute eigs using eigsh ({kwargs} for graph of size {g.num_nodes})'
            print(red(msg))
            vals, vecs = msg, msg

        g[f'{str(lap)}_vals'] = vals
        g[f'{str(lap)}_vecs'] = vecs
        print(f'takes {int(time() - t0)} to compute {lap}. k={k}.')
    return g


@timefunc
def input_check(g, size_check=True, eig=True, verbose=True, k=100):
    """  check input pyG graph g is connected
        and it is undirected
        weight is set
        remove unnecessary attributes

    :param eig: precompute eig
    """
    assert isinstance(g, Data), f'g is {type(g)}'
    allowed_attributes = ['edge_weight', 'edge_attr', 'edge_index', 'pos', ]
    allowed_attributes += ['sym_vecs', 'sym_vals', 'None_vecs', 'None_vals']

    if g.is_directed():
        warn('Convert directed pyG to undirect.')
        new_edge_index = to_undirected(g.edge_index)
        g.edge_index = new_edge_index

    if g.contains_isolated_nodes():
        num_comp(g)
        g = largest_cc(g)
        warn(f'input graph contains isolated nodes. Took the largest comp ({g.num_nodes})')

    g = __add_default_weight(g)
    if eig: g = precompute_eig(g, k=k)

    # remove unnecesseary attr
    for attr in g.keys:
        if attr not in allowed_attributes:
            g.__setitem__(attr, None)
            warn(f'Remove unnecessary attribute {attr}.')

    if 'sym_vec' not in g.keys:
        pass

    # set x attribute
    if 'pos' in g.keys:
        g.x = g.pos

    g = LocalDegreeProfile()(g)  # may need to change to other features
    g.x = torch.nn.functional.normalize(g.x, dim=0)

    if size_check and g.num_nodes < 200:
        exit(f'Graph size ({g.num_nodes}) being too small.')
    else:
        return g

    if verbose:
        summary(g)


@timefunc
def __add_default_weight(g, weight=1):
    """ given a pyG graph without edge weight, set all weights to be {weight}.
        if there is edge weight, do not modify it and return the original graph.
    """
    assert isinstance(g, Data)
    assert 'edge_index' in g.keys
    if 'edge_weight' not in g.keys:
        warn(f'No edge_weight found. Set edge_weight as {weight} by default')
        g.edge_weight = weight * torch.ones(g.edge_index.size(1))
    else:
        assert g.edge_weight.size(0) == g.edge_index.size(1)

    if 'edge_attr' not in g.keys:
        warn(f'No edge_attr found. Set edge_attr as {weight} by default')
        g.edge_attr = weight * torch.ones((g.edge_index.size(1), 1))

    return g


def syth_graphs(n=10, size=1000, type='er'):
    # generate a list of pyG synthetic graphs
    graphs = []
    geo_constant = size * (0.1) ** 2
    er_constant = size * 0.01

    for i in range(n):
        if type in ['er', 'random_er']:
            g = nx.erdos_renyi_graph(size, er_constant / size, seed=i)
            print(nx.info(g))
        elif type == 'geo':
            # g = nx.random_geometric_graph(size, 0.1, seed=i)
            g = nx.random_geometric_graph(size, np.sqrt(geo_constant / size), seed=i)
        elif type == 'ba':
            g = nx.barabasi_albert_graph(size, 4, seed=42)
        elif type == 'ws':
            g = nx.watts_strogatz_graph(size, 10, 0.1, seed=i)
        elif type == 'sbm':
            sizes = [int(size * 0.3), int(size * 0.7)]
            probs = [[0.04, 0.015],
                     [0.015, 0.04]]
            g = nx.stochastic_block_model(sizes, probs, seed=i)
        else:
            raise NotImplementedError
        size += 100
        graphs.append(from_networkx(g))
    graphs = [input_check(graph) for graph in graphs]
    return graphs


def set_loader(g, bs, shuffle=False, args=None):
    """
    :param g: pyG graph
    :param bs:
    :param shuffle:
    :return:
    """

    sub = subgraphs(g, assertion=True, args=args)
    summary(sub.g_sml, 'g_sml (in set loader)')

    keys = sub.double_edges
    subgraphs_list = sub.get_subgraphs(verbose=False, )
    data = list(zip(keys, subgraphs_list))
    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, num_workers=0)

    n_node_sml = sub.g_sml.num_nodes  # important: may be different from origianl n_node_sml
    return loader, sub, n_node_sml


class NonEgoGraphs(InMemoryDataset):
    """ similar to EgoGraphs, but for other graphs (sythetic graphs + Loukas's dataset ) """

    def __init__(self, dataset=None, transform=None, pre_transform=None):
        """ kwargs for function egographs """
        data_dir = os.path.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
        nonegograph_dir = os.path.join(data_dir, 'nonegographs')
        self.dataset = dataset
        root = os.path.join(nonegograph_dir, self.dataset)

        super(NonEgoGraphs, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return self.dataset  # dict2name(self.egograph_kwargs)

    def download(self):
        pass

    def _select_datasets(self):
        dataset = self.dataset
        if dataset == 'shape':
            datasets = shape_data(50, _k=10)
        elif dataset == 'random_geo':
            datasets = syth_graphs(n=50, size=700, type='geo')
        elif dataset in ['sbm', 'ws', 'ba', 'random_er']:
            datasets = syth_graphs(n=50, size=512, type=dataset)
        elif dataset in ['yeast', 'airfoil', 'bunny', 'minnesota']:
            datasets = loukas_data(name=dataset)
        elif dataset in ['amazons', 'coauthors']:
            datasets = MiscDataset().hybrid_graphs(name=dataset)
        else:
            raise NotImplementedError
        return datasets

    def process(self):
        # Read data into huge `Data` list.

        data_list = self._select_datasets()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class data_loader:
    def __init__(self, args, dataset='shape'):
        self.dataset = dataset
        train_indices = [int(item) for item in args.train_indices.split(',') if len(item) != 0]
        test_indices = [int(item) for item in args.test_indices.split(',') if len(item) != 0]
        self.n_graph = max(max(test_indices), max(train_indices)) + 1

        if args.debug:
            g = random_pygeo_graph(320, 1, 3000, 1, device='cpu')
            datasets = [g]
        else:
            if dataset in ['random_geo', 'random_er', 'sbm', 'ws', 'planetoid', 'yeast',
                           'airfoil', 'bunny', 'minnesota', 'ego_facebook', 'shape', 'ba', 'amazons', 'coauthors',
                           'faust']:
                datasets = NonEgoGraphs(dataset=dataset)
            elif dataset == 'pubmeds':
                # datasets = egographs(hop=4, size=5, dataset='PubMed')
                # kwargs = {'dataset': 'PubMed', 'hop': 5, 'size': 20, 's_low': 5000, 's_high': 10000}
                # kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample':'rw'}
                # datasets = EgoGraphs(**kwargs)[:self.n_graph]
                kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                          'sample': 'rw', 'n_vec': 500, 'w_len': args.w_len, 'include_self': False}
                datasets = EgoGraphs(**kwargs)[:self.n_graph]
                datasets = [data for data in datasets]

                kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1, 'sample': 'rw',
                          'include_self': True, 'n_vec': 500, 'w_len': 15000}
                datasets = [EgoGraphs(**kwargs)[0]] + datasets

            elif dataset == 'flickr':
                # load two dataset seprately, one include flickr itself, one include only subgraphs
                kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                          'sample': 'rw', 'n_vec': 500, 'w_len': args.w_len, 'include_self': False}
                datasets = EgoGraphs(**kwargs)[:self.n_graph]
                datasets = [data for data in datasets]

                # kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
                kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1,
                          'sample': 'rw', 'n_vec': 500, 'w_len': 15000, 'include_self': True}
                datasets = [EgoGraphs(**kwargs)[0]] + datasets

            elif dataset == 'reddit':
                kwargs = {'dataset': 'reddit', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
                datasets_ = EgoGraphs(**kwargs)
                datasets = datasets_[:1] + datasets_[2:3] + datasets_[2:15]
                del datasets_
                print(datasets)
                for d in datasets:
                    print(d)
                # exit()

            elif dataset == 'wiki-vote':
                kwargs = {'dataset': 'wiki-vote', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
                # kwargs = {'hop': 3, 'size': 5, 'dataset': 'wiki-vote', 's_low': 200, 's_high': 5000}
                datasets = EgoGraphs(**kwargs)
            elif dataset == 'citeseers':
                kwargs = {'dataset': 'CiteSeer', 'hop': 5, 'size': 20, 's_low': 200, 's_high': 5000}
                datasets = EgoGraphs(**kwargs)
            elif dataset == 'coauthor-cs':
                kwargs = {'dataset': 'Coauthor-CS', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw',
                          'n_vec': 800, 'w_len': args.w_len, }
                # kwargs = {'dataset': 'Coauthor-CS', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
                datasets = EgoGraphs(**kwargs)[:self.n_graph]

            elif dataset == 'coauthor-physics':
                kwargs = {'dataset': 'Coauthor-physics', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                          'sample': 'rw', 'n_vec': 800, 'w_len': args.w_len, }
                # kwargs = {'dataset': 'Coauthor-physics', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
                datasets = EgoGraphs(**kwargs)

            elif dataset == 'amazon-photo':
                kwargs = {'dataset': 'Amazon-photo', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                          'sample': 'rw'}
                # kwargs = {'dataset': 'Amazon-photo', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000} # both are not working
                datasets = EgoGraphs(**kwargs)
            elif dataset == 'amazon-computers':
                kwargs = {'dataset': 'Amazon-computers', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
                datasets = EgoGraphs(**kwargs)
            else:
                raise NotImplementedError

            assert max(args.train_idx, args.test_idx) < len(datasets), \
                f'Dataset is of len {len(datasets)}. Max idx is {max(args.train_idx, args.test_idx)}'

        print(f'Finish loading dataset {dataset} (len: {red(len(datasets))})')

        datasets = [handle_num_nodes(data) for data in datasets]
        datasets = [self.clip_feat(data, args, dim=args.n_bottomk) for data in datasets]
        datasets = datasets[:self.n_graph]

        self.datasets = datasets
        del datasets

    def select_kwargs_for_egographs(self, dataset='wiki-vote'):

        if dataset == 'pubmeds':
            # datasets = egographs(hop=4, size=5, dataset='PubMed')
            # kwargs = {'dataset': 'PubMed', 'hop': 5, 'size': 20, 's_low': 5000, 's_high': 10000}
            # kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample':'rw'}
            # datasets = EgoGraphs(**kwargs)[:self.n_graph]
            kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                      'sample': 'rw', 'n_vec': 500, 'w_len': args.w_len, 'include_self': False}
            datasets = EgoGraphs(**kwargs)[:self.n_graph]
            datasets = [data for data in datasets]

            kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1, 'sample': 'rw',
                      'include_self': True, 'n_vec': 500, 'w_len': 15000}
            datasets = [EgoGraphs(**kwargs)[0]] + datasets

        elif dataset == 'flickr':
            # load two dataset seprately, one include flickr itself, one include only subgraphs
            kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                      'sample': 'rw', 'n_vec': 500, 'w_len': args.w_len, 'include_self': False}
            datasets = EgoGraphs(**kwargs)[:self.n_graph]
            datasets = [data for data in datasets]

            # kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
            kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1,
                      'sample': 'rw', 'n_vec': 500, 'w_len': 15000, 'include_self': True}
            datasets = [EgoGraphs(**kwargs)[0]] + datasets

        elif dataset == 'reddit':
            kwargs = {'dataset': 'reddit', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
            datasets_ = EgoGraphs(**kwargs)
            datasets = datasets_[:1] + datasets_[2:3] + datasets_[2:15]
            del datasets_
            print(datasets)
            for d in datasets:
                print(d)
            # exit()

        elif dataset == 'wiki-vote':
            kwargs = {'dataset': 'wiki-vote', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw'}
            # kwargs = {'hop': 3, 'size': 5, 'dataset': 'wiki-vote', 's_low': 200, 's_high': 5000}
            datasets = EgoGraphs(**kwargs)
        elif dataset == 'citeseers':
            kwargs = {'dataset': 'CiteSeer', 'hop': 5, 'size': 20, 's_low': 200, 's_high': 5000}
            datasets = EgoGraphs(**kwargs)
        elif dataset == 'coauthor-cs':
            kwargs = {'dataset': 'Coauthor-CS', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw',
                      'n_vec': 800, 'w_len': args.w_len, }
            # kwargs = {'dataset': 'Coauthor-CS', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
            datasets = EgoGraphs(**kwargs)[:self.n_graph]

        elif dataset == 'coauthor-physics':
            kwargs = {'dataset': 'Coauthor-physics', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                      'sample': 'rw', 'n_vec': 800, 'w_len': args.w_len, }
            # kwargs = {'dataset': 'Coauthor-physics', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
            datasets = EgoGraphs(**kwargs)

        elif dataset == 'amazon-photo':
            kwargs = {'dataset': 'Amazon-photo', 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1,
                      'sample': 'rw'}
            # kwargs = {'dataset': 'Amazon-photo', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000} # both are not working
            datasets = EgoGraphs(**kwargs)
        elif dataset == 'amazon-computers':
            kwargs = {'dataset': 'Amazon-computers', 'hop': 4, 'size': 20, 's_low': 5000, 's_high': 10000}
            datasets = EgoGraphs(**kwargs)
        else:
            raise NotImplementedError

        assert max(args.train_idx, args.test_idx) < len(datasets), \
            f'Dataset is of len {len(datasets)}. Max idx is {max(args.train_idx, args.test_idx)}'

    def load(self, args, mode, verbose=False):
        assert mode in ['train', 'test']
        assert max(args.train_idx, args.test_idx) < len(self.datasets), \
            f'Dataset is of len {len(self.datasets)}. Max idx is {max(args.train_idx, args.test_idx)}'

        if mode == 'train':
            # important: to save memory, remove it when it is loaded. works only when each train graph is used once
            g = self.datasets[args.train_idx]
            self.datasets[args.train_idx] = None
        else:
            g = self.datasets[args.test_idx]
            print(f'Test: Load {red(args.test_idx)}-th shape from dataset {red(self.dataset)}.')

        if verbose: summary(g, 'original_graph')
        r = np.clip(args.ratio, 0, 0.999)
        n_node_sml = int(np.ceil((1 - r) * g.num_nodes))
        return g, n_node_sml

    @staticmethod
    def clip_feat(g, args, dim=50):
        # given a pyG graph, clip the precomputed feature(800 dim) to k-dim
        if args.lap in ['None', 'none', None]:
            g.None_vecs = g.None_vecs[:, :dim]
            if 'sym_vecs' in g: del g.sym_vecs
        elif args.lap == 'sym':
            g.None_vecs = g.sym_vecs[:, :dim]
            del g.None_vecs
        return g


@timefunc
def loukas_data(name='yeast'):
    """ dataset from locus's paper """
    # todo: make it faster

    g = graph_lib.real(-1, name)

    t0 = time()
    W = g.W.todense()
    g_nx = nx.from_numpy_array(W)
    pyG = from_networkx(g_nx)
    pyG.edge_weight = pyG.weight
    print(time() - t0)

    t1 = time()
    g = from_networkx(g_nx)
    print(time() - t1)

    g = input_check(g)
    return [g]


import argparse

parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--k', type=int, default=2, help='num of k')

if __name__ == '__main__':
    args = parser.parse_args()
    t0 = time()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'reddit')
    g = Reddit(path).data  # SNAPDataset(data_dir, 'wiki-vote')[0]
    summary(g)
    g = __add_default_weight(g)
    g = precompute_eig(g, k=args.k)
    summary(g)
    print(time() - t0)
    exit()

    graphs = shape_data(50, name='FAUST')
    for i, g in enumerate(graphs):
        summary(g, i)
    exit()
    # graphs = egographs(hop=4, size=20, dataset='PubMed')
    # egographs(dataset='PubMed', hop=5, size=20, s_low=5000, s_high=10000)
    graphs = shape_data(10, _k=10, name='FAUST')
    for g in graphs:
        print(g)
        print('-' * 10)

    exit()


    graphs = \
        ['minnesota', 'bunny', 'airfoil', 'yeast']
    # ['bunny']
    for name in graphs:
        banner(name)
        dataset = loukas_data(name=name)
        summary(dataset)

    exit()
    # _, shapes = simple_shape(k=10)  # shape_data(3, _k=5)
    shapes = shape_data(10, _k=10)
    exit()
    g = shape_data(3, _k=10)[2]  # simple_shape(k=10)[0]  #
    summary(g)

    loader, _ = set_loader(g, 500, 32)

    for batch in loader:
        indices_batch, graph_batch = batch
        summary(indices_batch, 'indices_batch')
        break

    for batch in loader:
        indices_batch, graph_batch = batch
        summary(indices_batch, 'indices_batch')
        break

    exit()

    data = shape_data(2)
    summary(data[0], 'data[0]')
    exit()

    for d in ['Cora', 'CiteSeer', 'PubMed']:
        data = planetoid(d)
        summary(data, d)
    exit()
