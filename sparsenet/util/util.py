# Created at 2020-04-16
# Summary: util functions
import collections
import math
import os
import random
import sys
import time
from functools import partial
from itertools import chain
from warnings import warn

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from colorama import init
from pygsp import graphs
from scipy.sparse import coo_matrix
from termcolor import colored
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, get_laplacian, from_networkx, to_networkx

nan1 = 0.12345
init()


def timefunc(method, threshold=1):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if int(te - ts) >= threshold:
                print(f'{method.__name__}: {pf(te - ts, precision=1)}s')
        return result

    return timed


tf = partial(timefunc, threshold=1)


def stats(x, precision=2, verbose=True, var_name='None'):
    """
    print the stats of a (np.array, list, pt.Tensor)

    :param x:
    :param precision:
    :param verbose:
    :return:
    """
    if isinstance(x, torch.Tensor): x = tonp(x)
    assert isinstance(x, (list, np.ndarray)), 'stats only take list or numpy array'

    ave_ = np.mean(x)
    median_ = np.median(x)
    max_ = np.max(x)
    min_ = np.min(x)
    std_ = np.std(x)
    pf_ = partial(pf, precision=precision)

    if verbose:
        ave_, min_, max_, median_, std_ = list(map(pf_, [ave_, min_, max_, median_, std_]))
        line = '{:>25}: {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(median) {:>8}(std)'.format(var_name, ave_, min_, max_,
                                                                                           median_, std_)
        print(line)

    return list(map(pf_, [ave_, min_, max_, median_, std_]))


def viz_graph(g, node_size=5, edge_width=1, node_color='b', color_bar=False, show=False):
    # g = nx.random_geometric_graph(100, 0.125)
    pos = nx.spring_layout(g)
    nx.draw(g, pos, node_color=node_color, node_size=node_size, with_labels=False, width=edge_width)
    if color_bar:
        # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(node_color), vmax=max(node_color)))
        sm._A = []
        plt.colorbar(sm)
    if show: plt.show()


def largest_cc(g):
    isinstance(g, Data)
    g = to_networkx(g).to_undirected()
    subgraphs = [g.subgraph(c).copy() for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    largest_cc = subgraphs[0]
    g = from_networkx(largest_cc)
    return g


def num_comp(g):
    assert isinstance(g, Data)
    g_nx = to_networkx(g).to_undirected()
    n_compoent = nx.number_connected_components(g_nx)

    comp_size = [len(c) for c in nx.connected_components(g_nx)]
    comp_size = sorted(comp_size, reverse=True)
    if n_compoent > 1:
        if n_compoent < 10:
            print(comp_size)
        else:
            print(f'Print size of first 10 compoents: {comp_size[:10]}')

    # assert n_compoent == 1, f'number of component is {n_compoent}'
    return n_compoent


def random_pygeo_graph(n_node, node_feat_dim, n_edge, edge_feat_dim, device='cpu', viz=False):
    """ random DIRECTED pyG graph """
    g = Data(x=torch.rand(n_node, node_feat_dim),
             edge_index=random_edge_index(n_edge, n_node),
             edge_attr=torch.rand(n_edge, edge_feat_dim).type(torch.LongTensor),
             edge_weight=torch.ones(n_edge))

    g_nx = to_networkx(g).to_undirected()
    n_compoent = nx.number_connected_components(g_nx)
    if n_compoent > 1 and viz: viz_graph(g_nx, show=True)
    assert n_compoent == 1, f'number of component is {n_compoent}'
    g = g.to(device)
    return g


def maybe_edge_weight(g):
    """ used for get_laplacian.
        edge_weigher will update edge weights, which is saved in g.edge_weight attribute
        get_laplacian will try to retrive latest g.edge_weight to compute loss
     """
    assert isinstance(g, torch_geometric.data.data.Data)
    try:
        return g.edge_weight
    except AttributeError:
        warn('Use default edge weight')
        return None


def random_edge_index(n_edge=200, n_node=20):
    """ generate random edge tensor of shape (2, n_edge) """
    assert n_edge % 2 == 0
    assert n_edge <= n_node * (n_node - 1), f'n_edge: {n_edge}; n_node: {n_node}'
    edges = []
    for i in range(n_edge // 2):
        a, b = np.random.choice(n_node, 2, replace=False).tolist()
        while (a, b) in edges:
            a, b = np.random.choice(n_node, 2, replace=False).tolist()
        edges.append((a, b))
        edges.append((b, a))
    edges = list(edges)
    edges = torch.LongTensor(np.array(edges).T)
    return edges


def random_edge_weight(n_edges):
    """
    :param edges: [2, n] tensor (output of random_edge_index)
    :return:
    """
    weights = []
    assert n_edges % 2 == 0
    for i in range(n_edges // 2):
        w = np.random.random()
        weights.append(w)
        weights.append(w)
    return torch.Tensor(weights)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    https://bit.ly/2YHzUYK

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def add_range(r1, r2):
    concatenated = chain(r1, r2)
    return concatenated


def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def banner(text='', ch='=', length=140, compact=False):
    """ http://bit.ly/2vfTDCr
        print a banner
    """
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    print(banner)
    if not compact:
        print()


def pf(nbr, precision=1):
    """ precision format """
    # assert type(nbr)==float
    if isinstance(nbr, torch.Tensor):
        nbr = np.float(nbr)

    if math.isnan(nbr):
        return 'nan'
    elif math.isinf(nbr):
        return 'inf'
    else:
        return round(nbr * (10 ** precision)) / (10 ** precision)


def set_thread(n=1):
    import os
    os.environ['MKL_NUM_THREADS'] = str(n)

    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)
    import torch
    torch.set_num_threads(n)


@timefunc
def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
        return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr


def nan_ratio(x):
    """ http://bit.ly/2PL7yaP
    """
    assert isinstance(x, np.ndarray)
    try:
        return np.count_nonzero(np.isnan(x)) / x.size
    except TypeError:
        return '-1 (TypeError)'


import scipy


def np2set(x):
    assert isinstance(x, np.ndarray)
    return set(np.unique(x))


@timefunc
def summary(x, name='x', terminate=False,
            skip=False, delimiter=None, precision=3,
            exit=False, highlight=False):
    if highlight:
        name = red(name)

    if skip:
        print('', end='')
        return ''

    if isinstance(x, list):
        print(f'{name}: a list of length {len(x)}')

        if len(x) < 6:
            for _x in x:
                summary(_x)

    elif isinstance(x, scipy.sparse.csc.csc_matrix):
        min_, max_ = x.min(), x.max()
        mean_ = x.mean()

        std1 = np.std(tonp(x))
        x_copy = x.copy()
        x_copy.data **= 2
        std2 = x_copy.mean() - (x.mean() ** 2)  # todo: std1 and std2 are different. 1
        pf_ = partial(pf, precision=precision)
        mean_, min_, max_, std1, std2 = list(map(pf_, [mean_, min_, max_, std1, std2]))

        line0 = '{:>10}: csc_matrix ({}) of shape {:>8}'.format(name, str(x.dtype), str(x.shape))
        line0 = line0 + ' ' * max(5, (45 - len(line0)))
        # line0 += 'Nan ratio: {:>8}.'.format(nan_ratio(x_))
        line1 = '  {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(std1) {:>8}(std2) {:>8}(unique) ' \
            .format(mean_, min_, max_, std1, std2, -1)
        line = line0 + line1
        print(line)

    elif isinstance(x, (np.ndarray,)):
        if x.size > 232960 * 10:
            return
        x_ = tonp(x)
        ave_ = np.mean(x_)
        median_ = np.median(x_)
        max_ = np.max(x_)
        min_ = np.min(x_)
        std_ = np.std(x_)
        unique_ = len(np.unique(x_))
        pf_ = partial(pf, precision=precision)
        ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))

        line0 = '{:>10}: array ({}) of shape {:>8}'.format(name, str(x.dtype), str(x.shape))
        line0 = line0 + ' ' * max(5, (45 - len(line0)))
        line0 += 'Nan ratio: {:>8}.'.format(nan_ratio(x_))
        line1 = '  {:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(median) {:>8}(std) {:>8}(unique) '.format(ave_, min_, max_,
                                                                                                     median_, std_,
                                                                                                     unique_)
        line = line0 + line1
        if np2set(x_) <= set([-1, 0, 1]):
            ratio1 = np.sum(x_ == 1) / float(x_.size)
            ratio0 = np.sum(x_ == 0) / float(x_.size)
            line += '|| {:>8}(1 ratio) {:>8}(0 ratio)'.format(pf(ratio1, 3), pf(ratio0, 3))

        if nan1 in x_:
            nan_cnt = np.sum(x_ == nan1)
            line += f'nan_cnt {nan_cnt}'

        # f'{name}: array of shape {x.shape}.'
        print(line)
        # print(f'{name}: a np.array of shape {x.shape}. nan ratio: {nan_ratio(x)}. ' + line)

    elif isinstance(x, (torch.Tensor)):
        if x.numel() > 232965 * 10:
            return
        x_ = tonp(x)
        if len(x_) == 0:
            print(f'{name}: zero length np.array')
        else:
            ave_ = np.mean(x_)
            median_ = np.median(x_)
            max_ = np.max(x_)
            min_ = np.min(x_)
            std_ = np.std(x_)
            unique_ = len(np.unique(x_))

            pf_ = partial(pf, precision=2)
            ave_, min_, max_, median_, std_, unique_ = list(map(pf_, [ave_, min_, max_, median_, std_, unique_]))
            line = '{:>8}(mean) {:>8}(min) {:>8}(max) {:>8}(median) {:>8}(std) {:>8}(unique)'.format(ave_, min_, max_,
                                                                                                     median_, std_,
                                                                                                     unique_)

            print(
                '{:20}'.format(name) + '{:20}'.format(str(x.data.type())[6:]) + '{:15}'.format(
                    str(x.size())[11:-1]) + line)
        # print(line)
        # print(f'{name}: a Tensor ({x.data.type()}) of shape {x.size()}')

    elif isinstance(x, tuple):
        print(f'{name}: a tuple of shape {len(x)}')
        if len(x) < 6:
            for ele in x:
                summary(ele, name='ele')

    elif isinstance(x, (dict, collections.defaultdict)):
        print(f'summarize a dict {name} of len {len(x)}')
        for k, v in x.items():
            # print(f'key is {k}')
            summary(v, name=k)

    elif isinstance(x, torch_geometric.data.data.Data):
        try:
            summary_pygeo(x, name=name)
        except:
            raise Exception('Check pytorch geometric install.')

    elif isinstance(x, pd.DataFrame):
        from collections import OrderedDict

        dataType_dict = OrderedDict(x.dtypes)
        banner(text=f'start summarize a df ({name}) of shape {x.shape}', ch='-')
        print('df info')
        print(x.info())
        print('\n')

        print('head of df:')
        # print(tabulate(x, headers='firstrow'))
        print(x.head())
        print('\n')

        try:
            print('continuous feats of Dataframe:')
            cont_x = x.describe().T
            print(cont_x)
            print(cont_x.shape)
            print('\n')
        except ValueError:
            print('x.describe().T raise ValueError')

        try:
            print('non-cont\' feats (object type) of Dataframe:')
            non_cont = x.describe(include=[object]).T
            print(non_cont)
            print(non_cont.shape)
        except ValueError:
            print('x.describe(include=[object]).T raise ValueError')

        banner(text=f'finish summarize a df ({name}) of shape {x.shape}', ch='-')

    elif isinstance(x, (int, float)):
        print(f'{name}(float): {x}')

    elif isinstance(x, str):
        print(f'{name}(str): {x}')

    else:
        print(f'{x}: \t\t {type(x)}')
        if terminate:
            exit(f'NotImplementedError for input {type(x)}')
        else:
            pass

    if delimiter is not None:
        assert isinstance(delimiter, str)
        print(delimiter)

    if exit:
        sys.exit()


def dict2name(d):
    """
    :param d: {'n_epoch': 300, 'bs': 32, 'n_data': 10, 'scheduler': True}
    :return: bs_32_n_data_10_n_epoch_300_scheduler_True
    """
    assert isinstance(d, dict)
    keys = list(d.keys())
    keys.sort()
    name = ''
    for k in keys:
        name += f'{k}_{d[k]}_'
    return name[:-1]


def update_dict(d1, d2):
    # use d1 to update d2, return updated d2.
    # keys of d1 has to be a subset of keys of d2.
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert set(d1.keys()) <= set(d2.keys()), 'Keys of d1 has to be a subset of keys of d2.'
    for k, v in d1.items():
        d2[k] = v
    return d2


def hasany(s, s_list):
    """
    :param s: a string
    :param s_list: a list of str
    :return:
    """
    return any(ele in s for ele in s_list)


def slicestr(s, f=None, t=None):
    """
    :param s: a string
    :param f: from
    :param t: to
    :return:
    """
    from_idx = s.index(f)
    to_idx = s.index(t)
    return s[from_idx:to_idx]


def summary_pygeo(data, stat=False, precision=2, name=None):
    assert isinstance(data, torch_geometric.data.data.Data)
    print(f'Summary of {name} (torch_geometric.data.data.Data):')

    for k, v in data:
        print('     ', sep=' ', end=' ')
        if isinstance(v, torch.Tensor):
            if v.ndim == 1:
                summary(v, name=k, precision=precision)
            else:
                if v.size()[1] != 0:
                    summary(v, name=k, precision=precision)
                else:
                    warn(f'Empty edge index: {v}')
        elif isinstance(v, str):
            summary(v, k)
        else:
            NotImplementedError

    if stat:
        for k, v in data:
            stats(v, var_name=k)


def subset_graph(g, indices, relabel_nodes=None):
    """
    :param g: pyG graph where node index are contigious
    :param indices:
    :param relabel_nodes: if true, relabel nodes of the subgraph
    :return:
    """
    if isinstance(indices, torch.Tensor): indices = indices.tolist()
    if isinstance(indices, set): indices = list(indices)

    assert isinstance(indices, list)
    assert isinstance(g, torch_geometric.data.data.Data)

    sub_edge_index, sub_edge_attr = subgraph(indices, g.edge_index, g.edge_attr, relabel_nodes=relabel_nodes)
    g_subindices = torch.tensor(indices)
    g_subx = g.x.index_select(0, g_subindices)
    g_sub = Data(x=g_subx, edge_index=sub_edge_index, edge_attr=sub_edge_attr, node_index=g_subindices)
    return g_sub


def assert_nonan(x):
    res = torch.isnan(x)
    assert (res == False).all(), 'contains Nan'


def make_dir(dir):
    # has side effect

    if dir == None:
        return

    if not os.path.exists(dir):
        os.makedirs(dir)


def args_print(args, one_line=False):
    """ pretty print cmd with lots of args
    """
    for i in range(20):
        args = args.replace('  ', ' ')

    arglis = args.split(' ')
    new_arglist = []
    for i, token in enumerate(arglis):
        if '--' in token:
            token = '\n' + token
        elif token in ['-u', 'nohup']:
            pass
        elif '.py' in token:
            pass
        elif 'python' in token:
            pass
        else:
            space = (30 - len(arglis[i - 1])) * ' '
            token = space + token  # '{:>35}'.format(token) #
        new_arglist.append(token)

    newargs = ' '.join(new_arglist) + '\n'

    if not one_line:
        print(newargs)
    else:
        newargs = one_liner(newargs)
        print(newargs)


def one_liner(cmd):
    """ convert cmd that takes many lines into just one line """
    assert isinstance(cmd, str)
    cmd = cmd.replace('\n', '')
    for _ in range(10):
        cmd = cmd.replace('  ', ' ')
    return cmd


def sig_dir():
    from sparsenet.util.dir_util import DIR
    return DIR


def fig_dir():
    return f'{sig_dir()}sparsenet/paper/tex/Figs/'


def tb_dir():
    return f'{tb_dir()}/result/tensorboardx/'


def model_dir():
    dir = f'{sig_dir()}result/model/'
    make_dir(dir)
    return dir


def red(x):
    return colored(x, "red")


def tex_dir():
    tex_dir = f'{sig_dir()}sparsenet/paper/tex/iclr_table/'
    make_dir(tex_dir)
    return tex_dir


def random_laplacian(n):
    from torch_geometric.utils.random import erdos_renyi_graph
    edge_index = erdos_renyi_graph(n, 0.1)
    i, v = get_laplacian(edge_index, None, normalization=None)
    return i, v


def runcmd(cmd, print_only=False):
    cmd = cmd.replace('--', ' --')
    banner('Execution of following cmds:', compact=True)
    if len(cmd) > 50 and '--' in cmd:
        args_print(cmd)
    else:
        print(cmd)

    if not print_only:
        os.system(cmd)

    if len(cmd) > 50 and '--' in cmd:
        args_print(cmd)


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


def pyg2gsp(g):
    """
    convert pyG graph to gsp graph.
    discard any info from pyG graph, and only take graph topology.
     """
    assert isinstance(g, torch_geometric.data.Data)
    edge_indices, edge_weight = tonp(g.edge_index), tonp(g.edge_weight)
    row, col = edge_indices[0, :], edge_indices[1, :]

    # memory efficient
    n = g.num_nodes
    W = scipy.sparse.csr_matrix((edge_weight, (row, col)), shape=(n, n))
    gspG = graphs.Graph(W)
    return gspG


def dic2tsr(d, dev='cuda'):
    """ given a dict where key (size N) are consecutive numbers and values are also numbers (at most n),
        convert it into a tensor of size (N) where index is the key value is the value of d.
    """
    N = len(d)
    assert N == max(d.keys()) + 1, f'keys ({N}) are not consecutive. Max key is {max(d.keys)}'
    tsr = [0] * N
    for k in d:
        tsr[k] = d[k]
    return torch.tensor(tsr).to(dev)


if __name__ == '__main__':
    from scipy.sparse import csc_matrix

    n = 400
    x = csc_matrix((n, n), dtype=np.int8)
    print(x.mean())
    print(x.max())
    print(x.min())

    std1 = np.std(tonp(x))
    x_copy = x.copy()
    x_copy.data **= 2
    std2 = x_copy.mean() - x.mean() ** 2
    print(std1, std2)
    # summary(x, 'x')

    exit()
    # edge = random_edge_index(n_edge=200, n_node=20)
    from sparsenet.util.sample import sample_N2Nlandmarks
    from sparsenet.util.graph_util import __map_back_from_edge, get_bipartite

    n_node, n_edge = 320, 1000
    node_dim = 1
    edge_feat_dim = 1

    g = Data(x=torch.rand(n_node, node_dim),
             edge_index=random_edge_index(n_edge, n_node),
             edge_attr=torch.rand(n_edge, edge_feat_dim).type(torch.LongTensor))
    summary(g, 'original_graph')

    G = to_networkx(g)
    G_prime, Assignment = sample_N2Nlandmarks(G, 10)
    print(G_prime.edges.data())
    edges = [e for e in G_prime.edges]
    print('Select edge:', edges[0])
    G1, G2, crossing_edges = __map_back_from_edge(g, edges[0], Assignment)
    summary(G1, 'G1')
    summary(G2, 'G2')
    print('Crossing edge size:', crossing_edges[0].shape)
    summary(get_bipartite(G1, G2, crossing_edges))
