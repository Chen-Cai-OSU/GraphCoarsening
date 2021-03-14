# Created at 2020-06-06
# Summary: look at the eigenvalue of datasets
import argparse
import sys

import matplotlib.pyplot as plt

from sparsenet.model.loss import get_laplacian_mat
from sparsenet.util.data import planetoid, syth_graphs, shape_data, loukas_data, egographs, EgoGraphs, hybrid_graphs
from sparsenet.util.loss_util import vec_generator
from sparsenet.util.name_util import loukas_datasets
from sparsenet.util.util import summary, tonp


def get_eigenval(g, k=40, normalization='sym', v = False):
    # get eigenvalues from pyg graph
    L = get_laplacian_mat(g.edge_index, g.edge_weight, g.num_nodes, normalization=normalization)
    vals = vec_generator().bottomk_vec(L, None, k, val=True)
    if v: print(vals)
    summary(vals, 'vals')
    return vals


parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--lap', type=str, default='None', help='Laplacian type', choices=[None, 'sym', 'rw', 'none'])
parser.add_argument('--n_bottomk', type=int, default=50, help='Number of Bottom K eigenvector')
parser.add_argument('--data', type=str, default='PubMed', help='data')

from sparsenet.util.name_util import big_ego_graphs
if __name__ == '__main__':
    args = parser.parse_args()
    k = args.n_bottomk
    lap = None if args.lap == 'None' else args.lap

    # flickr
    # kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 5000, 's_low': -1, 's_high': -1, 'sample': 'rw'}
    n_vec = 500 if args.data in ['flickr', 'PubMed'] else 800
    size = 1 if args.data in ['flickr', 'PubMed'] else 50

    # kwargs = {'dataset': args.data, 'hop': -1, 'size': size, 's_low': -1, 's_high': -1,
    #           'sample': 'rw', 'n_vec': n_vec, 'w_len': 15000, 'include_self': True}

    kwargs = {'dataset': args.data, 'hop': -1, 'size': 50, 's_low': -1, 's_high': -1, 'sample': 'rw',
              'n_vec': 800, 'w_len': 5000, }

    data = EgoGraphs(**kwargs)
    # key = f'{str(args.lap)}_vals'
    key = 'None_vals'
    print(data[0])
    vals = tonp(data[0][key])
    summary(vals, 'vals')
    plt.plot(vals, label=f'{args.data} lap {args.lap}')
    plt.legend()
    plt.show()
    sys.exit()

    exit()
    for name in ['Amazon-photo', 'Coauthor-physics', 'Coauthor-CS']: # ['CiteSeer','PubMed', 'wiki-vote', 'Coauthor-CS', 'CoraFull', 'Amazon-photo']: #['wiki-vote']:
        s_low = 5000 if name in big_ego_graphs else 200
        s_high = 10000 if name in big_ego_graphs else 5000
        size = 20
        # if name in ['Coauthor-CS', 'CoraFull']: size = 3
        kwargs = {"hop": 3, "size": size, "dataset": name, 's_low': s_low, 's_high': s_high}
        data = EgoGraphs(**kwargs)
        key = f'{str(args.lap)}_vals'
        vals = tonp(data[0][key])
        summary(vals, 'vals')
        plt.plot(vals, label=f'{name}')

    plt.legend(loc='center right')
    plt.title(f'Eigenvalues of {name}. Laplacian: {lap}')
    plt.show()
    exit()

    for name in ['geo', 'sbm', 'ws', 'er']:
        for i in [10]:
            g =  syth_graphs(n=10, size=512, type=name)[0]
            vals = get_eigenval(g, k=50,  normalization=lap)
            plt.plot(vals, label=f'{name} {i}')
        plt.legend(loc='center right')
        plt.title(f'Eigenvalues of {name}. Laplacian: {lap}')
    plt.show()

    # planetoid
    name = 'planetoid'
    for i in range(3):
        g =  planetoid()[i]
        vals = get_eigenval(g, k=50,  normalization=lap)
        plt.plot(vals, label=f'{name} {i}')
    plt.legend(loc='center right')
    plt.title(f'Eigenvalues of {name}. Laplacian: {lap}')
    plt.show()

    # shape
    name = 'shape'
    for i in range(1, 20, 5):
        g =  shape_data(20, _k=10)[i]
        vals = get_eigenval(g, k=50,  normalization=lap)
        plt.plot(vals, label=f'{name} {i}')
    plt.legend(loc='center right')
    plt.title(f'Eigenvalues of {name}. Laplacian: {lap}')
    plt.show()

    # loukas data
    for name in loukas_datasets:
        g = loukas_data(name=name)[0]
        vals = get_eigenval(g, k=50, normalization=lap)
        plt.plot(vals, label=f'{name}')
    plt.legend(loc='center right')
    plt.title(f'Eigenvalues of {name}. Laplacian: {lap}')
    plt.show()

    sys.exit()
    for i in range(3):
        g = planetoid()[i] # syth_graphs(n=10, size=512, type='sbm')[0] # data_loader(None, dataset='sbm').load()[1]
        vals = get_eigenval(g)
        plt.plot(vals, label=f'planetoid {i}')
    plt.show()
    plt.legend(loc='center right')
