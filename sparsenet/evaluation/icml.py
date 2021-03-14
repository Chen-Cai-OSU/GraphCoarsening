# Created at 2020-05-11
# Summary: small experiments for icml
import sys

import networkx as nx

from sparsenet.evaluation.icml.util import low_eig
from sparsenet.evaluation.metric import eig
from sparsenet.evaluation.sanity_check import energy
from sparsenet.model.loss import get_laplacian_mat
from sparsenet.util.model_util import random_pyG
import matplotlib.pyplot as plt
import numpy as np
from sparsenet.util.util import summary, tonp, fix_seed, timefunc
import argparse
from matplotlib import rc

# https://bit.ly/3cv2mkB
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

parser = argparse.ArgumentParser(description='sanity check')
parser.add_argument('--n_edge_rm', type=int, default=1, help='')
parser.add_argument('--graph', type=str, default='GEO', help='')
parser.add_argument('--n_eig', type=int, default=20, help='')
parser.add_argument('--action', type=str, default='drop', help='', choices=['drop', 'increase'])

parser.add_argument('--eigenvalue', action='store_true')
parser.add_argument('--energy', action='store_true')
parser.add_argument('--coef', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--show', action='store_true')

parser.add_argument('--loweig', action='store_true', help='Use combination of lower eigenvector') # todo: more on this
parser.add_argument('--gaussian', action='store_true', help='Use Gaussian Random vector')
parser.add_argument('--middle', action='store_true', help='look at middle eigenvalues')

@timefunc
def plot(g, n_edges_to_change, fig_num = 0, verbose = False, legend=False, n_eig = 20,
         energy_=False, eigen=False, action = 'drop', coef = False, name='cora'):

    fix_seed()

    if action == 'drop':
        g_drop = helper.rm_pyG_edges(g, n=n_edges_to_change)  # process_nx_graph(g_drop, add_weight=True)
    elif action == 'increase':
        g_drop = helper.increase_random_edge_w(g, n=n_edges_to_change, w=10000)
    else:
        NotImplementedError

    if verbose:
        summary(g, 'g')
        summary(g_drop, 'g_drop')

    L1 = get_laplacian_mat(g.edge_index, g.edge_weight, n_node, normalization='sym')
    L2 = get_laplacian_mat(g_drop.edge_index, g_drop.edge_weight, n_node, normalization='sym')

    if eigen or coef: # no need to compute eigenvector for energy
        w1, v1 = eig(L1)
        w2, v2 = eig(L2)

    if eigen:
        # eig value

        # ax[fig_num].plot()
        # ax[fig_num].hist(w1, label=f'$w_1$')
        # ax[fig_num].hist(w2, label=f'$w_2$')

        if args.middle:
            diff = w2 - w1
            ax[fig_num].scatter(range(len(diff)), diff, marker='o')
            middle = n_node // 2
            # ax[fig_num].plot(w1[middle-n_eig: middle+n_eig], marker='o', label=f'$w_1 middle {middle}$.')
            # ax[fig_num].plot(w2[middle-n_eig: middle+n_eig], marker='o', label=f'$w_2 middle {middle}$.')
        else:
            if n_eig != len(w1):
                ax[fig_num].plot(w1[:n_eig], marker='o', label=f'$w_1$. First {n_eig}')
                ax[fig_num].plot(w2[:n_eig], marker='o', label=f'$w_2$. First {n_eig}')
                ax[fig_num].plot(w1[-n_eig:], marker='o', label=f'$w_1$. Last {n_eig}')
                ax[fig_num].plot(w2[-n_eig:], marker='o', label=f'$w_2$. Last {n_eig}')
            else:
                ax[fig_num].plot(w1[:n_eig], marker='o', label=f'$w_1$.')
                ax[fig_num].plot(w2[:n_eig], marker='o', label=f'$w_2$.')


        percent = int(100 * n_edges_to_change / (g.num_edges // 2))
        if percent % 10 == 1: percent -= 1
        if percent % 10 == 9: percent += 1

        title = f'{action} {percent}\% edges.'
        ax[fig_num].set_title(title)
        ax[fig_num].set_ylim([0, 2])
        # ax[fig_num].set_yscale('log', basey=2)
        if legend:
            ax[fig_num].legend(loc='center right')

    if coef:
        fix_seed()

        if args.gaussian:
            v = np.random.normal(0, 1, (g.num_nodes, 3))
        else:
            v = np.random.random((g.num_nodes, 3))
        # v = np.random.random((g.num_nodes, 3))

        v_sm = {'v1_sm0': v, 'v2_sm0': v}
        for i in range(3):
            tmp1 = np.dot(np.identity(len(w1)) - tonp(L1), v_sm[f'v1_sm{i}'])
            tmp2 = np.dot(np.identity(len(w2)) - tonp(L2), v_sm[f'v2_sm{i}'])
            v_sm[f'v1_sm{i + 1}'] = tmp1
            v_sm[f'v2_sm{i + 1}'] = tmp2


        v2_coef0 = np.dot(v2.T, v_sm['v2_sm0'])
        v2_coef1 = np.dot(v2.T, v_sm['v2_sm1'])
        v2_coef2 = np.dot(v2.T, v_sm['v2_sm2'])

        v1_coef0 = np.dot(v1.T, v_sm['v1_sm0'])
        v1_coef1 = np.dot(v1.T, v_sm['v1_sm1'])
        v1_coef2 = np.dot(v1.T, v_sm['v1_sm2'])

        # summary(coef0, 'coef0')
        # summary(coef1, 'coef1')
        # summary(coef2, 'coef2')
        # print()

        # ax[fig_num].scatter(range(n_eig-1), v1_coef0[1:, 0],   s=1, label=f'$v1_sm0$')
        ax[fig_num].scatter(range(n_eig-1), v1_coef1[1:, 0],   s=5, label=f"$c_1$")
        # ax[fig_num].scatter(range(n_eig-1), v1_coef2[1:, 0],   s=5, label=f'$v1_sm2$')

        # ax[fig_num].scatter(range(n_eig - 1), v2_coef0[1:, 0], s=1, label=f'$v2_sm0$')
        ax[fig_num].scatter(range(n_eig - 1), v2_coef1[1:, 0], s=5, label=f"$c_1'$")
        # ax[fig_num].scatter(range(n_eig - 1), v2_coef2[1:, 0], s=5, label=f'$v2_sm2$')


        # ax[fig_num].scatter(range(n_eig-1), coef3[1:, 0],   label=f'v_sm3')

        ax[fig_num].set_ylim([-1, 1])
        percent = int(100 * n_edges_to_change / (g.num_edges // 2))
        if percent % 10 == 1: percent -= 1
        if percent % 10 == 9: percent += 1
        title = f'{action} {percent}\% edges.'
        ax[fig_num].set_title(title)
        if legend:
            ax[fig_num].legend(loc='center right')

    if energy_:
        # signal

        fix_seed()

        if args.gaussian:
            v = np.random.normal(0, 1, (g.num_nodes, 20))
        elif args.loweig:
            k = 30
            if name == 'cora': k = 400
            if name == 'citeseer': k = 400
            print(k)

            v = low_eig(g).mix_low_eig(k=k, n_vec=20, mode='full')
            summary(v)
        else:
            v = np.random.random((g.num_nodes, 20))
        # v = normalize(v, axis=0)
        v_sm = {'v1_sm0': v, 'v2_sm0': v}
        for i in range(2):
            tmp1 = np.dot(np.identity(n_node) - tonp(L1), v_sm[f'v1_sm{i}'])
            tmp2 = np.dot(np.identity(n_node) - tonp(L2), v_sm[f'v2_sm{i}'])

            v_sm[f'v1_sm{i + 1}'] = tmp1
            v_sm[f'v2_sm{i + 1}'] = tmp2

        D1_sm0 = energy(v_sm['v1_sm0'], L1)
        D2_sm0 = energy(v_sm['v2_sm0'], L2)

        D1_sm = energy(v_sm['v1_sm1'], L1)
        D2_sm = energy(v_sm['v2_sm1'], L2)

        D1_sm2 = energy(v_sm['v1_sm2'], L1)
        D2_sm2 = energy(v_sm['v2_sm2'], L2)

        ax[fig_num].scatter(range(n_eig), D1_sm0[:n_eig],  label=f'$E_0$')
        ax[fig_num].scatter(range(n_eig), D2_sm0[:n_eig], label=f"$E'_0$")

        ax[fig_num].scatter(range(n_eig), D1_sm[:n_eig], label=f'$E_1$')
        ax[fig_num].scatter(range(n_eig), D2_sm[:n_eig], label=f"$E'_1$")

        ax[fig_num].scatter(range(n_eig), D1_sm2[:n_eig], label=f'$E_2$')
        ax[fig_num].scatter(range(n_eig), D2_sm2[:n_eig], label=f"$E'_2$")

        percent = int(100 * n_edges_to_change / (g.num_edges//2))
        if percent % 10 == 1: percent -= 1
        if percent % 10 == 9: percent += 1

        title = f'{action} {percent}\% edges.'
        ax[fig_num].set_title(title)
        ax[fig_num].set_yscale('log', basey=10)

        # set y axis scale
        if args.gaussian:
            if name not in ['cora', 'citeseer']: ax[fig_num].set_ylim([10**(-3), 10**3])
            if name in ['cora']: ax[fig_num].set_ylim([10 ** (2.5), 10 ** 4])
            if name in ['citeseer']: ax[fig_num].set_ylim([10 ** (2.9), 10 ** 4])
        elif args.loweig and name in ['cora', 'citeseer']:
            pass
        else:
            if name not in ['cora', 'citeseer']: ax[fig_num].set_ylim([10**(-3), 10**1.5])
            if name in ['cora']: ax[fig_num].set_ylim([10**(1.5), 10**3])
            if name in ['citeseer']: ax[fig_num].set_ylim([10 ** (1.9), 10 ** 3])
        if legend:
            ax[fig_num].legend(loc='center right')

def data_dir(action='drop'):
    return f'/home/cai.507/Documents/DeepLearning/sparsifier/sparsenet/data/icml_grl/fig/{action}/'

if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed()
    helper = random_pyG()
    n_node = 200
    # n_edges_to_change = args.n_edge_rm

    if args.graph == 'GEO':
        g_nx = nx.random_geometric_graph(n_node, 0.2, seed=42)
        g = helper.process_nx_graph(g_nx, add_weight=True)
        name = 'Random Geometric Graph'
    elif args.graph == 'ER':
        g_nx = nx.erdos_renyi_graph(n_node, 0.05, seed=42)
        g = helper.process_nx_graph(g_nx, add_weight=True)
        name = f'Erdos Renyi Graph G({n_node}, 0.05)'
    elif args.graph == 'WS':
        g_nx = nx.watts_strogatz_graph(n_node, 20, p=0.05, seed=42)
        g = helper.process_nx_graph(g_nx, add_weight=True)
        name = 'Watts-Strogatz Graph'
    elif args.graph == 'SBM2':
        sizes = [100, 100]
        n_node = np.sum(sizes)
        probs = [[0.1, 0.01],
                 [0.01, 0.1]]
        g_nx = nx.stochastic_block_model(sizes, probs, seed=0)
        name = 'Stochastic Block Model with 2 Blocks.'
        g = helper.process_nx_graph(g_nx, add_weight=True)
    elif args.graph == 'SBM4':
        sizes = [50, 50, 50, 50]
        n_node = np.sum(sizes)
        probs = [[0.1, 0.008, 0.008, 0.008],
                 [0.008, 0.2, 0.008, 0.008],
                 [0.008, 0.008, 0.3, 0.008],
                 [0.008, 0.008, 0.008, 0.4]]
        g_nx = nx.stochastic_block_model(sizes, probs, seed=0)
        name = 'Stochastic Block Model with 4 Blocks.'
        g = helper.process_nx_graph(g_nx, add_weight=True)
    elif args.graph == 'complete':
        name = 'Complete Graph '
        n_node = 50
        g_nx = nx.complete_graph(n = n_node)
        g = helper.process_nx_graph(g_nx, add_weight=False, uniform_weight=True)
    elif args.graph == 'BA':
        name = 'Barabasi-Albert Graph'
        n_node = 200
        g_nx = nx.barabasi_albert_graph(200, 4, seed=42)
        g = helper.process_nx_graph(g_nx, add_weight=False, uniform_weight=True)
    elif args.graph == 'cora':
        name = 'Cora'
        g = helper.get_planetoid('cora')
        n_node = g.num_nodes
        g = helper.process_nx_graph(g, add_weight=False, uniform_weight=True)
    elif args.graph == 'citeseer':
        name = 'CiteSeer'
        g = helper.get_planetoid('citeseer')
        n_node = g.num_nodes
        g = helper.process_nx_graph(g, add_weight=False, uniform_weight=True)
    elif args.graph == 'pubmed':
        name = 'PubMed'
        g = helper.get_planetoid('pubmed')
        n_node = g.num_nodes
        g = helper.process_nx_graph(g, add_weight=False)
    else:
        NotImplementedError

    # print(name)
    # viz_graph(g_nx)
    # sys.exit()

    n_edges = [.1, .2, .3, .4, .5] + [.6, .7, .8, .9]
    n_edges = [int(ratio * g.num_edges // 2) for ratio in n_edges]
    fig, ax = plt.subplots(1, len(n_edges), figsize=(4*len(n_edges), 4))

    for (i, n_edges_to_change) in enumerate(n_edges):
        legend = True if i % 3 == 2 else False
        plot(g, n_edges_to_change, fig_num=i, n_eig=args.n_eig, legend=legend, eigen=args.eigenvalue,
             energy_=args.energy, verbose = args.verbose, action=args.action, coef = args.coef, name=args.graph)

    main_title = f"{name}: {n_node} nodes, {g.num_edges//2} edges."

    if args.coef or args.energy:
        if args.gaussian:
            main_title = main_title + ' Gaussain Random Vector $G(0, 1)$.'
        elif args.loweig:
            main_title = main_title
        else:
            main_title = main_title + ' Uniform Random Vector $U(0, 1)$.'
    plt.suptitle(main_title, fontsize=14)
    if args.show:
        plt.show()

    if args.eigenvalue: fname = f'{args.graph}_eig'
    if args.energy:
        fname = f'{args.graph}_energy'
    if args.coef:
        fname = f'{args.graph}_coef'

    if args.coef or args.energy:
        if args.gaussian:
            fname = fname + '_gaussian'
        elif args.loweig:
            fname = fname + '_loweig'
        else:
            fname = fname + '_uniform'

    print(fname)
    # plt.savefig(f'{data_dir(action=args.action)}{fname}.jpg')
    # plt.savefig(f'{data_dir(action=args.action)}{fname}.png', bbox_inches='tight')
    fname = f'{data_dir(action=args.action)}{fname}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    print(f'Finish saving at {fname}')

