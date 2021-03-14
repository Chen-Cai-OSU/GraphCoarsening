# Created at 2020-05-09
# Summary: try to understand how changing weight of ONE edge affect the laplacian spectrum

import matplotlib.pyplot as plt
import numpy as np
import scipy

from sparsenet.evaluation.metric import eig
from sparsenet.model.loss import get_laplacian_mat
from sparsenet.util.model_util import random_pyG
from sparsenet.util.util import summary, random_pygeo_graph, fix_seed, pf, tonp, random_edge_weight, \
    angle_between

# plt.style.use('seaborn-whitegrid')
# plt.style.use('ggplot')
from sklearn.preprocessing import normalize
import argparse

parser = argparse.ArgumentParser(description='sanity check')
parser.add_argument('--weight', type=float, default=2.879, help='')
parser.add_argument('--modify', type=int, default=1, help='')
parser.add_argument('--log', action='store_true', help='log scale')
parser.add_argument('--verbose', action='store_true', help='log scale')
parser.add_argument('--action', type=str, default='add_w', help='actions', choices=['add_w', 'reduce_w'])


def angle_diff(v1, v2):
    """ measure the angle difference of two eigenvectors """
    n = v1.shape[0]
    diff = []
    for i in range(n):
        angle = angle_between(v1[:, i], v2[:, i])
        angle = min(angle, np.pi - angle)
        angle = np.rad2deg(angle)
        diff.append(angle)
    return np.array(diff)


def energy(v1, L1):
    """ compute the energy
        v1: n * d
        L1 : n * n
        return tr(v.T * L * v)
    """

    L1 = tonp(L1)
    assert v1.shape[0] == L1.shape[0] == L1.shape[1]
    E = np.dot(np.dot(v1.T, L1), v1)
    E = np.diag(E)
    return E


def sample_index(n_edge):
    idx = np.random.choice(range(n_edge), 1)[0]
    if idx % 2 == 0:
        idx += 1
    return idx, idx - 1


if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed()
    n_node, n_edge = 300, 1000  # 6, 20
    nfeat_dim = 1
    efeat_dim = 1
    n_modify = args.modify

    G = random_pyG()
    g = random_pygeo_graph(n_node, nfeat_dim, n_edge, efeat_dim, device='cpu')
    edge_weight = random_edge_weight(n_edge)  # set the edge weight of original graph to be random
    g.edge_weight = edge_weight
    summary(g, 'origin_g')
    L1 = get_laplacian_mat(g.edge_index, g.edge_weight, n_node, normalization='sym')

    # change weight of one edge
    edge_weight2 = edge_weight
    if args.action == 'add_w':
        for n in range(n_modify):
            idx1, idx2 = sample_index(n_edge)  # 3, 2
            # print(f'Change weight ({idx1}) from {pf(edge_weight[idx1], 10)} to {args.weight}')
            edge_weight2[idx1] = args.weight
            edge_weight2[idx2] = args.weight
            g.edge_weight = edge_weight2
        L2 = get_laplacian_mat(g.edge_index, g.edge_weight, n_node, normalization='sym')

    # remove the same edge
    elif args.action == 'reduce_w':
        edge_weight3 = edge_weight
        for n in range(n_modify):
            idx1, idx2 = sample_index(n_edge)  # 3, 2
            edge_weight3[idx1] = 0.001
            edge_weight3[idx2] = 0.001
            g.edge_weight = edge_weight3
        L2 = get_laplacian_mat(g.edge_index, g.edge_weight, n_node, normalization='sym')
    else:
        NotImplementedError
        # contract edge # todo: implement this

    w1, v1 = eig(L1)
    w2, v2 = eig(L2)
    summary(v1[:, 0], 'v1[:, 0]')
    summary(v2[:, 0], 'v2[:, 0]')
    v_random1 = np.random.random(v2.shape)
    v_random1 = normalize(v_random1, axis=0)

    # banner('First eig vec')
    # plt.plot(v1[:, 0])
    # plt.plot(v2[:, 0])
    # sys.exit()

    if args.verbose:
        summary(L1, 'L1')
        summary(L2, 'L2')
        summary(w1, 'w1')
        summary(w2, 'w2')

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    # eig value
    # ax[0].plot(w1, marker='o', label='w1. First 20')
    # ax[0].plot(w2, marker='o', label='w2. First 20')
    ax[0].plot(w1[:20], marker='o', label='w1. First 20')
    ax[0].plot(w2[:20], marker='o', label='w2. First 20')
    ax[0].plot(w1[-20:], marker='o', label='w1. Last 20')
    ax[0].plot(w2[-20:], marker='o', label='w2. Last 20')
    ax[0].set_title(f'Comp EigVal. Weight={args.weight}. N_modify={n_modify}')
    # ax[0].set_yscale('log', basey=2)
    ax[0].legend()

    # eig vector
    eig_angle_diff = angle_diff(v1, v2)
    summary(eig_angle_diff, 'eig_angle_diff')
    ax[1].plot(eig_angle_diff[:30], marker='o', label='First 30')
    ax[1].plot(eig_angle_diff[-30:], marker='o', label='Last 30')
    ax[1].set_title('Eigvector angle difference')
    ax[1].legend()

    # eig cononical angle
    for i in range(10, n_node, 40):
        angles = scipy.linalg.subspace_angles(v1[:, :i], v2[:, :i])
        ax[2].plot(angles, label=f'{i}')
    ax[2].set_yscale('log', basey=2)
    ax[2].set_xscale('log', basex=2)
    ax[2].legend()
    ax[2].set_title('subspace_angles')

    # energy diff
    D1 = energy(v1, L1) - energy(v1, L2)
    D2 = energy(v2, L1) - energy(v2, L2)
    v_smooth = {'v_smooth0': v_random1}
    for i in range(10):
        tmp = np.dot(tonp(L1), v_smooth[f'v_smooth{i}'])
        tmp = normalize(tmp, axis=0)
        v_smooth[f'v_smooth{i + 1}'] = tmp

    D3 = energy(v_random1, L1) - energy(v_random1, L2)
    D4 = energy(v_smooth[f'v_smooth{0}'], L1) - energy(v_smooth[f'v_smooth{0}'], L2)
    D5 = energy(v_smooth[f'v_smooth{1}'], L1) - energy(v_smooth[f'v_smooth{1}'], L2)
    _D5 = energy(v_smooth[f'v_smooth{3}'], L1) - energy(v_smooth[f'v_smooth{3}'], L2)
    D6 = energy(v_smooth[f'v_smooth{9}'], L1) - energy(v_smooth[f'v_smooth{9}'], L2)

    # ax[3].plot(D1, label='v1')
    # ax[3].plot(D2, label='v2')
    # ax[3].plot(D3, label='random1', marker=',', alpha=.3)
    ax[3].plot(D4, label='smooth0', marker=',', alpha=1)
    ax[3].plot(D5, label='smooth1', marker=',', alpha=1)
    ax[3].plot(_D5, label='smooth3', marker=',', alpha=1)
    ax[3].plot(D6, label='smooth9', marker=',', alpha=1)

    ax[3].set_title('Energy Diff')
    ax[3].legend()

    # loc diff
    # loc_diff1 = v1[:, idx2] - v1[:, idx1]
    # loc_diff2 = v2[:, idx2] - v2[:, idx1]
    # _idx1, _idx2 = 101, 100
    # random_loc_diff1 = v1[:, _idx2] - v1[:, _idx1]
    # random_loc_diff2 = v2[:, _idx2] - v2[:, _idx1]
    #
    # ax[4].plot(loc_diff1, label='key loc v1')
    # ax[4].plot(loc_diff2, label='key loc v2')
    # ax[4].plot(random_loc_diff1, label='random loc v1')
    # ax[4].plot(random_loc_diff2, label='random loc v2')
    # ax[4].set_title('Key Loc diff')
    # ax[4].legend()

    plt.show()
