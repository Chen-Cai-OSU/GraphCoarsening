# Created at 2020-06-02
# Summary:

from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_lib as graph_lib
import graph_coarsening.graph_utils as graph_utils

import numpy as np
import scipy as sp
from scipy import io
from scipy.linalg import circulant
import time
import os

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import pygsp as gsp
gsp.plotting.BACKEND = 'matplotlib'

# Parameters
graphs  = ['yeast','minnesota', 'bunny', 'airfoil']
methods = ['heavy_edge', 'variation_edges', 'variation_neighborhoods', 'algebraic_JC', 'affinity_GS', 'kron']
K_all   = np.array([10,40], dtype=np.int32)
r_all   = np.linspace(0.1, 0.9, 17, dtype=np.float32)

print('k: ', K_all, '\nr: ', r_all)

rerun_all = False
rewrite_results = False
if rerun_all:

    algorithm = 'greedy'
    max_levels = 20
    n_methods = len(methods)
    n_graphs = len(graphs)

    for graphIdx, graph in enumerate(graphs):

        N = 4000  # this is only an upper bound (the actual size depends on the graph)
        G = graph_lib.real(N, graph)
        N = G.N
        if N < 100: continue

        # precompute spectrum needed for metrics
        if K_all[-1] > N / 2:
            [Uk, lk] = eig(G.L)
        else:
            offset = 2 * max(G.dw)
            T = offset * sp.sparse.eye(G.N, format='csc') - G.L
            lk, Uk = sp.sparse.linalg.eigsh(T, k=K_all[-1], which='LM', tol=1e-6)
            lk = (offset - lk)[::-1]
            Uk = Uk[:, ::-1]

        subspace = np.zeros((n_methods, len(K_all), len(r_all)))
        failed = np.zeros((n_methods, len(K_all), len(r_all)))
        ratio = np.zeros((n_methods, len(K_all), len(r_all)))

        for KIdx, K in enumerate(K_all):

            print('{} {}| K:{:2.0f}'.format(graph, N, K))

            for rIdx, r in enumerate(r_all):

                n_target = int(np.floor(N * (1 - r)))
                if K > n_target:
                    print('Warning: K={}>n_target={}. skipping'.format(K, n_target))
                    continue  # K = n_target

                for methodIdx, method in enumerate(methods):

                    # algorithm is not deterministic: run a few times
                    if method == 'kron':
                        if KIdx == 0:
                            n_iterations = 2
                            n_failed = 0
                            r_min = 1.0
                            for iteration in range(n_iterations):

                                Gc, iG = kron_coarsening(G, r=r, m=None)
                                metrics = kron_quality(iG, Gc, kmax=K_all[-1], Uk=Uk[:, :K_all[-1]], lk=lk[:K_all[-1]])

                                if metrics['failed']:
                                    n_failed += 1
                                else:
                                    r_min = min(r_min, metrics['r'])
                                    for iKIdx, iK in enumerate(K_all):
                                        subspace[methodIdx, iKIdx, rIdx] += metrics['error_subspace'][iK - 1]

                            subspace[methodIdx, :, rIdx] /= (n_iterations - n_failed)
                            failed[methodIdx, :, rIdx] = 1 if (r_min < r - 0.05) else 0
                            ratio[methodIdx, :, rIdx] = r_min

                            if np.abs(r_min - r) > 0.02: print(
                                'Warning: ratio={} instead of {} for {}'.format(r_min, r, method))

                    else:
                        C, Gc, Call, Gall = coarsen(G, K=K, r=r, max_levels=max_levels, method=method,
                                                    algorithm=algorithm, Uk=Uk[:, :K], lk=lk[:K])
                        metrics = coarsening_quality(G, C, kmax=K, Uk=Uk[:, :K], lk=lk[:K])

                        subspace[methodIdx, KIdx, rIdx] = metrics['error_subspace'][-1]
                        failed[methodIdx, KIdx, rIdx] = 1 if (metrics['r'] < r - 0.05) else 0
                        ratio[methodIdx, KIdx, rIdx] = metrics['r']

                        if np.abs(metrics['r'] - r) > 0.02:
                            print('Warning: ratio={} instead of {} for {}'.format(metrics['r'], r, method))

        if rewrite_results:
            filepath = os.path.join('..', 'results', 'experiment_approximation_' + graph + '.npz')
            print('.. saving to "' + filepath + '"')
            np.savez(filepath, methods=methods, K_all=K_all, r_all=r_all, subspace=subspace, failed=failed)

print('done!')
