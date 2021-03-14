# Created at 2020-06-04
# Summary:
# from sparsenet.util.sys_util import status
import os
n=30
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['OMP_NUM_THREADS'] = str(n)
os.environ['OPENBLAS_NUM_THREADS'] = str(n)
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
os.environ['NUMEXPR_NUM_THREADS'] = str(n)
import torch
torch.set_num_threads(n) # always import this first
print(f'Thread ({__file__}): {n}')

import networkx as nx

from sparsenet.util.util import summary, pf, timefunc, banner

from scipy.sparse.linalg import eigs, eigsh
from time import time
import functools
import numpy as np
tf = functools.partial(timefunc, threshold = 0.1)

# # @functools.lru_cache(maxsize=None)

@tf
def test_speed(n = 1000, k = 50, method = 'eigsh', which='LM'):
    t0 = time()
    g = nx.random_geometric_graph(n, 5/np.sqrt(n), seed=42)
    print(nx.info(g))
    m = nx.laplacian_matrix(g).asfptype()
    # print(type(m))

    t0 = time()
    if method == 'eigsh':
        vals, vecs = eigsh(m, k=k, which=which, tol=1e-3)
    elif method == 'eigsh_sigma':
        vals, vecs = eigsh(m, k=k, which=which, sigma=0)
    elif method == 'eigs':
        vals, vecs = eigs(m, k=k, which=which, tol=1e-3)
    else:
        raise NotImplementedError
    vals, vecs = vals.real, vecs.real
    t1 = time()
    print(f'method {method} for n {n}: {pf(t1-t0, 2)}')
    summary(vecs, 'vecs inside')
    summary(vals, 'vals inside')
    print(f'{method}({which}): {int(t1-t0)}')
    return vals, vecs

if __name__ == '__main__':
    from joblib import Parallel, delayed

    kwargs = {'method': 'eigsh', 'which': 'SM', 'k': 200}
    for n in \
            [1000, ]: # 232965
        # range(int(1e4), int(1e7), int(5e4)):
        vals, vecs = test_speed(n, method='eigsh', which='LM', k=200)
        vals_sm, vecs = test_speed(n, method='eigsh', which='SM', k=200)
        # vals_sm_sigma, vecs = test_speed(n, method='eigsh_sigma', which='SM', k=100)
        # summary(np.abs(vals_sm_sigma - vals_sm), 'vecs')
        # summary(vals, 'vals')
        banner(f'Finish n={n}')
        print()

