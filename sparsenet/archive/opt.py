# Created at 2020-08-24
# Summary: min ||L_w - U diag(lambda) U^T ||
#           s.t. UU^T = I, w > 0
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from graph_coarsening import coarsen
from scipy.stats import ortho_group
from signor.utils.random_ import fix_seed
from tqdm.notebook import tqdm
import cvxpy as cp
from sparsenet.util.data import loukas_data
from sparsenet.util.util import pyg2gsp, timefunc, pf, summary, make_dir, sig_dir


def ini(n=10, ubd=10):
    lbd = [0] + np.random.uniform(0, ubd, n - 1).tolist()
    lbd.sort()
    U = ortho_group.rvs(n)
    return np.array(lbd), U


def L2B(L, m):
    # convert np.array to indicence matrix B of size (m, n)
    n = L.shape[0]
    B = np.zeros((m, n))
    indices = np.nonzero(L)
    i = 0
    w = []
    for k in range(len(indices[0])):
        r, c = indices[0][k], indices[1][k]
        if r < c:
            w.append(-L[r, c])
            B[i][r] = 1
            B[i][c] = -1
            i += 1
    w = np.array(w)
    assert np.max(np.abs(B.T @ np.diag(w) @ B - L)) < 1e-5
    return B


@timefunc
def iter(U, lbd=None, verbose=False, mask=None, clip=False):
    L = np.dot(np.dot(U, np.diag(lbd)), U.conj().T)  # not necessary a Laplacian

    if mask is not None:
        assert mask.shape == L.shape
        L = np.multiply(L, mask)
    assert np.max(np.abs(L - L.T)) < 1e-3, f'L not symmetric: {np.max(np.abs(L - L.T))}'

    if clip:
        np.fill_diagonal(L, 0)
        L = np.where(L > 0, -1, L)
        diag = -np.sum(L, axis=0)
        np.fill_diagonal(L, diag)
    else:
        L = -np.abs(L)
        np.fill_diagonal(L, 0)
        diag = -np.sum(L, axis=0)
        np.fill_diagonal(L, diag)

    error = np.linalg.norm(L - U @ np.diag(lbd) @ U.T, 'fro')
    print(f'residual norm (old-iter) at {i} is {error}.')

    # update U
    eigv, U = np.linalg.eigh(L)
    if verbose:
        print(eigv)
    return U, eigv





@timefunc
def new_iter(U, B, i = 0, lbd=None, verbose=False, mask=None, ):
    # L = np.dot(np.dot(U, np.diag(lbd)), U.conj().T)  # not necessary a Laplacian
    m = B.shape[0]
    W = cp.Variable(shape=(m))
    constraint = [W >= 0]

    obj = cp.Minimize(cp.norm(B.T @ cp.diag(W) @ B - U @ cp.diag(lbd) @ U.T, 'fro'))
    prob = cp.Problem(obj, constraint)
    prob.solve(solver=cp.SCS, max_iters=10000, warm_start=True)
    if prob.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    print(f'residual norm at {i} is {pf(prob.value, 3)}.')
    W = W.value
    L = B.T @ np.diag(W) @ B

    # update U
    eigv, U = np.linalg.eigh(L)
    if verbose:
        print(eigv)
    return U, eigv

@timefunc
def update_gamma(B, U, Gamma2, W=None, lbd = None, beta=10, reg = True, max_iter = 10000):
    # B: m * n
    m, n = B.shape
    update_w = True
    if W is None:
        update_w = False
        W = np.ones(m)

    Gamma1 = cp.Variable(shape=(n))
    Gamma2 = np.random.random(n) if Gamma2 is None else Gamma2

    # update Gamma1
    constraint = [Gamma1 >= 1e-3]
    obj = cp.Minimize(cp.norm(cp.diag(Gamma1) @ B.T @ cp.diag(W) @ B @ cp.diag(Gamma2) - U @ cp.diag(lbd) @ U.T , 'fro') + beta * cp.norm(cp.diag(Gamma1 - Gamma2), 'fro'))
    prob = cp.Problem(obj, constraint)
    prob.solve(solver=cp.SCS, max_iters = max_iter)
    summary(Gamma1.value, 'Gamma1', highlight=True)
    Gamma1_np = Gamma1.value

    # update Gamma2
    if reg:
        Gamma1 = Gamma1_np
        Gamma2 = cp.Variable(shape=(n))
        obj = cp.Minimize(cp.norm(cp.diag(Gamma1) @ B.T @ cp.diag(W) @ B @ cp.diag(Gamma2) - U @ cp.diag(lbd) @ U.T , 'fro') + beta * cp.norm(cp.diag(Gamma1 - Gamma2), 'fro'))
        constraint = [Gamma2 >=1e-3]
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, max_iters = max_iter)
        Gamma2_np = Gamma2.value
    else:
        Gamma2_np = Gamma1_np
    summary(Gamma2_np, 'Gamma2', highlight=True)

    if update_w:
        W = cp.Variable(shape=(m))
        constraint = [W >= 1e-10]
        obj = cp.Minimize(cp.norm(cp.diag(Gamma1_np) @ B.T @ cp.diag(W) @ B @ cp.diag(Gamma2_np) - U @ cp.diag(lbd) @ U.T, 'fro'))
        prob = cp.Problem(obj, constraint)
        prob.solve(solver=cp.SCS, max_iters=max_iter)
        summary(W.value, 'W', highlight=True)
        W = W.value

    # update U
    L = np.diag(Gamma1_np) @ B.T @ np.diag(W) @ B @ np.diag(Gamma2_np)
    eigv, U = np.linalg.eigh(L)
    summary(U, 'U', highlight=True)
    print()

    return Gamma1_np, Gamma2_np, U, eigv



import argparse

parser = argparse.ArgumentParser(description='laplacian opt')
parser.add_argument('--n', type=int, default=200, help='number of nodes')
parser.add_argument('--k', type=int, default=-1, help='first k eigenvals')
parser.add_argument('--n_iter', type=int, default=30, help='number of total iterations')
parser.add_argument('--tor', type=float, default=1e-3, help='tor')
parser.add_argument('--no_plot', action='store_true')
parser.add_argument('--no_mask', action='store_true')
parser.add_argument('-v', '--vertex', action='store_true', help='vertex')
parser.add_argument('--cvx', action='store_true', help='use cvx update')
parser.add_argument('--clip', action='store_true')
parser.add_argument('--sc', action='store_true', help='Sanity check. ')
parser.add_argument('--data', type=str, default='yeast')
parser.add_argument('--max_iter', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)

if __name__ == '__main__':
    args = parser.parse_args()
    fix_seed(seed=args.seed)
    N = args.n
    tor = args.tor
    n_total_iter = args.n_iter

    G = loukas_data(name=args.data)[0]
    # kwargs = {'dataset': 'PubMed', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1,
    #           'sample': 'rw', 'n_vec': 5, 'w_len': 5000, 'include_self': True}
    # G = EgoGraphs(**kwargs)[0]
    # G = random_pygeo_graph(100, 1, 1000, 1)
    # n_node = 150
    # G = nx.random_geometric_graph(n_node, 0.2, seed=42)
    # n_edge = G.number_of_edges() * 2
    # G = from_networkx(G)
    # G.edge_weight = torch.ones(n_edge)
    # summary(G, 'G')

    # edge_index = barabasi_albert_graph(100, 5)
    # summary(edge_index, 'edge_index')
    # n_edge = edge_index.size()[1]
    # G = Data(x=torch.rand(n_node, 1),
    #          edge_index=edge_index,
    #          edge_attr=torch.rand(n_edge, 1).type(torch.LongTensor),
    #          edge_weight=torch.ones(n_edge))
    #
    # summary(G, 'G')
    # G = planetoid()[0]
    G = pyg2gsp(G)
    # fig, ax = plt.subplots()
    # plot_gsp(ax, gspG=G, title='')
    # plt.show()
    # exit()

    kwargs = {'K': int(40), 'r': 0.5, 'method': 'variation_edges', 'max_levels': 20}

    C, Gc, Call, Gall = coarsen(G, **kwargs)
    n = Gc.N

    lap = np.array(Gc.L.todense())
    B = L2B(lap, Gc.Ne)

    if args.no_mask: lap = np.random.random((n, n))
    lbd, U = G.e[:n], ortho_group.rvs(n)
    Gamma1, Gamma2 = np.random.random(n) + 0.5, np.random.random(n) + 0.5
    # Gamma1, Gamma2 = np.ones(n), np.ones(n)

    # planted solution
    if args.sc:
        if args.vertex:
            true_Gamma = np.random.random(n) + 0.5
            _L = np.diag(true_Gamma) @ B.T  @ B @ np.diag(true_Gamma)
            lbd, _ = np.linalg.eigh(_L)
        else:
            true_W = np.random.random(Gc.Ne) + 0.5
            _L =  B.T @ np.diag(true_W) @ B
            lbd, _ = np.linalg.eigh(_L)

    summary(abs(lbd - Gc.e), 'ini', highlight=True)

    eigvals = [lbd]
    diffs = []
    for i in tqdm(range(n_total_iter)):
        if args.cvx:
            if args.vertex:
                W = np.random.rand(B.shape[0]) + 0.5
                try:
                    Gamma1, Gamma2, U, eigv = update_gamma(B, U, Gamma2, lbd=lbd, W=W,
                                                           max_iter=args.max_iter, reg=True, beta=min(i, 20))
                except cp.error.SolverError:
                    print('Encounter SolverError. Break.')
                    break
            else:
                U, eigv = new_iter(U, B, i=i, lbd=lbd)

        else:
            U, eigv = iter(U, lbd=lbd, mask=np.sign(abs(lap)), clip=args.clip)

        eigvals.append(eigv)
        diff = max(abs(eigvals[-1] - eigvals[0]))
        print(f'iter {i} diff: {pf(diff, 3)}\n')
        diffs.append(diff)

        if i > 0 and diff < tor:
            print(f'break at iter {i}')
            summary(diff, 'diff', highlight=True)
            break

    k = n if args.k == -1 else args.k
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
    print(ax)
    ax.plot(G.e[:k], label=f'G.e')
    ax.plot(Gc.e[:k], label=f'Gc.e')
    ax.plot(eigvals[-1][:k], label='After Opt')
    ax.set_title(args.data)
    ax.legend()

    f = os.path.join(sig_dir(), 'sparsenet', 'paper', 'tex', 'Figs', 'opt', '')
    make_dir(f)
    name = f'{args.data}-vertex-{args.vertex}-sc-{args.sc}-1.pdf'
    print(name)
    f = os.path.join(f, name)
    plt.savefig(f, dpi=200, bbox_inches='tight')

    sys.exit()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
    print(ax)
    ax[0].plot(G.e[:k], label=f'G.e')
    ax[0].plot(Gc.e[:k], label=f'Gc.e')
    # ax[0].plot(eigvals[5][:k], label='after opt-5')
    ax[0].plot(eigvals[-1][:k], label='After Opt')
    # ax[0].plot(lbd[:k], label='lambda')
    ax[0].set_title(args.data)
    ax[0].legend()

    if not args.no_plot:
        diffs = [np.log10(diff) for diff in diffs[1:]]
        ax[1].plot(diffs)
        ax[1].set_title('log 10 error over iterations')

    f = os.path.join(sig_dir(), 'sparsenet', 'paper', 'tex', 'Figs', 'opt', '')
    make_dir(f)
    name = f'{args.data}-vertex-{args.vertex}-sc-{args.sc}.pdf'
    print(name)
    f = os.path.join(f, name)
    plt.savefig(f, dpi=200, bbox_inches='tight')
