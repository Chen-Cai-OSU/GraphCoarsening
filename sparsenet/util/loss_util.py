# Created at 2020-05-20
# Summary: Implement a class so that one can get all sort of random vecs.

from copy import deepcopy

import numpy as np
import scipy as sp
import torch
from torch.sparse import mm as smm

from deprecated import deprecated
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs, eigsh

from sparsenet.model.loss import get_sparse_projection_mat
from sparsenet.util.cut_util import pyG_conductance
from sparsenet.util.torch_util import sparse_mm2, sparse_matrix2sparse_tensor
from sparsenet.util.util import timefunc as tf, fix_seed, summary, random_laplacian, pf, tonp, red, dic2tsr

fix_seed()


class vec_generator(object):
    def __init__(self):
        pass

    def _normalize(self, X):
        """
        :param X: Input vec mat.
        :return: Normalized vec mat.
        """
        return X / ((X ** 2).sum(0, keepdim=True)).sqrt()

    def _sparse_tensor2_sparse_numpyarray(self, sparse_tensor):
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

    def _laplacian2adjacency(self, laplacian):
        """
        :param laplacian: Input laplacian mat.
        :return: return adajacency, basically remove diagonal elements, make non-diagonal elements positive.
        """
        values, indices = laplacian._values(), laplacian._indices()
        mask = [False if (u == v) else True for _, (u, v) in enumerate(indices.t().tolist())]
        new_values, new_indices = -values[mask], indices[:, mask]
        return torch.sparse.FloatTensor(new_indices, new_values, laplacian.size())

    def random_vec(self, N, num_vec, normalized=True, reproducible=False):
        """
        :param N: Dimension of the vec
        :param num_vec: Number of random vec
        :param normalized: If normalized the, L2 norm of return vectors will be 1.
        :param reproducible: if reproducible=True, then the random seeds are fixed.
        :return: A N * num_vec random vec tensor.
        """
        if reproducible:
            fix_seed()
        X = torch.rand(N, num_vec) - 0.5
        if normalized:
            X = self._normalize(X)
        return X

    @tf
    def bottomk_vec(self, laplacian, k, which='SM', val=False):
        """
        :param laplacian: The input laplacian matrix, should be a sparse tensor.
        :param k: The top K (smalleset) eigenvectors.
        :param which: LM, SM, LR, SR, LM, SM largest/smallest magnitude, LR/SR largest/smallest real value.
        more details see scipy.sparse.linalg.eigs
        :return: return top K eigenvec. in the format of a N * k tensor. All vectors are automatically normalized.
        """
        assert isinstance(laplacian, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)), \
            f'input laplacian must be sparse tensor. Got {type(laplacian)}'

        # we need to convert the sparse tensor to scipy sparse mat, so that we can apply
        # the functions scipy.sparse.linalg.eigs() which should be faster than other methods.
        scipy_lap = self._sparse_tensor2_sparse_numpyarray(laplacian)
        M, N = scipy_lap.shape
        assert (M == N and k < N - 1), f'Input laplacian must be a square matrix. ' \
            f'To use scipy method, {k} (#eigvecs) < {N - 1} (size of laplacian - 1).'

        try:
            vals, vecs = eigsh(scipy_lap, k=k, which=which, tol=1e-3)
            vecs = torch.FloatTensor(vecs.real)
        except sp.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            print(red('Eigsh failed. Try computing with eigs'))
            vals, vecs = eigs(scipy_lap, k=k, which=which, tol=0)
            vecs = torch.FloatTensor(vecs.real)
        except:
            exit(f'Convergence Error in bottomk_vec when computing {k} eigenvecotrs.')  # shape dataset has such problem

        vecs = self._normalize(vecs)  # no effect
        if val:
            return vals
        else:
            return vecs

    def random_projected_vec(self, laplacian, num_vec, power_method_iter=5, reproducible=False):
        """
        :param laplacian: The laplacian matrix, used to generate the adjacency mat for power method.
        :param num_vec: Number of starting random vectors.
        :param reproducible: fix random seed?
        :param power_method_iter: How many times we apply f(i+1) = Af(i)/|Af(i)|
        :return: Return a num_vec N*1 vectors, in the form of N * num_vec matrix. Each vector was applied power method
        by #power_method_iter times.
        """
        assert isinstance(laplacian,
                          (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)), 'Input laplacian must be' \
                                                                                      'sparse tensor.'
        size = laplacian.size()
        assert (size[0] == size[1]), 'Input laplacian must be a square mat.'
        vectors = self.random_vec(size[0], num_vec, normalized=True, reproducible=reproducible)
        print('Original vecs:', vectors)
        adjacency = self._laplacian2adjacency(laplacian)
        for i in range(power_method_iter):
            vectors = self._normalize(smm(adjacency, vectors))
        return vectors


class loss_manager(object):
    def __init__(self, signal='bottomk', device='cuda'):
        assert signal in ['bottomk', 'random', 'random_proj'], f'signal {signal} is not implemented!'
        self.gen = vec_generator()
        if signal == 'bottomk':
            method = 'bottomk_vec'
        elif signal == 'random':
            method = 'random_vec'
        elif signal == 'random_proj':
            method = 'random_projected_vec'
        else:
            NotImplementedError

        self.method = method
        self.dev = device
        self.L1 = None
        self.vals_L1 = None
        self.inv_asgmt = None
        self.Projection = None
        self.D1 = None

    @tf
    def set_C(self, C=None):
        # todo: add comment
        if C is not None:
            self.C = C  # csc_matrix of shape (n, N)
            self.pi = sparse_matrix2sparse_tensor(self.C.T.dot(self.C).tocoo(),
                                                  dev=self.dev)  # mainly used for rayleigh quotient
            tmp = self.C.dot(np.ones((self.C.shape[1], 1))).reshape(-1)
            assert np.min(tmp) > 0, f'min of tmp is {np.min(tmp)}'
            # self.Q = np.diag(tmp)
            # self.invQ = np.diag(1/tmp)

            n = self.C.shape[0]
            diag_indices = [list(range(n))] * 2
            i = torch.LongTensor(diag_indices)
            v = torch.FloatTensor(1.0 / tmp)
            self.invQ = torch.sparse.FloatTensor(i, v, torch.Size([n, n])).to(self.dev)

    def set_precomute_x(self, g, args, k=40, v=False):
        key = f'{args.lap}_vecs'
        self.x = g[key][:, :k].to(self.dev)
        if v:
            summary(self.x, red(f'precomputed test vector {key}'))

    def set_x(self, *args, **kwargs):
        print(red('Recompute eigenvector'))
        self.x = getattr(self.gen, self.method)(*args, **kwargs).to(self.dev)
        summary(self.x, 'test vector')

    @tf
    def set_s(self, n, k=40):
        """ set a random set of nodes for condunctance.
            Generate a list (len k) of random nodes in ORIGINAL graph as test subset
        """
        self.s_list = []
        self.s_list_tsr = []
        for _ in range(k):
            _size = np.random.choice(range(int(n / 4.0), int(n / 2.0)))
            s = np.random.choice(range(n), size=_size, replace=False).tolist()
            self.s_list.append(s)
            self.s_list_tsr.append(torch.tensor(s))

    ############ condunctance_loss related ############
    @tf
    def _build_inv_asgnment(self, assgnment):
        if self.inv_asgmt is None:
            self.inv_asgmt = {v: key for (key, value) in assgnment.items() for v in
                              value}  # key is the nodes in large graph.
            self.inv_asgmt_tsr = dic2tsr(self.inv_asgmt, dev=self.dev)

    @tf
    def get_s_prime(self, s):
        """ assume self.inv_asgmt is built.
        From s generate s_prime. used for condunctance_loss.
        """
        s = s.to(self.dev)
        if isinstance(s, torch.Tensor):
            s_prime = torch.index_select(self.inv_asgmt_tsr, 0, s)
            s_prime = torch.unique(s_prime)  # remove duplicates
        elif isinstance(s, list):
            s_prime = [self.inv_asgmt[s_] for s_ in s]
            s_prime = list(set(s_prime))
        else:
            summary(s, 's')
            raise NotImplementedError(f's is {type(s)}')

        return s_prime

    @tf
    def condunctance_loss(self, g1, g2, assgnment, verbose=False):
        """
        todo: slow for shape dataset: 1.2s each batch
        :param g1: edge_index1, edge_attr1 for original graph
        :param g2: edge_index2, edge_attr2 for smaller graph
        :param assgnment: dict
        :return:
        """

        edge_index1, edge_attr1 = g1
        edge_index2, edge_attr2 = g2
        self._build_inv_asgnment(assgnment)
        loss = 0
        for i, s in enumerate(self.s_list_tsr):

            cond1 = pyG_conductance(edge_index1, edge_attr1, s.tolist(), t=None, dev=self.dev)
            s_prime = self.get_s_prime(s)
            cond2 = pyG_conductance(edge_index2, edge_attr2, s_prime.tolist(), t=None, dev=self.dev)
            loss += torch.abs(cond1 - cond2)

            if verbose:
                print(f's: {len(s)}. s_prime: {len(s_prime)}')
                summary(np.array(s), 's')
                summary(edge_index1, 'edge_index1')
                summary(edge_attr1, 'edge_attr1')
                print(red(f'cond1-{i}: {pf(cond1, 2)}. cond2-{i}: {pf(cond2, 2)}'))

        return loss / len(self.s_list)

    ############ quadratic_loss related ###############
    @deprecated(reason="to be refactord")
    def _set_d(self, L, power=0.5):
        """ from sparse tensor L to degree matrix """
        # todo: speed up. 3
        dev = L.device
        n = L.shape[0]
        idx = torch.LongTensor([[i, i] for i in range(n)]).T.to(dev)
        diag = torch.diag(L.to_dense())
        diag = diag ** (power)
        deg = torch.sparse.FloatTensor(idx, diag, torch.Size([n, n]))
        return deg

    def quaratic_loss(self, L1, L2, assignment, verbose=False, inv=False,
                      rayleigh=False, dynamic=False,
                      comb=(None, None)):
        """
        modfied from random_vec_loss.
        :param L1: Laplace of original graph
        :param L2: Laplace of smaller graph
        :param Projection
        :param inv: inverse Laplacian. (Not Really Working)
        :param: rayleigh: normalized x
        :param: dynamic: dynamic projection. Update projection in runtime.
        :param: comb: combinatorial L1, L2. Only used for normalized Laplacian.
        :return loss, ratio
        """
        L1, L2 = L1.to(self.dev), L2.to(self.dev)
        if self.Projection is None:
            self.Projection = get_sparse_projection_mat(L1.shape[0], L2.shape[0], assignment).to(self.dev)  # sparse tensor
            Projection = self.Projection
        else:
            Projection = self.Projection

        if dynamic:
            L1_comb, L2_comb = comb
            assert L1_comb is not None
            if self.D1 is None:
                self.D1 = self._set_d(L1_comb, power=-0.5)
                D1 = self.D1
            else:
                D1 = self.D1
            D2 = self._set_d(L2_comb, power=0.5)
            Projection = sparse_mm2(Projection, D1, D2)

        X_prime = smm(Projection, self.x)

        if inv:
            raise NotImplementedError
        else:
            quadL1 = torch.mm(self.x.t(), smm(L1, self.x))
            qualL2 = torch.mm(X_prime.t(), smm(L2, X_prime))

        diff = torch.abs(torch.diag(quadL1 - qualL2))
        if rayleigh:
            assert self.pi is not None
            denominator = torch.diag(torch.mm(self.x.t(), smm(self.pi, self.x)))  # (n_bottomk,)
            diff = diff / denominator

        loss = torch.mean(diff)
        ratio = torch.sum(torch.diag(qualL2)) / torch.sum(torch.diag(quadL1))
        ratio = torch.abs(torch.log(ratio))
        if verbose:
            bad_indices = tonp((diff / loss > 1).nonzero())
            print(bad_indices.reshape(-1))
        return loss, ratio

    @tf
    def eigen_loss(self, L1, L2, k, args=None, g1=None, skip=False):
        """ compare the first k eigen difference;  L1 is larger than L2
        :param args
        :param g1: used for retrive precomputed spectrum
        """
        if skip: return -1

        # get eigenvalues of L1
        if self.vals_L1 is None:
            # compute eigenvals only once
            self.L1 = L1  # doesn't seem to be useful any more
            key = str(args.lap) + '_vals'
            vals_L1 = g1[key][:k].numpy()
            vals_L1 = deepcopy(vals_L1)  # if not deepcopy, g1 None_vals[0] will get modified
            self.vals_L1 = vals_L1
        else:
            vals_L1 = self.vals_L1

        # get eigenvalues of L2
        if args.cacheeig:
            raise NotImplementedError
        else:
            vals_L2 = self.gen.bottomk_vec(L2, k, which='SM', val=True).real

        # compute the eigenvalues error
        vals_L1 = vals_L1[:len(vals_L2)]  # in case vals_L1 and vals_L2 are of different length
        bad_indices = np.nonzero(vals_L1 < 1e-5)
        if len(bad_indices) > 1:
            print(red(f'There are {len(bad_indices)} nearly zero eigenvalues.'))
        err = np.abs(vals_L1 - vals_L2) / (vals_L1 + 1e-15)
        err[0] = 0
        err[bad_indices] = 0
        return np.mean(err)


if __name__ == '__main__':

    LM = loss_manager(signal='bottomk')

    # exit()
    gen = vec_generator()
    # print(gen.random_vec(N=3, num_vec=2,reproducible=True))
    # print(gen.random_vec(N=3, num_vec=2,reproducible=True))

    i = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    v = torch.FloatTensor([0, 1, 2, 3])
    sparse_mat = torch.sparse.FloatTensor(i, v, torch.Size((4, 4)))
    LM = loss_manager()
    eigloss = LM.eigen_loss(sparse_mat, sparse_mat, 2, g1=None)
    print(eigloss)
    bottomk_vec = gen.bottomk_vec(laplacian=sparse_mat, k=2)

    for i in range(2):
        summary(bottomk_vec[:, i], f'bottomk_vec[:, {i}]')
    exit()

    # i = torch.LongTensor([[0, 1, 2, 3, 1, 2, 0, 3, 1, 3],[0, 1, 2, 3, 2, 1, 3, 0, 3, 1]])
    # v = torch.FloatTensor([1, 2, 1, 2, -1, -1, -1, -1, -1, -1])
    projected_vec = gen.random_projected_vec(laplacian, 5, 2, reproducible=True)
    summary(projected_vec, 'projected_vec')

    projected_vec = gen.random_projected_vec(laplacian, 5, 2, reproducible=True)
    print('Projected vecs', projected_vec)
    exit()

    n = 100
    i, v = random_laplacian(n)
    summary(i, 'i')
    summary(v, 'v')
    laplacian = torch.sparse.FloatTensor(i, v, torch.Size((n, n)))

    for _ in range(5):
        eigenvec = gen.bottomk_vec(laplacian, 2)
        summary(eigenvec, 'eigenvec')
    exit()
