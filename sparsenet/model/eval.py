# Created at 2020-05-27
# Summary: add train and eval for model training. Extracted from example.py. Contains import abstraction.

import logging
from copy import deepcopy
from time import time

import torch
from torch.autograd import Variable as V

from sparsenet.model.loss import get_laplacian_mat
from sparsenet.util.data import set_loader
from sparsenet.util.loss_util import loss_manager
from sparsenet.util.torch_util import sparse_mm
from sparsenet.util.train_util import check_laplacian
from sparsenet.util.util import summary, pf, red, banner, fix_seed, timefunc as tf

fix_seed()


def apply_gnn(batch, model, dev, verbose=False, ini=False):
    """
    :param batch:
    :param model:
    :return:
    """
    indices_batch, graph_batch = batch
    if verbose:
        summary(graph_batch, 'graph_batch', highlight=True)

    _bs = len(indices_batch[0])
    indices_batch = torch.stack(indices_batch, dim=0).t().contiguous().view((2 * _bs))  # https://bit.ly/2ARazSd
    indices_batch, graph_batch = indices_batch.to(dev), graph_batch.to(dev)
    pred = model(graph_batch, ini=ini)  # tensor of size (bs, out_dim)
    return pred, indices_batch


@tf
def correction(LM, L2, args):
    if args.strategy == 'loukas' and args.correction:
        # remark: not ideal but a reasonable workaround. Memory intensive.
        # L2_correction = torch.sparse.mm(torch.sparse.mm(LM.invQ, L2.to_dense()).to_sparse(),
        #                                 LM.invQ.to_dense())
        # L2_correction = L2_correction.to_sparse() # remark: has small difference with current version
        L2_correction = sparse_mm(L2, LM.invQ)
    else:
        L2_correction = L2
    return L2_correction


class tester(object):
    def __init__(self, name='default', comment='evalulation', dev='cuda'):
        self.test_data = {}
        self.test_data_comb = {}
        self.original_graph = {}
        self.sparse_graph = {}
        self.dev = dev
        self.name = name
        self.comment = comment

    def set_test_data(self, args, data_loader, verbose=False):
        """ set the data for evalulation """
        self.rayleigh_flag = True if args.loss == 'rayleigh' else False

        if args.test_idx in self.test_data.keys():
            print(f'Test graph {args.test_idx} has been processed. Skip.')
            return

        g_T, _ = data_loader.load(args, mode='test')
        self.original_graph[args.test_idx] = g_T

        test_loader, sub_T, n_sml_T = set_loader(g_T, args.bs, shuffle=True, args=args)

        L1_T = sub_T.L(g_T, normalization=args.lap)
        T_L2_ = sub_T.baseline0(normalization=args.lap)

        L1_T_comb = sub_T.L(g_T, normalization=None) if args.dynamic else None
        L2_T_comb = sub_T.baseline0(normalization=None) if args.dynamic else None
        self.L1_T_comb = L1_T_comb
        self.test_data_comb[args.test_idx] = L2_T_comb

        self.T_L2_ = T_L2_
        T_L2_trival = sub_T.trivial_L(sub_T.g_sml)

        LM_T = loss_manager(signal='bottomk', device=self.dev)

        try:
            LM_T.set_precomute_x(g_T, args, k=args.n_bottomk - args.offset)
        except:
            print(red('Recompute eigenvector'))
            LM_T.set_x(L1_T, g_T.num_nodes, args.n_bottomk - args.offset, which='SM')

        LM_T.set_C(sub_T.C)
        LM_T.set_s(g_T.num_nodes, k=args.n_bottomk)

        bl_loss_T, bl_ratio_loss_T = LM_T.quaratic_loss(L1_T, T_L2_, sub_T.assignment, inv=args.inv,
                                                        rayleigh=self.rayleigh_flag, dynamic=args.dynamic,
                                                        comb=(L1_T_comb, L2_T_comb))

        trival_loss_T, _ = LM_T.quaratic_loss(L1_T, T_L2_trival, sub_T.assignment, inv=args.inv,
                                              rayleigh=self.rayleigh_flag, dynamic=args.dynamic,
                                              comb=(L1_T_comb, T_L2_trival))  # todo: look at trivial loss
        T_L2_correction = correction(LM_T, T_L2_, args)

        bl_eigen_loss_T = LM_T.eigen_loss(L1_T, T_L2_correction, args.n_bottomk - args.offset, args=args,
                                          g1=g_T) if args.valeigen else torch.tensor(-1)  # todo: change back


        if verbose:
            summary(L1_T, 'L1_T')
            summary(T_L2_, 'T_L2_')
            print(f'Baseline 0 loss: {red(bl_loss_T)}')

        edge_weight_sml_T_buffer = deepcopy(sub_T.g_sml.edge_weight).to(self.dev)
        edge_index_sml_T = sub_T.g_sml.edge_index.to(self.dev)

        test_data = g_T, test_loader, edge_weight_sml_T_buffer, sub_T, L1_T, \
                    bl_loss_T, bl_ratio_loss_T, trival_loss_T, bl_eigen_loss_T, n_sml_T, edge_index_sml_T, LM_T
        self.test_data[args.test_idx] = test_data
        banner(f'Finish setting {args.dataset} graph {args.test_idx}', compact=True, ch='-')

    def delete_test_data(self, idx):
        del self.test_data[idx]

    @tf
    def eval(self, model, args, verbose=False):

        t0_ = time()
        model.eval()

        g_T, test_loader, edge_weight_sml_T_buffer, sub_T, L1_T, bl_loss_T, \
        bl_ratio_loss_T, trival_loss_T, bl_eigen_loss_T, n_sml_T, edge_index_sml_T, LM_T = \
            self.test_data[args.test_idx]

        L2_ini_comb = self.test_data_comb[
            args.test_idx]  # get_laplacian_mat(edge_index_sml_T, edge_weight_sml_T_buffer, n_sml_T, normalization=None) if args.dynamic else None

        L1_T_comb = sub_T.L(g_T, normalization=None) if args.dynamic else None
        # summary(L2_ini_comb, 'Test: L2_ini_comb', highlight=True)
        # summary(L1_T_comb, 'Test: L1_T_comb', highlight=True)
        # summary(edge_weight_sml_T_buffer, 'Test: edge_weight_sml_T_buffer', highlight=True)
        comb = (L1_T_comb, L2_ini_comb)

        for step_, batch_T in enumerate(test_loader):
            pred_T, indices_batch_T = apply_gnn(batch_T, model, self.dev, ini=args.ini)
            edge_weight_sml_T = V(edge_weight_sml_T_buffer)
            edge_weight_sml_T[indices_batch_T] = pred_T.view(-1).repeat_interleave(2)
            # L2_T = get_laplacian_mat(edge_index_sml_T, edge_weight_sml_T, n_sml_T, normalization=args.lap)

            if verbose:
                summary(pred_T, f'test: pred_T at step {step_}')
                summary(edge_weight_sml_T, f'test: edge_weight_sml_T at {step_}')
                summary(indices_batch_T, f'test: indices_batch_T at {step_}')
                print()

        L2_T = get_laplacian_mat(edge_index_sml_T, edge_weight_sml_T, n_sml_T, normalization=args.lap)
        L2_T_correction = correction(LM_T, L2_T, args)

        loss_T, ratio_loss_T = LM_T.quaratic_loss(L1_T, L2_T, sub_T.assignment, inv=args.inv,
                                                  rayleigh=self.rayleigh_flag,
                                                  verbose=True)  # dynamic=args.dynamic, comb=comb)

        # expansive, so only calculate when needed
        eigen_loss_T = LM_T.eigen_loss(L1_T, L2_T_correction, args.n_bottomk - args.offset, args=args,
                                       g1=g_T) if args.valeigen else torch.tensor(-1)  # todo: change back

        t1_ = time()
        msg = 'Generalize!' if loss_T < min(bl_loss_T, trival_loss_T) else ''
        n_sig = 3
        logging.info(' ' * 12 +
                     f'Graph-{args.dataset}: {args.test_idx}. '
                     f'{red("Test-Val")}({pf(t1_ - t0_, 1)}):  {pf(loss_T, n_sig)}({pf(ratio_loss_T, n_sig)}) / '
                     f'{pf(bl_loss_T, n_sig)}({pf(bl_ratio_loss_T, n_sig)}) / {pf(trival_loss_T)}. {red(msg)}. '
                     f'Eigenloss: {pf(eigen_loss_T, n_sig)}. '
                     f'Bl_Eigenloss: {pf(bl_eigen_loss_T, n_sig)}.')

        n_gen = 1 if msg == 'Generalize!' else 0
        impr_ratio = min(bl_loss_T, trival_loss_T) / loss_T
        eigen_ratio = (bl_eigen_loss_T - eigen_loss_T) / bl_eigen_loss_T  # bl_eigen_loss_T / eigen_loss_T
        return n_gen, impr_ratio.item(), eigen_ratio.item()


class trainer:

    def __init__(self, name='default', comment='test tensorboard', dev='cuda'):
        self.n_graph = 0  # number of graphs that has been processed
        self.train_data = {}
        self.train_data_comb = {}
        self.dev = dev
        self.name = name
        self.comment = comment
        self.original_graph = {}

    def set_train_data(self, args, data_loader):
        """ quite similar with set_test_data """
        self.rayleigh_flag = True if args.loss == 'rayleigh' else False

        if args.train_idx in self.train_data.keys():
            print(f'Train graph {args.train_idx} has been processed. Skip.')
            return

        g, _ = data_loader.load(args, mode='train')
        self.original_graph[args.train_idx] = g
        train_loader, sub, n_sml = set_loader(g, args.bs, shuffle=True, args=args)

        L1 = sub.L(g, normalization=args.lap)
        L1_comb = sub.L(g, normalization=None) if args.dynamic else None
        g_sml, assignment = sub.g_sml, sub.assignment
        edge_index_sml = g_sml.edge_index.to(self.dev)
        L2_ = sub.baseline0(normalization=args.lap)
        L2_comb = sub.baseline0(normalization=None) if args.dynamic else None
        self.train_data_comb[args.train_idx] = (L1_comb, L2_comb)

        self.L2_ = L2_
        L_trivial = sub.trivial_L(g_sml)  # todo: look at trivial loss tomorrow

        summary(L1, 'L1')
        summary(L2_, 'L2_baseline0')

        LM = loss_manager(signal='bottomk', device=self.dev)
        # test vector as slightly different when adding loukas_quality argument.
        # Not sure why but seems the change is very minor.
        try:
            LM.set_precomute_x(g, args, k=args.n_bottomk)
        except:
            LM.set_x(L1, g.num_nodes, args.n_bottomk, which='SM')
        LM.set_C(sub.C)
        LM.set_s(g.num_nodes, k=args.n_bottomk)


        bl_loss, bl_ratio = LM.quaratic_loss(L1, L2_, sub.assignment, inv=args.inv, rayleigh=self.rayleigh_flag,
                                             dynamic=args.dynamic, comb=(L1_comb, L2_comb))
        trivial_loss, trivial_ratio = LM.quaratic_loss(L1, L_trivial, sub.assignment, inv=args.inv,
                                                       rayleigh=self.rayleigh_flag, dynamic=args.dynamic,
                                                       comb=(L1_comb, L_trivial))
        T_L2_correction = correction(LM, L2_, args)
        skip_flag = True if g.num_nodes > 1e3 else False
        bl_eigen_loss = LM.eigen_loss(L1, T_L2_correction, args.n_bottomk, args=args, g1=g, skip=skip_flag)



        edge_weight_sml_buffer = deepcopy(sub.g_sml.edge_weight).to(self.dev)
        train_data = g, train_loader, edge_weight_sml_buffer, \
                     sub, L1, bl_loss, bl_ratio, \
                     trivial_loss, trivial_ratio, \
                     n_sml, edge_index_sml, LM, bl_eigen_loss

        assert args.train_idx not in self.train_data.keys(), \
            f'Overwrite self.train_data for key {args.train_idx}. Check carefully!'

        self.train_data[args.train_idx] = train_data
        return

    def delete_train_data(self, idx):
        del self.train_data[idx]

    @tf
    def train(self, model, optimizer, args, verbose=False):
        g, train_loader, edge_weight_sml_buffer, sub, L1, bl_loss, bl_ratio, trivial_loss, \
        trivial_ratio, n_sml, edge_index_sml, LM, bl_eigen_loss = self.train_data[args.train_idx]

        L2_ini = get_laplacian_mat(edge_index_sml, edge_weight_sml_buffer, n_sml, normalization=args.lap)
        L1_comb, L2_ini_comb = self.train_data_comb[args.train_idx]
        summary(L1_comb, 'Train: L1_comb', highlight=True)
        summary(L2_ini_comb, 'Train: L2_ini_comb', highlight=True)
        loss_ini, _ = LM.quaratic_loss(L1, L2_ini, sub.assignment, verbose=False, inv=args.inv, dynamic=args.dynamic,
                                       comb=(L1_comb, L2_ini_comb))

        logging.info(f'Initial quaratic loss: {red(pf(loss_ini, 3))}.')
        for n_iter in range(1, args.n_epoch + 1):
            t0 = time()

            for step, batch in enumerate(train_loader):
                model.train()
                pred, indices_batch = apply_gnn(batch, model, self.dev, ini=args.ini)
                edge_weight_sml = V(edge_weight_sml_buffer)
                edge_weight_sml[indices_batch] = pred.view(-1).repeat_interleave(2)

                L2 = get_laplacian_mat(edge_index_sml, edge_weight_sml, n_sml, normalization=args.lap)
                comb = (L1_comb, L2_ini_comb)

                loss, ratio = LM.quaratic_loss(L1, L2, sub.assignment, verbose=False, inv=args.inv,
                                               rayleigh=self.rayleigh_flag)

                optimizer.zero_grad()
                loss.backward(retain_graph=False)  # https://bit.ly/2LbZNaR
                optimizer.step()
                if args.lap_check and args.lap not in ['rw'] and n_iter < 2 and step % 3 == 0:
                    check_laplacian(L2, step, eps=1e-5)

            L2_correction = correction(LM, L2, args)
            skip_flag = True if g.num_nodes > 1e2 else False
            eigen_loss = LM.eigen_loss(L1, L2_correction, args.n_bottomk, args=args, g1=g, skip=skip_flag)

            space = '\n' if verbose else ''
            n_sig = 3
            logging.info(f'{args.dataset}-Idx {args.train_idx}-Epoch: {n_iter}. '
                         f'Train({pf(time() - t0)}): {pf(loss, n_sig)}({pf(ratio, n_sig)})'
                         f' / {pf(bl_loss, n_sig)}({pf(bl_ratio, n_sig)}) / {pf(trivial_loss, n_sig)}. '
                         f'Eigenloss: {pf(eigen_loss, n_sig)}. {space}'
                         f'Bl_Eigenloss: {pf(bl_eigen_loss, n_sig)}')

        banner(f'Finish training {args.dataset} {args.train_idx} for {args.n_epoch} epochs.')
        self.n_graph += 1
