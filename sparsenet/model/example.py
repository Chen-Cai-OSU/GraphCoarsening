# Created at 2020-04-17
# Summary: a simple example to illustrate data pipeline

import os

from sparsenet.util.model_util import ModelEvaluator

n = 2
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['OMP_NUM_THREADS'] = str(n)
os.environ['OPENBLAS_NUM_THREADS'] = str(n)
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
os.environ['NUMEXPR_NUM_THREADS'] = str(n)
import torch

torch.set_num_threads(n)  # always import this first

from time import time
import argparse
import logging
import sys

import numpy as np

from sparsenet.model.eval import tester, trainer  # train, set_train_data
from sparsenet.model.model import GNN_graphpred
from sparsenet.util.data import data_loader
from sparsenet.util.name_util import set_model_dir
from sparsenet.util.train_util import monitor
from sparsenet.util.util import fix_seed, banner, red, pf, slack

parser = argparse.ArgumentParser(description='Graph edge sparsification')

# model
parser.add_argument('--n_layer', type=int, default=3, help='number of layer')
parser.add_argument('--emb_dim', type=int, default=50, help='embedding dimension')
parser.add_argument('--ratio', type=float, default=0.5, help='reduction ratio')
parser.add_argument('--n_vec', type=int, default=100, help='number of random vector')
parser.add_argument('--force_pos', action='store_true', help='Force the output of GNN to be positive')
parser.add_argument('--dataset', type=str, default='ws', help='the name of dataset')
parser.add_argument('--w_len', type=int, default=5000, help='walk length')

# optim
parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--n_epoch', type=int, default=50, help='')
parser.add_argument('--bs', type=int, default=600, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_bottomk', type=int, default=40, help='Number of Bottom K eigenvector')

# debug
parser.add_argument('--lap', type=str, default='none', help='Laplacian type',
                    choices=[None, 'sym', 'rw', 'none', 'None'])
parser.add_argument('--debug', action='store_true', help='debug. Smaller graph')
parser.add_argument('--tbx', action='store_true', help='write to tensorboardx.')
parser.add_argument('--inv', action='store_true', help='use inverse Laplacian for loss')
parser.add_argument('--viz', action='store_true', help='visualization of weights of sparsified graph. Save to dir.')
parser.add_argument('--show', action='store_true', help='Show the figure.')
parser.add_argument('--ini', action='store_true', help='initialilze the output of gnn to be near the weights of g_sml')
parser.add_argument('--testonly', action='store_true', help='Skip the training. Only test.')
parser.add_argument('--valeigen', action='store_true', help='Use eigen_ratio as metric to select model')
parser.add_argument('--cacheeig', action='store_true', help='save and load cached eigenvector of coarse graph')
parser.add_argument('--mlp', action='store_true', help='use a mlp baseline')

# parser.add_argument('--verbose', action='store_true', help='control the info level for real graph')
parser.add_argument('--train_idx', type=int, default=0, help='train index of the shape data. Do not change.')
parser.add_argument('--test_idx', type=int, default=0, help='test index of the shape data. Do not change.')
parser.add_argument('--cur_idx', type=int, default=-1, help='Current index. used for save coarsening graphs')
parser.add_argument('--lap_check', action='store_true', help='check the laplacian is normal during training')
parser.add_argument('--n_cycle', type=int, default=1, help='number of cycles')
parser.add_argument('--trial', type=int, default=0, help='trial. Act like random seed')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--loss', type=str, default='quadratic', help='quadratic loss',
                    choices=['quadratic', 'conductance', 'rayleigh'])
parser.add_argument('--offset', type=int, default='0', help='number of offset eigenvector')

parser.add_argument('--correction', action='store_true', help='Apply Laplacian correction')
parser.add_argument('--dynamic', action='store_true', help='Dynamic projection')
parser.add_argument('--loukas_quality', action='store_true', help='Compute the coarsening quality of loukas method')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--train_indices', type=str, default='0,',
                    help='train indices of the dataset')  # https://bit.ly/3dtJtPn
parser.add_argument('--test_indices', type=str, default='0,', help='test indices of the dataset')
parser.add_argument('--strategy', type=str, default='loukas', help='coarsening strategy', choices=['DK', 'loukas'])
parser.add_argument('--method', type=str, default='variation_edges', help='Loukas methods',
                    choices=['variation_neighborhoods', 'variation_edges', 'variation_cliques',
                             'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'variation_neighborhood',
                             'DK_method'])
from sparsenet.util.args_util import argsparser

if __name__ == '__main__':
    t0 = time()
    args = parser.parse_args()
    AP = argsparser(args)
    args = AP.args

    dev = args.device
    M = monitor()
    fix_seed(seed=args.seed)
    dataset_loader = data_loader(args, dataset=args.dataset)

    train_indices, val_indices, test_indices = AP.set_indices()
    nfeat_dim, efeat_dim, out_dim = 5, 1, 1
    model = GNN_graphpred(args.n_layer, args.emb_dim, nfeat_dim, efeat_dim, out_dim,
                          force_pos=args.force_pos, mlp=args.mlp).to(dev)
    optimizer_gnn = torch.optim.Adam(model.parameters(), args.lr)
    TE = tester(dev=dev)
    logging.basicConfig(level=getattr(logging, args.log.upper()),
                        handlers=[logging.StreamHandler(sys.stdout)])

    ################################################################
    ME = ModelEvaluator(model, dataset_loader, dev, optimizer_gnn)
    ME.set_modelpath(set_model_dir(args, train_indices, val_indices, test_indices))
    model, args = ME.find_best_model(model, train_indices, val_indices, args)
    ME.test_model(model, test_indices, AP, args)
    exit()
    ################################################################


    OUT_PATH = set_model_dir(args, train_indices, val_indices, test_indices)

    if args.testonly:
        model.load_state_dict(torch.load(OUT_PATH + f'checkpoint-best-n-gen.pkl'))
        for idx_ in test_indices:
            args.test_idx = idx_
            TE.set_test_data(args, dataset_loader)
            TE.eval(model, args, verbose=False)
            banner(f'{args.dataset}: finish testing graph {test_indices}.')
        exit()

    # train
    val_score = {}
    best_n_gen = -1e10
    best_impr_ratio = -1e30
    best_eigen_ratio = -1e30
    TR = trainer(dev=dev)

    for idx in train_indices:  # *args.n_cycle:
        args.train_idx = idx
        args.cur_idx = idx
        TR.set_train_data(args, dataset_loader)
        TR.train(model, optimizer_gnn, args, verbose=False)
        TR.delete_train_data(idx)

        # validate
        val_score[idx] = {'n_gen': [], 'impr_ratio': [], 'eigen_ratio': []}
        for idx_ in val_indices:
            args.test_idx = idx_
            args.cur_idx = idx_
            TE.set_test_data(args, dataset_loader)
            n_gen, impr_ratio, eigen_ratio = TE.eval(model, args, verbose=False)

            val_score[idx]['n_gen'].append(n_gen)
            val_score[idx]['impr_ratio'].append(impr_ratio)
            val_score[idx]['eigen_ratio'].append(eigen_ratio)

        banner(f'{args.dataset}: finish validating graph {val_indices}.')
        cur_impr_ratio = np.mean(val_score[idx]['impr_ratio'])
        cur_eigen_ratio = np.mean(val_score[idx]['eigen_ratio'])

        # save the model if it works well on val data
        print(cur_eigen_ratio, best_eigen_ratio)
        if cur_eigen_ratio > best_eigen_ratio:
            best_eigen_ratio = cur_eigen_ratio
            torch.save(model.state_dict(), OUT_PATH + f'checkpoint-best-eigen-ratio.pkl')
            print(red(f'Save model for train idx {idx}. Best-eigen-ratio is {pf(best_eigen_ratio, 2)}.'))

        if cur_impr_ratio > best_impr_ratio:
            best_impr_ratio = cur_impr_ratio
            torch.save(model.state_dict(), OUT_PATH + f'checkpoint-best-improve-ratio.pkl')
            print(red(f'Save model for train idx {idx}. Best-improve-ratio is {pf(best_impr_ratio, 2)}.'))

    for idx_ in val_indices:
        TE.delete_test_data(idx_)

    model_name = AP.set_model_name()
    model.load_state_dict(torch.load(OUT_PATH + model_name))

    for idx_ in test_indices:
        args.test_idx = idx_
        args.cur_idx = idx_
        TE.set_test_data(args, dataset_loader)
        TE.eval(model, args, verbose=False)
        banner(f'{args.dataset}: finish testing graph {idx_}.')

    if args.tbx: TR.writer.close()

