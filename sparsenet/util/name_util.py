# Created at 2020-06-06
# Summary: some global names

import os
import os.path as osp

from sparsenet.util.util import fig_dir, make_dir, model_dir

loukas_datasets = ['minnesota', 'airfoil', 'yeast', 'bunny']
methods = ['variation_edges', 'variation_neighborhoods', 'algebraic_JC', 'heavy_edge' 'affinity_GS']
syn_graphs = ['random_geo', 'random_er', 'ws', 'ba', 'shape']

big_ego_graphs = ['PubMed', 'Coauthor-CS', 'Coauthor-physics', 'Amazon-photo', 'Amazon-computers', 'yelp', 'reddit',
                  'flickr']
ego_graphs = big_ego_graphs + ['CiteSeer']


def set_figname(args, name='subgraph'):
    """ used to set dir where figure is saved"""
    dir = os.path.join(fig_dir(), args.dataset,
                       f'ratio_{args.ratio}',
                       f'method_{args.method}',
                       f'n_epoch_{args.n_epoch}',
                       f'n_bottomk_{args.n_bottomk}',
                       f'lap_{args.lap}',
                       '')
    dir = dir.replace('_', '-')
    make_dir(dir)
    name = name.replace('_', '-')
    return dir + name + '.pdf'


def set_model_dir(args, train_indices, val_indices, test_indices):
    """ used to set dir where model is saved """
    OUT_PATH = os.path.join(model_dir(), args.dataset,
                            f'ratio_{args.ratio}',
                            f'strategy_{args.strategy}',
                            f'method_{args.method}',
                            f'train_{len(train_indices)}',
                            f'val_{len(val_indices)}',
                            f'test_{len(test_indices)}',
                            f'loss_{args.loss}',
                            f'n_epoch_{args.n_epoch}',
                            f'n_bottomk_{args.n_bottomk}',
                            f'lap_{args.lap}',
                            f'bs_{args.bs}',
                            f'lr_{args.lr}',
                            f'ini_{args.ini}',
                            # f'correction_{args.correction}'
                            '')
    if args.dataset in ['coauthor-cs', 'coauthor-physics', 'flickr', 'pubmeds']:
        OUT_PATH = os.path.join(OUT_PATH, f'w_len_{args.w_len}', '')
    make_dir(OUT_PATH)
    return OUT_PATH


def set_coarsening_graph_dir(args):
    coarse_dir = osp.join(model_dir(), '..', 'coarse_graph')

    if args.strategy == 'loukas':
        dir = osp.join(coarse_dir,
                       'loukas',
                       args.dataset,
                       f'ratio_{args.ratio}',
                       f'method_{args.method}',
                       f'n_bottomk_{args.n_bottomk}',
                       f'cur_idx_{args.cur_idx}',
                       '')

    elif args.strategy == 'DK':
        dir = osp.join(coarse_dir,
                       'DK',
                       args.dataset,
                       f'ratio_{args.ratio}',
                       f'cur_idx_{args.cur_idx}',
                       '')

    else:
        raise NotImplementedError

    if args.dataset in ['coauthor-cs', 'coauthor-physics', 'flickr', 'pubmeds']:
        dir = os.path.join(dir, f'w_len_{args.w_len}', '')

    make_dir(dir)
    return dir


def set_eigenvec_dir(args):
    eig_dir = osp.join(model_dir(), '..', 'eigenvec')
    assert args.dataset in ['coauthor-cs', 'coauthor-physics', 'flickr', 'pubmeds']
    dir = osp.join(eig_dir,
                   args.strategy,
                   args.dataset,
                   f'ratio_{args.ratio}',
                   f'method_{args.method}',
                   f'n_bottomk_{args.n_bottomk}',
                   f'cur_idx_{args.cur_idx}',
                   f'w_len_{args.w_len}',
                   '')
    make_dir(dir)
    return dir
