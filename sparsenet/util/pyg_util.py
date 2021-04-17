# Created at 2020-06-07
# Summary:
import os
n=2
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['OMP_NUM_THREADS'] = str(n)
os.environ['OPENBLAS_NUM_THREADS'] = str(n)
os.environ['MKL_NUM_THREADS'] = str(n)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
os.environ['NUMEXPR_NUM_THREADS'] = str(n)
import torch
torch.set_num_threads(n) # always import this first
status = f'{n}'
print(f'thread status {__file__}: {status}')

import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

from sparsenet.util.data import EgoGraphs, precompute_eig, shape_data, syth_graphs, loukas_data
from sparsenet.util.name_util import big_ego_graphs, syn_graphs, loukas_datasets
from sparsenet.util.util import summary, \
    random_pygeo_graph


class NonEgoGraphs(InMemoryDataset):
    """ similar to EgoGraphs, but for other graphs (sythetic graphs + Loukas's dataset ) """

    def __init__(self, dataset=None, transform=None, pre_transform=None):
        """ kwargs for function egographs """
        data_dir = os.path.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
        nonegograph_dir = os.path.join(data_dir, 'nonegographs')
        self.dataset = dataset
        root = os.path.join(nonegograph_dir, self.dataset)

        super(NonEgoGraphs, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1']

    @property
    def processed_file_names(self):
        return self.dataset  # dict2name(self.egograph_kwargs)

    def download(self):
        pass

    def _select_datasets(self):
        dataset = self.dataset
        if dataset == 'shape':
            datasets = shape_data(50, _k=10)
        elif dataset == 'faust':
            datasets = shape_data(50, _k=10, name='FAUST')
        elif dataset == 'random_geo':
            datasets = syth_graphs(n=50, size=700, type='geo')  # random_geo(n=10, size=512)
        elif dataset == 'random_er':
            datasets = syth_graphs(n=50, size=512, type='er')  # random_er(n=10, size=512)
        elif dataset in ['sbm', 'ws', 'ba']:
            datasets = syth_graphs(n=50, size=512, type=dataset)
        elif dataset in ['yeast', 'airfoil', 'bunny', 'minnesota']:
            datasets = loukas_data(name=dataset)
        else:
            NotImplementedError

        return datasets

    def process(self):
        # Read data into huge `Data` list.

        data_list = self._select_datasets()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

import argparse
parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--dataset', type=str, default='Coauthor-CS', help='dataset for egographs')
parser.add_argument('--lap', type=str, default='None')
parser.add_argument('--n_vec', type=int, default=100)
parser.add_argument('--w_len', type=int, default=5000)


if __name__ == '__main__':
    def main():
        args = parser.parse_args()
        for dataset in [args.dataset]:  # big_ego_graphs:  # ['CiteSeer','PubMed', 'wiki-vote']:
            for size in [50]:  # [20]:
                # kwargs = {"hop": -1, "size": size, "dataset": dataset, 's_low': -1, 's_high': -1, 'sample': 'rw',
                #           'n_vec':args.n_vec, 'w_len':args.w_len, 'include_self': False}
                kwargs = {'dataset': 'flickr', 'hop': -1, 'size': 1, 's_low': -1, 's_high': -1,
                          'sample': 'rw', 'n_vec': 500, 'w_len': 15000, 'include_self': True}

                data = EgoGraphs(**kwargs)
                for d in data:
                    print(d)
                # print(data[0])
            # for idx, g in enumerate(data):
            #     g = clip_feat(g, args, dim=52)
            #     if idx < 5: print(g)
            # new_data = [clip_feat(g, args, dim=52) for g in data]
        del data
        print('hello')

    main()
    exit()

    g = syth_graphs(1, size=1000)
    summary(g)
    exit()
    for dataset in ['faust']:  # syn_graphs + loukas_datasets:
        data = NonEgoGraphs(dataset=dataset)
        for i, d in enumerate(data):
            summary(d, i, highlight=True)

    exit()


    # dir = os.path.join(data_dir, 'wiki-vote')
    for dataset in [args.dataset]:  # big_ego_graphs:  # ['CiteSeer','PubMed', 'wiki-vote']:
        for hop in [2,]:  # [3,4,5]:  # [3, 4, 5, 6]:  # [2,3,4,5,6]:
            for size in [20]:  # [20]:
                s_low = 5000 if dataset in big_ego_graphs else 200
                s_high = 10000 if dataset in big_ego_graphs else 5000
                kwargs = {"hop": hop, "size": size, "dataset": dataset, 's_low': s_low, 's_high': s_high}
                data = EgoGraphs(**kwargs)
                print(data[0])
                # continue

                if False:  # dataset == 'wiki-vote' and hop==3 and size==10:
                    print(data)
                    print(data[0]['None_vals'][:5])
                    for g in data:
                        summary(g)
    exit()


    for dataset in syn_graphs + loukas_datasets:
        data = NonEgoGraphs(dataset=dataset)
        summary(data[0], dataset, highlight=True)
        # data_cmp = syth_graphs(n=20, size=512, type='ws')
        # print(data_cmp[5])

    exit()
    g = random_pygeo_graph(1000, 1, 20000, 1)
    summary(g, 'beofre')
    g = precompute_eig(g)
    summary(g, 'after')
    exit()
