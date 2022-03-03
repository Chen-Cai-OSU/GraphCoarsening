# Created at 2020-05-27
# Summary: test generalization

import os

from sparsenet.util.util import runcmd
from sparsenet.util.name_util import big_ego_graphs
from sparsenet.util.dir_util import PYTHON
python = PYTHON
warn = False
warn_cmd = ' -W ignore ' if not warn else ''


class tester:
    def __init__(self):
        self.loukas_datasets = ['minnesota', 'airfoil', 'yeast', 'bunny']
        self.syn_datasets = ['ws', 'random_geo', 'shape', 'sbm', 'random_er', ]  # ego_facebook
        self.file = 'sparsenet/model/example.py '
        self.methods = ['affinity_GS', 'algebraic_JC', 'heavy_edge', 'variation_edges', 'variation_neighborhoods',
                        ]  # 'heavy_edge' 'affinity_GS', 'kron'
        self.method = ['variation_neighborhood']  # it's best in most cases

        self.args = ' --lap none ' \
            f' --train_idx 0 --test_idx 0  --n_bottomk 40 --force_pos ' \
            f'--n_cycle 100 --seed 0 ' \
            f'--train_indices 0 --test_indices ,'

        self.cmd = f'{python} {warn_cmd} {self.file} {self.args} '

    def viz_test(self):
        train_indices = '0,1,2,3,4,5,6'  # '10,11,' # '0,1,2,3,4,'
        test_indices = '13,14,15,16,17,18,' #'5,6,8,9,10,11,12,13,14,15,15,17,18,19'
        for data in \
                ['faust']:
            for method in self.method:  # [1e-4, 1e-5, 1e-6]:
                special_args = f'--n_epoch 20  --lap None --loukas_quality --bs 600 --lr 1e-3 --ini --viz  '
                cmd = f'{self.cmd} --dataset {data} --ratio 0.5 --strategy loukas ' \
                    f'--method {method} --train_indices {train_indices} --test_indices {test_indices} {special_args}  '
                runcmd(cmd)

    def generalization(self):
        train_indices = \
            '2,3,4,5,6,7,8,9,10,11,12,13,14,'
            # '9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,'
            #'2,3,4,5,6,7,8,9,10,11,12,13,14,'
            # '0,1,2,3,4,'
            # '2,3,4,5,6,7,8,9,'
            # '0'
            # '1,2,3,4,5,'
            # '2,3,4,5,6,7,8,9' \
        test_indices = \
            '0'
            # '5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,'
            # '0'
            # '0,1,2,3,4,5,6,7,8,9,10'
        # '6,7,8,9,10,11,12,13,14,15,16,17,18,19'
        for data in \
                ['coauthor-cs']:
            # self.syn_datasets:
            # ['random_er']:
            # ['random_geo', ]:
            # self.loukas_datasets:

            for method in  self.method:  # [1e-4, 1e-5, 1e-6]:
                special_args = f'--bs 600 --lr 1e-3 --n_epoch 20  --lap sym --device cuda  ' \
                    f'--loss quadratic --n_bottomk 500 --correction --ini --valeigen --w_len 5000 --offset 100 '
                cmd = f'{self.cmd} --dataset {data} --ratio 0.3 --strategy loukas ' \
                    f'--method {method} --train_indices {train_indices} --test_indices {test_indices} {special_args}  '
                # cmd = cmd.replace('--strategy loukas ', '--strategy DK ')
                runcmd(cmd)

    def metric_test(self):
        args = ' --loukas_quality '

        file = 'sparsenet/evaluation/metric.py '
        for data in self.loukas_datasets[:1]:
            for ratio in [.5]:
                for method in self.methods:  # ['variation_neighborhoods']:
                    cmd = f'{python} {warn_cmd} {file} {args} --dataset {data} ' \
                        f'--strategy DK --ratio {ratio} --method {method}'
                    runcmd(cmd)

    def loukas_quality_test(self):
        """ test the effect of using with not using argument loukas_quality. """
        train_indices = '0,'
        test_indices = ','
        for data in ['bunny']:
            for method in ['variation_edges']: # self.methods:
                for ratio in  [0.3, 0.5, 0.7]:
                    special_args = f'--bs 600  --n_epoch 50  --device cuda  '  # --loukas_quality
                    cmd = f'{self.cmd} --dataset {data} --ratio {ratio} --strategy loukas --correction  ' \
                        f'--method {method} --train_indices {train_indices} --test_indices {test_indices} {special_args}  ' # --ini
                    runcmd(cmd)

    def feature_test(self):
        train_indices = '0'
        test_indices = ','
        for data in \
                ['shape']:

            for method in self.methods:
                special_args = f'--bs 600  --n_epoch 50   '
                cmd = f'{self.cmd} --dataset {data} --ratio 0.5 --strategy loukas --device cpu ' \
                    f'--method {method} --train_indices {train_indices} --test_indices {test_indices} {special_args}'
                # cmd = cmd.replace('--strategy loukas ', '--strategy DK ')
                runcmd(cmd)

    def fit_test(self):
        """ test all loukas's datasets """
        datasets = ['bunny']# [ 'airfoil', 'yeast', 'bunny']
        train_indices = '0,,'
        methods = [
            'variation_neighborhoods']  # ['heavy_edge', 'variation_edges',  'algebraic_JC', 'affinity_GS'] # important: exclude and kron

        for data in datasets:
            for ratio in [.5]:
                for method in methods:
                    special_args = ''# '--lr 1e-4 --bs 6000' if data == 'bunny' else ''  # large bs for bunny
                    cmd = f'{self.cmd} --dataset {data} ' \
                        f'--strategy loukas --ratio {ratio} --method {method} --train_indices  {train_indices} --correction --ini {special_args} '
                    runcmd(cmd)

    def otherloss_test(self):
        """ test all loukas's datasets """
        datasets = ['ws', ]
        train_indices =  '0,1,2,3,'# '0,1,2,3,4,'
        test_indices = '5,6,7,8,9,10,11,' # '5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,' # '10,11,12,13,14,15,16,17,18,19,' # '5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,' # '5,6,7,8,9,10,11,12,13,14,15,16'
        methods = self.method # [ ]  # ['heavy_edge', 'variation_edges',  'algebraic_JC', 'affinity_GS']

        for data in datasets:
            for ratio in [.7]:
                for method in methods: # ['affinity_GS']:
                    self.cmd = self.cmd.replace('--n_bottomk 40 ', '--n_bottomk 40 ')
                    # self.cmd = self.cmd.replace('--lap none ', '--lap sym ')
                    cmd = f'{self.cmd} --dataset {data} ' f' --n_epoch 20  --device cpu --w_len 5000 ' \
                        f'--ratio {ratio} --method {method} --loss quadratic  --dynamic  --ini --valeigen' \
                        f'--train_indices {train_indices} --test_indices {test_indices} --n_layer 3 --emb_dim 50  '
                    # cmd += ' --strategy DK '
                    # cmd = cmd.replace(method, 'DK_method')
                    runcmd(cmd)

    def debug_test(self):
        args = ' --n_epoch 50 --lap none ' \
            f' --train_idx 0 --test_idx 0  --n_bottomk 40 --force_pos ' \
            f'--n_cycle 100 --device cuda --seed 0 ' \
            f'--train_indices 0, --test_indices ,'
        kwargs = {'dataset': 'random_geo', 'n_bottomk': 40, 'ratio': 0.7, 'seed': 0, 'method': 'variation_edges'}
        # {'dataset': 'ws', 'n_bottomk': 40, 'ratio': 0.7, 'seed': 0, 'method': 'variation_edges'}

        cmd = f'{python} {warn_cmd} {self.file} {args} --dataset {kwargs["dataset"]} ' \
            f'--strategy loukas --ratio {kwargs["ratio"]} --method {kwargs["method"]} --seed {kwargs["seed"]}'
        runcmd(cmd)

    def local_var_nbr_test(self):
        args = ' --n_epoch 50 --lap none ' \
            f' --train_idx 0 --test_idx 0  --n_bottomk 40 --force_pos ' \
            f'--n_cycle 1 --device cuda --seed 0 ' \
            f'--train_indices 0, --test_indices ,'

        for data in ['bunny']: #self.loukas_datasets:
            for ratio in [.5]:
                for method in ['variation_neighborhoods']:
                    special_args = '--lr 1e-4 --bs 6000 --ini ' if data == 'bunny' else ''  # large bs for bunny
                    cmd = f'{python} {warn_cmd} {self.file} {args} --dataset {data} ' \
                        f'--strategy loukas --ratio {ratio} --method {method} {special_args}'
                    runcmd(cmd)


if __name__ == '__main__':
    # tester().feature_test()
    # tester().local_var_nbr_test()
    # tester().loukas_quality_test()
    # tester().generalization()
    # tester().viz_test()
    tester().otherloss_test()
    # tester().metric_test()
    # tester().fit_test()

    exit()
    for data in \
            ['minnesota', 'bunny', 'airfoil', 'yeast']:
        # ['random_er', 'random_geo']: #

        cmd = f'{python} {warn_cmd} sparsenet/model/example.py --bs 600 --n_epoch 30 --lap none ' \
            f' --train_idx 0 --test_idx 0 --dataset {data} --n_bottomk 40 --ratio 0.5 --force_pos ' \
            f'--n_cycle 100 --device cuda --seed 0'  # # --lap_check
        print(cmd)
        os.system(cmd)
