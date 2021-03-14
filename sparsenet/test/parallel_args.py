# Created at 2020-05-18
# Summary: run test in parallel

""" implement loop over args quickly
    python file --arg 1
    python file --arg 2
    python file --arg 3

    run the above cmds simulatenously and save the result at
    ./parallel_args/file/1.log
    ./parallel_args/file/2.log
    ./parallel_args/file/3.log
    ...
"""

import os
from sparsenet.util.util import tf, make_dir, banner, args_print, one_liner, sig_dir


class parallel_args():
    def __init__(self, python, file, arg, range):
        """
        :param python:
        :param file: **.py include non-modified args
        :param arg: modified arg
        """

        self.python = python
        if '-u' not in self.python:
            self.python += ' -u '
        if file[-1] != ' ': file = file + ' '
        self.file = file
        self.arg = arg
        self.range = range
        self.dir = f'./parallel_args/{self.getfile(file)}/'
        make_dir(self.dir)

    @tf
    def gen_cmds(self, **kwargs):
        cmds = []
        for v in self.range:
            arg = f'--{self.arg} {v}'
            cmd = ' '.join([self.python, self.file, arg])
            if v == '': v = 'none'
            os.system(f'echo {cmd} > {self.dir + str(v)}.log')

            if kwargs.get('direct', False) or kwargs.get('nohup', False):
                if v == '': v = 'none'
                cmd += f' >> {self.dir + str(v)}.log'

            if kwargs.get('nohup', False):
                cmd = 'nohup ' + cmd + ' ; ' # change back to &

            cmds.append(cmd)
        self.cmds = cmds

    @tf
    def run(self, print_only=True, gnu=False, **kwargs):
        """

        :param print_only:
        :param gnu: use gnu parallel
        :param kwargs:
        :return:
        """
        self.gen_cmds(**kwargs)

        if gnu:
            cmds = [cmd.replace('nohup', '') for cmd in self.cmds]
            cmds = [cmd.replace('&', '') for cmd in cmds]
            exe = not print_only
            write_cmds_for_parallel(cmds, exe=exe, nohup=True)
            return

        for cmd in self.cmds:
            args_print(cmd)
            if not print_only:
                os.system(cmd)

    def getfile(self, file):
        """ from "abc.py --a 10" get abc """
        res = file.split(' ')[0]
        assert '.py' in res, f'support .py file only. Got {res}/{file}'
        res = res.split('/')[-1]
        return res[:-3]


def write_cmds_for_parallel(cmds, exe=False, nohup=False, **kwargs):
    """
    write a list of cmds into a script that will be used for gnu parallel bin
    :param cmds: a list of cmds
    :param exe: execute with parallel
    :return:

    cmds = ['sleep 10'] * 20
    write_cmds_for_parallel(cmds, exe=True, jobs=10)

    """

    assert isinstance(cmds, list)
    cmds = [one_liner(cmd) for cmd in cmds]
    cmds = [cmd + ' 2>&1' for cmd in cmds]  # direct both output and error to file
    cmds = [cmd + '\n' for cmd in cmds if cmd[-1] != '\n']

    file = f'{sig_dir()}utils/scheduler/tmp_.sh'
    with open(file, 'w') as f:
        f.writelines('#!/usr/bin/env bash\n')
        f.writelines(cmds)

    banner(f'parallel executing cmds from {file}')
    cmd = f'cat {file}'
    os.system(cmd)

    cmd = f' time parallel --jobs {kwargs.get("jobs", 5)} < {file} '
    if exe:
        if nohup: cmd = 'nohup ' + cmd + '&'
        banner(cmd)
        os.system(cmd)
    else:
        banner('No Exe: ' + cmd)


if __name__ == '__main__':
    # python

    python = '/home/cai.507/anaconda3/envs/sparsifier/bin/python  -W ignore '

    file = \
        'sparsenet/util/pyg_util.py --n_vec 500  --dataset flickr'
    # 'sparsenet/util/cut_util.py'
    # 'sparsenet/util/cut_util.py'
    # 'sparsenet/model/example.py --bs 600 --n_epoch 200 --lap_check --train_idx 1 --n_bottomk 50 --ratio 0.5'
    # 'sparsenet/model/example.py --n_epoch 300 --lap_check --bs 512  --n_layer 4'#

    arg = \
    'w_len'
        # 'dataset'
    # 'dataset'
    # 'dataset'
    # "n_bottomk"
    # "lap"
    # "idx"

    range = \
        [5000, 10000]
        # ['PubMed']
        #[15000, 20000, ]
    # ['er', 'ws', 'ba', 'geo']
    # [2,4,8]
    # ['Amazon-photo', 'Amazon-computers', 'Coauthor-physics', 'Coauthor-CS', 'wiki-vote']
    # ['ws', 'geo', 'er', 'ba', 'sbm']
    # ['CiteSeer', 'PubMed', 'wiki-vote', 'yelp', 'reddit', 'Coauthor-CS', 'Coauthor-physics', 'Amazon-photo', 'Amazon-computers']
    # ['random_er', 'random_geo', 'sbm', 'shape', 'ego_facebook']
    # [10, 20, 30, 40, 50, 60, 70, 80]
    # ['none', 'sym', 'rw']
    # list(range(10))

    prun = parallel_args(python, file, arg, range)
    prun.run(nohup=True, print_only=False, gnu=False, jobs=5)
