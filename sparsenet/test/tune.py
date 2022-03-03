# Created at 2020-05-29
# Summary: tune the main.py

from signor.configs.util import load_configs, dict_product, dict2arg
from signor.utils.cli import runcmd
from signor.utils.dict import update_dict
from signor.utils.shell.parallel_args import write_cmds_for_parallel
from sparsenet.util.dir_util import PYTHON

import argparse

parser = argparse.ArgumentParser(description='run all exps')
parser.add_argument('--loukas', action='store_true', help='run experiments on Loukas JMLR paper')
parser.add_argument('--run', action='store_true', help='run all exp')

python = f'{PYTHON} -W ignore  '
fname = 'sparsenet/test/main.py '

class tuner:
    def __init__(self, python=python, f=fname, config_f=None):
        self.python = python
        self.f = f
        self.cfg_dic = load_configs(config_f)
        self.comb = dict_product(self.cfg_dic)

    def gen_cmds(self, verbose=False):
        self.cmds = []
        for arg_dict in self.comb:
            arg = dict2arg(arg_dict, sacred=True)
            cmd = ' '.join([self.python, self.f, arg])
            self.cmds.append(cmd)
            if verbose: print(cmd)

    def execute(self, firstk = None, mode='serial', dryrun=True, **kwargs):
        """
        :param firstk:
        :param mode:
        :param dryrun:
        :param kwargs: for gnu mode
        :return:
        """
        n = min(len(self.cmds), firstk) if firstk is not None else len(self.cmds)

        if mode == 'serial':
            for cmd in self.cmds[:n]:
                runcmd(cmd, print_only=dryrun)

        elif mode == 'gnu':
            parallel_kwargs = {'exe': not dryrun, 'nohup': True, 'jobs': 2, 'multi_gpu': None, 'file': 'conductance', 'shuffle':False}
            parallel_kwargs = update_dict(kwargs, parallel_kwargs)
            file = write_cmds_for_parallel(self.cmds[:n], **parallel_kwargs) # important
            return file
        else:
            NotImplementedError



if __name__ == '__main__':
    args = parser.parse_args()
    config_file = './sparsenet/test/configs/conductance.yaml'
    # config_file = 'sparsenet/test/configs/loukas.yaml' if args.loukas else 'sparsenet/test/configs/generalize.yaml'

    T = tuner(config_f=config_file)
    print(T.cfg_dic)
    T.gen_cmds()

    if args.run:
        T.execute(firstk=None, mode='gnu', dryrun=False)
    else:
        T.execute(firstk=None, mode='gnu', dryrun=True)

