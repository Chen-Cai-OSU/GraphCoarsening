# Created at 2020-06-23
# Summary: generate all results for the paper
import argparse
import os
from pprint import pprint

from sparsenet.test.configs.generate_configs import config_generator
from sparsenet.test.tune import tuner
from sparsenet.util.util import update_dict, runcmd

python = '/home/cai.507/anaconda3/envs/sparsifier/bin/python -W ignore  '
fname = 'sparsenet/test/main.py '
config_dir = './sparsenet/test/configs/'


def task(config_f, **kw):
    """
    :param config_f:
    :param kw: kwargs for tuner.execute
    :return: cmd used for chaining all cmds
    """
    config_file = config_f  # './sparsenet/test/configs/generalize.yaml'
    T = tuner(config_f=config_file)
    pprint(T.cfg_dic)
    T.gen_cmds()
    file = T.execute(firstk=kw.get('firstk', None), mode='gnu', dryrun=not args.run, **kw)

    cmd = f"nohup parallel --jobs {kw.get('jobs', 5)} < {file} > {kw.get('file', 'tmp')}.log;"
    return cmd


parser = argparse.ArgumentParser(description='run all exps')
parser.add_argument('--loukas', action='store_true', help='run experiments on Loukas JMLR paper')
parser.add_argument('--run', action='store_true', help='run all exp')
parser.add_argument('--runall', action='store_true', help='run all exp')

if __name__ == '__main__':
    args = parser.parse_args()
    default_kwargs = {'nohup': True, 'jobs': 1, 'multi_gpu': None, 'file': 'tmp_', 'shuffle': True} # , 'firstk': None
    cmds = []

    config_f = os.path.join(config_dir, 'MLP_config','MLP_sml.yaml')
    kwargs = {'jobs': 30, 'file': 'eigen_syn', 'multi_gpu': False, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()

    for data in ['big']:
        for method in  [ 'DK']:
            f = f'{data}'
            if method == 'DK': f += '_DK'
            f += '.yaml'

            config_f = os.path.join(config_dir, 'norm_quad', f)
            G = config_generator(data=data, method=method, lap='sym', loss='quad', eigen=False)
            kwargs = {'jobs': G.set_njobs(), 'file': f'{data}_{method}', 'multi_gpu': False, 'shuffle': False}

            kwargs = update_dict(kwargs, default_kwargs)
            cmd = task(config_f, **kwargs)
            cmds.append(cmd)

    final_cmd = ' '.join(cmds)
    for cmd in cmds:
        print(cmd)
    print(final_cmd)
    runcmd(final_cmd)
    exit()

    config_f = os.path.join(config_dir, 'comb_quad', 'small.yaml')
    # config_f = os.path.join(config_dir, 'normalize_ego_lap.yaml')
    kwargs = {'jobs': 10, 'file': 'normalize_ego_lap', 'multi_gpu': False, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()


    config_f = os.path.join(config_dir, 'normalize_lap.yaml')
    kwargs = {'jobs': 20, 'file': 'normalize_lap', 'multi_gpu': False, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()


    config_f = os.path.join(config_dir, 'conductance.yaml')
    kwargs = {'jobs': 5, 'file': 'conductance', 'multi_gpu': False, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()

    config_f = os.path.join(config_dir, 'eigen_syn.yaml')
    kwargs = {'jobs': 6, 'file': 'eigen_syn'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)
    exit()

    config_f = os.path.join(config_dir, 'cmp_eigen.yaml')
    kwargs = {'jobs': 15, 'file': 'cmp_eigen', 'multi_gpu': None, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()

    config_f = os.path.join(config_dir, 'conductance_large.yaml')
    kwargs = {'jobs': 3, 'file': 'conductance_large', 'multi_gpu': None, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    exit()




    config_f = os.path.join(config_dir, 'eigen.yaml')
    kwargs = {'jobs': 10, 'file': 'eigen'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)
    # use parallel to run all scripts
    for cmd in cmds:
        print(cmd)

    if args.runall:
        final_cmd = ' '.join(cmds)
        print(final_cmd)
        os.system(final_cmd)
    exit()

    exit()
    # syn graphs (150). All ws/shape/er/ws/ba/, two laplacians. (No coauthors)
    config_f = os.path.join(config_dir, 'generalize.yaml')
    kwargs = {'jobs': 20, 'file': 'generalize'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)

    # egographs (60). coauthor-cs/coauthor-physics/pubmeds/flickr
    config_f = os.path.join(config_dir, 'egographs.yaml')
    kwargs = {'jobs': 10, 'file': 'egographs'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)

    # DK (80). ws/shape/er/ws/ba/ + two laplacians and conductance (important: removed for a moment)
    config_f = os.path.join(config_dir, 'DK.yaml')
    kwargs = {'jobs': 20, 'file': 'DK'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)


    # conductance (100). ws/shape/er/ws/ba
    config_f = os.path.join(config_dir, 'conductance.yaml')
    kwargs = {'jobs': 2, 'file': 'conductance', 'multi_gpu': True, 'shuffle': False}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmd = cmd.replace(';', '')
    cmd += ' &'
    if args.run or args.runall:
        os.system(cmd)


    ########## coauthors ##########

    # (16) DK-coauthors
    config_f = os.path.join(config_dir, 'DK_coauthors.yaml')
    kwargs = {'jobs': 10, 'file': 'DK_coauthors'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)

    # co-authors (40)
    config_f = os.path.join(config_dir, 'generalize_coauthors.yaml')
    kwargs = {'jobs': 10, 'file': 'generalize_coauthors'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)

    # DK_egographs (40)
    config_f = os.path.join(config_dir, 'DK_egographs.yaml')
    kwargs = {'jobs': 8, 'file': 'DK_egographs'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    cmds.append(cmd)

    ########## viz ##########
    # viz (20). shape
    config_f = os.path.join(config_dir, 'viz.yaml')
    kwargs = {'jobs': 1, 'file': 'viz'}
    kwargs = update_dict(kwargs, default_kwargs)
    cmd = task(config_f, **kwargs)
    # cmd = cmd.replace(';', '')
    # cmd += ' &'
    # print(cmd)
    # if args.run or args.runall:
    #     os.system(cmd)
    # cmds.append(cmd)

