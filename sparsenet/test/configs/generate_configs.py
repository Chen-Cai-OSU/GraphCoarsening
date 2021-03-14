# Created at 20200910
# Summary: generate config file
import os
from pprint import pprint

from signor.configs.util import load_configs
from signor.ioio.dir import cur_dir
from sparsenet.util.util import make_dir
from yaml import dump
graphs = {'big': ['pubmeds', 'coauthor-cs', 'coauthor-physics', 'flickr'],
          'small': ['random_er', 'shape', 'random_geo', 'ws', 'ba']}
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq

# todo: change later. this is only for test.
# graphs = {'big': ['coauthor-cs'],
#           'small': ['random_geo', ]
#           }

methods = {'DK': ['DK_method'],
            'loukas': ["variation_neighborhood", 'heavy_edge', 'variation_edges', 'algebraic_JC', 'affinity_GS']}

# methods = {'DK': ['DK_method'],
#            'loukas': ["variation_neighborhood"]}

def cvter(lis):
    # convert a list of num to "\"2,3,4,5,6,7,8,9,10,11,12,13,14,\""
    l = [str(l) for l in lis]
    l.append('')
    l = ','.join(l)
    # return f'\\"{l}\\'  #
    return  r'"\"' + l + r'"\"'

indices_big = {'train': cvter(list(range(2, 15))), 'test': cvter([0])}
indices_small = {'train': cvter(list(range(5))), 'test': cvter(list(range(5, 25)))}


class config_generator(object):
    def __init__(self, data='small', method='loukas', lap='none', loss='quad', eigen=False):
        self.data = data
        self.method = method
        self.lap = lap
        self.loss = loss
        self.eigen = eigen
        self._set_indices()

        assert self.data in ['big', 'small']
        assert self.method in ['DK', 'loukas']
        assert self.lap in ['none', 'sym']
        assert self.loss in ['quad', 'cond']

        self.cfg_key = ['ini', 'lap', 'device', 'n_epoch', 'strategy', 'method', 'n_bottomk', 'bs', 'ratio', 'dataset',
                        'train_indices', 'test_indices', 'seed', 'w_len', 'offset', 'correction', 'valeigen', 'dynamic']

    def generate_dict(self, yaml=False, check = False):
        dict = {}
        for key in self.cfg_key:
            dict[key] = {}

        for key in self.cfg_key:
            method = f'_set_{key}'
            ret = self.__getattribute__(method)()
            dict[key]['values'] = ret if isinstance(ret, list) else [ret]
            dict[key]['format'] = 'values'
            dict[key]['dtype'] = dict[key]['values'][0].__class__.__name__

        print(dict)

        if yaml:
            file = self.set_folder() + self.set_yamlf()
            print(file)
            with open(file, 'w') as f:
                dump(dict, f, default_style=None)

        if check:
            dict = load_configs(file)

            print(dict)

        return dict

    def set_folder(self):
        if self.eigen:
            f = 'eigenloss'
        elif self.loss == 'quad':
            f = 'comb_quad' if self.lap == 'none' else 'norm_quad'
        elif self.loss == 'cond':
            f = 'conductance'
        else:
            raise NotImplementedError
        curdir = os.path.join(eval(cur_dir()), f, '')
        make_dir(curdir)
        return curdir

    def set_yamlf(self):
        f = self.data
        if self.method == 'DK':
            f += '_DK'
        return f'{f}.yaml'

    def _set_valeigen(self):
        if self.lap == 'none' and self.eigen:
            return True
        else:
            return False

    def _set_dataset(self):
        return graphs[self.data]

    def _set_method(self):
        return methods[self.method]

    def _set_strategy(self):
        if self.method == 'DK':
            return 'DK'
        else:
            return 'loukas'

    def _set_indices(self):
        if self.data == 'big':
            indices = indices_big
        else:
            indices = indices_small
        self.indices = indices

    def _set_train_indices(self):
        return self.indices['train']

    def _set_test_indices(self):
        return self.indices['test']

    def _set_lap(self):
        return self.lap

    def _set_device(self):
        if self.loss == 'cond':
            return 'cuda'
        else:
            return 'cpu'

    def _set_correction(self):
        if self.eigen:
            assert self.lap == 'none' and self.loss == 'quad'
            return True
        else:
            return False

    def _set_n_bottomk(self):
        if self.lap == 'sym' and self.loss == 'quad' and self.data == 'big':
            return 200
        elif self.eigen and self.data == 'big':
            return 200 # remark: will be slow. double check.
        elif self.loss == 'cond'and self.data == 'big':
            return 10
        elif self.data == 'small':
            return 40
        else:
            return 200

    def _set_ini(self):
        if self.eigen and self.data == 'big': # remark: maybe also need to set true for small graph
            return True
        else:
            return False

    def _set_seed(self):
        return 0

    def _set_w_len(self):
        if self.eigen and self.data == 'big':
            return 15000 # todo: is it really helpng?
        elif self.lap == 'sym' and self.data == 'big' and self.loss == 'quad':
            return 5000 # still use 5000 beacsue 15000 dosen't seem help much and also takes much longer (4-5 times) time
        else:
            return 5000

    def _set_offset(self):
        # remark: maybe not very useful
        return 0

    def _set_n_epoch(self):
        if self.loss == 'quad' and self.data == 'big' and self.lap == 'sym':
            return 30

        if self.loss == 'cond':
            return 10

        if self.data == 'small':
            return 50
        else:
            return 20

    def _set_bs(self):
        return 600

    def _set_ratio(self):
        return [0.3, 0.5, 0.7] if self.data =='small' else [0.3, 0.5, 0.7, 0.9]

    def set_njobs(self):
        if self.loss == 'cond':
            return 5 if self.data == 'small' else 3

        if self.data == 'small':
            return 15
        else:
            return 10

    def _set_dynamic(self):
        if self.lap == 'sym' and self.loss == 'quad':
            return True
        else:
            return False


if __name__ == '__main__':
    # G = config_generator(data='small', method='loukas', lap='none', loss='quad', eigen=False)
    # file = G.set_folder() + G.set_yamlf()
    # load_configs(file)
    # exit()

    for data in ['small', 'big']:
        for method in ['loukas', 'DK']:
            # quad loss for combinatorial laplace
            G = config_generator(data=data, method=method, lap='none', loss='quad', eigen=False)
            G.generate_dict(yaml=True, check=True)

            # quad loss for normalized laplace
            G = config_generator(data=data, method=method, lap='sym', loss='quad', eigen=False)
            G.generate_dict(yaml=True)

            # quad loss for normalized laplace (eigen loss)
            G = config_generator(data=data, method=method, lap='none', loss='quad', eigen=True)
            G.generate_dict(yaml=True)

            # conductance loss for normalized laplace
            # G = config_generator(data=data, method=method, lap='none', loss='cond', eigen=False)
            # G.generate_dict(yaml=True)
