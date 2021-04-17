# Created at 2020-06-04
# Summary: database utils

import numpy as np
from signor.utils.database.mongodb_util import mongo_util
from signor.utils.dict import filter_dict

# from signor.ioio.dir import data
from sparsenet.util.util import banner, red, hasany, slicestr, timefunc, pf


class str_parser:
    def __init__(self):
        pass

    def get_error_eigenvalue(self, s):
        " s = error_eigenvalue: array (float64) of shape    (40,)     Nan ratio:      0.0.     0.206(mean)      0.0(min)    0.248(max)    0.214(median)    0.039(std)     40.0(unique) "
        assert 'error_eigenvalue' in s, f'Expect error_eigenvalue in {s}'
        ret = slicestr(s, f='Nan ratio', t='(mean')
        return np.float(ret[-7:])

    def get_eigenloss(self, s):
        assert 'Eigenloss' in s
        # s = 'INFO:root:Idx 0-Epoch: 40. Train(2.0): 0.0(0.048) / 0.0(0.232) / 0.1. Eigenloss: 0.037. Subspaceloss: 0.0'
        ret = slicestr(s, f='Eigenloss', t='Subspace')
        ret = ret.split(':')  # ['Eigenloss', ' 0.037. ']
        ret = ret[-1].replace(' ', '')  # 0.037.
        return np.float(ret.rstrip('.'))


class parser(object):
    def __init__(self, df):
        self.df = df
        self.curlog = None
        self.warn_msg = ['Convergence Error', 'ArpackNoConvergence', "can't allocate memory", 'MemoryError',
                         'CUDA error: out of memory', ' CUDA out of memory', 'CUDA error', ' No such file or directory',
                         'Killed']
        self.train_str = ['error_eigenvalue', 'Train']
        self.strP = str_parser()

    def _get_log(self, s, allowed_list=['INFO']):
        """
        :param s: a long string. df['result'][2300]
        :param allowed_list: a list of allowed strings
        :return: a list of strings only filters are allowed
        """
        assert isinstance(s, str), f'Expect string. Got {type(s)}'
        for msg in self.warn_msg:
            if msg in s:
                # warn(msg) # todo: understand why it only appear once
                print(red(msg))
                return  # it's better not to turn anything when there is error
                # exit()

        s = s.split('\n')
        s = [line for line in s if hasany(line, allowed_list)]  # line.startswith('INFO')
        return s

    def _get_test_error(self, idx, entry='Test', verbose=True):
        """
        :param entry of interest: filter log according to this
        :param s: df['result'][2300]
        :return: a list of filtered string, raw string df['result'][idx]
        """
        s = self.df['result'][idx]
        banner(f'{entry} Log', ch='-', compact=True, length=40)
        s_list = self._get_log(s)
        if s_list is None:
            return [], ''

        lines = [line for line in s_list if entry in line]
        if verbose:
            for line in lines:
                print(line)
        return lines, s

    def _get_train_his(self, idx, allowed_list=[]):
        s = self.df['result'][idx]
        banner('Train Log', ch='-', compact=True, length=40)
        s_list = self._get_log(s, allowed_list=['INFO'] + allowed_list)

        lines = []
        for line in s_list:
            if hasany(line, allowed_list):
                lines.append(line)
        return lines

    def fit_better(self, idx, params):
        """ used to find the entries related to the training error """
        params = filter_dict(params, keys=['dataset', 'method', 'ratio', 'seed', 'n_bottomk'], inclde=True)
        print(f"Parameters: {params}")
        lines = self._get_train_his(idx, allowed_list=['error_eigenvalue', 'Epoch: 50', ])
        for line in lines:
            # print(line)
            try:
                loukas_algo = self.strP.get_error_eigenvalue(line)
                print('loukas algo:', loukas_algo)
                params['loukas'] = loukas_algo
            except:
                eigenloss = self.strP.get_eigenloss(line)
                print('error with learning:', eigenloss)
                params['eigenloss'] = eigenloss
        return params  # one dict

    def generalize_better(self, idx, params, check=False):
        params = filter_dict(params, keys=['dataset', 'method', 'ratio', 'seed', 'n_bottomk', 'lap'], inclde=True)
        print(f"Parameters: {params}")
        lines, s = self._get_test_error(idx, verbose=True, entry='Test')

        if len(lines) not in [3, 9] and check:
            banner('Check why the test results is less than 9.')
            print(s)
        return lines

    def get_summary(self, idx, debug=False, verbose=False):
        """ return a parameter list
        {'bs': 600, 'n_epoch': 50, 'train_indices': '0,1,2,', 'test_indices': '3,4,5,',
        'dataset': 'ego_facebook', 'n_bottomk': 40, 'n_layer': 3, 'ratio': 0.7,
        'n_cycle': 1, 'device': 'cpu', 'seed': 0}
        """

        if debug:
            print(self.df['result'][idx])
            banner(f'All info for Exp {idx}')
            exit()

        if self.df['status'][idx] != 'COMPLETED':
            print(f"Exp {idx} is not Completed ({self.df['status'][idx]}).")
            return

        params = self.df['config'][idx]
        dur_time = self.df['stop_time'][idx] - self.df['start_time'][idx]
        run_time = dur_time.seconds / 60.0

        banner(red(f'Exp {idx}.'))
        print(f'Duration: {pf(run_time, 2)}min')

        # the output for fit test
        if params.get('strategy', 'DK') == 'loukas':
            # return self.fit_better(idx, params)
            # check = False if run_time < 200 else True
            self.generalize_better(idx, params, check=True)

        elif params.get('strategy', 'DK') == 'DK':
            # params = filter_dict(params, keys=['dataset', 'method', 'ratio', 'seed', 'n_bottomk'], inclde=True)
            print(f"Parameters: {params}")
            self._get_test_error(idx, verbose=True, entry='Test')


@timefunc
def get_rawdf(db='sparsifier'):
    MU = mongo_util(db=db)
    df = pd.DataFrame(list(MU.db.runs.find(None)))
    df = df.loc[:, '_id, config, result, info, status, stop_time, start_time'.split(', ')]
    return df


@timefunc
def find_duplicate(param, db='sparsifier', start=1):
    """
    check the duplicate of tda db
    :param db:
    :param param: # param = {...,(more) 'clf': 'svm', 'epd': True, 'feat': 'sw', 'fil': 'random', 'graph': 'imdb_binary', 'n_cv': 1, 'norm': True, 'permute': False, 'ss': True}

    :return: true if there is duplicate else false
    """
    if isinstance(param, dict):
        param = set(param.values())

    configs = mongo_util(db=db).all_configs(start=start)
    for idx, d in configs.items():
        if param <= set(d.values()):
            print(f'find duplicate at idx {idx}')
            print(f'Exp {idx} configs: {d.values()}')
            print(f'param: {param}')
            return True

    print('No duplicate found')
    return False


if __name__ == '__main__':
    configs = mongo_util(db='sparsifier').all_configs(start=4000)
    print(configs[5000])

    exit()
    import pandas as pd

    df = get_rawdf()
    print(df)
    exit()
    P = parser(df)

    loukas_table = []
    for idx in \
            range(900, 914):
        # range(715, 914):
        # range(685, 715):
        # range(660, 685):
        # [625]:
        P.get_summary(idx, debug=False)
        continue

    row_dict = P.get_summary(idx=idx, debug=False)
    print(row_dict)
    if row_dict != None and len(row_dict) == 6:
        loukas_table.append(row_dict)

    # loukas_table = pd.DataFrame(loukas_table, columns = ['dataset', 'ratio', 'seed', 'method', 'loukas', 'eigenloss'])
    # print(loukas_table, )
    # summary(loukas_table)
