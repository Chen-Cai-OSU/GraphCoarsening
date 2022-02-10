# Created at 2021-03-13
# Summary: argparse related
from warnings import warn

import numpy as np


class argsparser():
    def __init__(self, args):
        if args.lap in ['None', 'none']:
            args.lap = None
        self.args = args

    def set_indices(self):
        args = self.args

        train_indices = [int(item) for item in args.train_indices.split(',') if len(item) != 0]
        test_indices = [int(item) for item in args.test_indices.split(',') if len(item) != 0]
        val_indices = np.random.choice(test_indices, 5, replace=False).tolist() if len(test_indices) > 10 else []
        test_indices = [idx for idx in test_indices if idx not in val_indices]

        if len(val_indices) == len(test_indices) == 0:  # for datasets with single graph
            test_indices = train_indices
            val_indices = train_indices

        # todo: better handling
        if len(val_indices) == 0:
            assert len(train_indices) > 1
            if len(train_indices) < 5:
                n_sample = 1
            else:
                n_sample = 5 if len(train_indices) < 15 else 10
            val_indices = np.random.choice(train_indices, n_sample, replace=False).tolist()
            train_indices = [idx for idx in train_indices if idx not in val_indices]

        # todo: handle this case more elegantly
        if args.dataset == 'coauthors':  # handle coauthors
            args.n_epoch = 20
            train_indices = [0]
            test_indices = [1]
            val_indices = [0]

        print(f'train_indices: {train_indices}.\n '
              f'val_indices: {val_indices}. \n '
              f'test_indices: {test_indices}.')
        self.args = args
        return train_indices, val_indices, test_indices

    def set_model_name(self):
        args = self.args
        model_name = 'checkpoint-best-eigen-ratio.pkl' if args.valeigen else 'checkpoint-best-improve-ratio.pkl'
        return model_name
