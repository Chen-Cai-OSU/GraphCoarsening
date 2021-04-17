# Created at 2020-05-06
# Summary: utils for sparsenet.

import numpy as np
import torch

from sparsenet.model.eval import trainer, tester  # train, set_train_data
from sparsenet.util.util import banner, red, pf


class ModelEvaluator():
    def __init__(self, model, dataset_loader, dev, optimizer):
        self.dev = dev
        self.optimizer = optimizer
        self.model = model
        self.dataset_loader = dataset_loader

    def set_modelpath(self, path):
        self.modelpath = path

    def train(self, idx, TR, model, args):
        """ train the model for one graph """
        TR.set_train_data(args, self.dataset_loader)
        TR.train(model, self.optimizer, args, verbose=False)
        TR.delete_train_data(idx)
        return model

    def validate(self, idx, val_indices, TE, model, args):
        val_score = self.val_score
        val_score[idx] = {'n_gen': [], 'impr_ratio': [], 'eigen_ratio': []}
        for idx_ in val_indices:
            args.test_idx = idx_
            args.cur_idx = idx_
            TE.set_test_data(args, self.dataset_loader)
            n_gen, impr_ratio, eigen_ratio = TE.eval(model, args, verbose=False)

            val_score[idx]['n_gen'].append(n_gen)
            val_score[idx]['impr_ratio'].append(impr_ratio)
            val_score[idx]['eigen_ratio'].append(eigen_ratio)

        banner(f'{args.dataset}: finish validating graph {val_indices}.')

        cur_impr_ratio = np.mean(val_score[idx]['impr_ratio'])
        cur_eigen_ratio = np.mean(val_score[idx]['eigen_ratio'])
        print(cur_eigen_ratio, self.best_eigen_ratio)
        self.val_score[idx] = val_score[idx]
        return cur_impr_ratio, cur_eigen_ratio

    def save(self, idx, model, mode='eigen-ratio'):
        """ save model for training graph idx """
        assert mode in ['eigen-ratio', 'improve-ratio']
        f = f'checkpoint-best-{mode}.pkl'

        if mode == 'eigen-ratio':
            torch.save(model.state_dict(), self.modelpath + f)
            print(red(f'Save model for train idx {idx}. Best-eigen-ratio is {pf(self.best_eigen_ratio, 2)}.'))
        elif model == 'improve-ratio':
            torch.save(model.state_dict(), self.modelpath + f)
            print(red(f'Save model for train idx {idx}. Best-improve-ratio is {pf(self.best_impr_ratio, 2)}.'))

    def find_best_model(self, model, train_indices, val_indices, args):
        """ save the best model on validation dataset """

        self.TR = trainer(dev=self.dev)
        self.TE = tester(dev=self.dev)

        self.val_score = {}
        self.best_n_gen = -1e10
        self.best_impr_ratio = -1e30
        self.best_eigen_ratio = -1e30
        self.train_indices = train_indices
        self.val_indices = val_indices

        for idx in self.train_indices:
            args.train_idx = idx
            args.cur_idx = idx

            model = self.train(idx, self.TR, model, args)
            cur_impr_ratio, cur_eigen_ratio = self.validate(idx, val_indices, self.TE, model, args)

            # save the model if it works well on val data
            if cur_eigen_ratio > self.best_eigen_ratio:
                self.best_eigen_ratio = cur_eigen_ratio
                self.save(idx, model, mode='eigen-ratio')

            if cur_impr_ratio > self.best_impr_ratio:
                self.best_impr_ratio = cur_impr_ratio
                self.save(idx, model, mode='improve-ratio')
        return model, args

    def test_model(self, model, test_indices, AP, args):
        model_name = AP.set_model_name()

        model.load_state_dict(torch.load(self.modelpath + model_name))

        for idx_ in test_indices:
            args.test_idx = idx_
            args.cur_idx = idx_
            self.TE.set_test_data(args, self.dataset_loader)
            self.TE.eval(model, args, verbose=False)
            banner(f'{args.dataset}: finish testing graph {idx_}.')


if __name__ == '__main__':
    pass
