# Created at 2020-05-17
# Summary: test sparsenet/model/example.py

import os
from _warnings import warn


class setup:
    def __init__(self, user='cc'):
        if user == 'cc':
            self.dir = '/home/cai.507/Documents/DeepLearning/sparsifier/sparsenet/'
            self.py = '/home/cai.507/anaconda3/envs/sparsifier/bin/python '

        elif user == 'dk':
            pass

        else:
            NotImplementedError

        self.file = os.path.join(self.dir, 'model', 'example.py')

if __name__ == '__main__':
    warn('why there is space')
    print('abc')
    print('abc')
    exit()

    s = setup(user='cc')
    cmd = f'{s.py} {s.file}'
    print(cmd)
    os.system(cmd)
