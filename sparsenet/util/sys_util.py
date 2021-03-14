# Created at 2020-06-21
# Summary: set thread number

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

# status=None