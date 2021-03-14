# Created at 2020-07-16
# Summary: monitor the training process
import os

from signor.monitor.time import time_analysize
from signor.ioio.dir import sig_dir

import argparse

from sparsenet.util.util import runcmd

parser = argparse.ArgumentParser(description='Baseline for graph sparsification')
parser.add_argument('--idx', type=int,)

if __name__ == '__main__':
    args = parser.parse_args()
    idx = args.idx
    f = f'{sig_dir()}monitor/log'
    cmd = f" python sparsenet/paper/generalize.py --check --check_idx {idx} | grep '[0-9]s' > {f}"
    runcmd(cmd)
    time_analysize(f=f)
