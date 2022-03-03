# Created at 2020-06-21
# Summary: test sacred configuration


from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver
from signor.ioio.dir import mktemp

from signor.utils.cli import runcmd, capture

from sparsenet.util.db_util import find_duplicate
from sparsenet.util.dir_util import PYTHON

client = MongoClient('localhost', 27017)
EXPERIMENT_NAME = 'test'
YOUR_CPU = None
DATABASE_NAME = 'test'

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))

python = f'{PYTHON} -W ignore '
fname = 'sparsenet/model/example.py ' # f'{sig_dir()}graph/paper/PairNorm/main.py'


@ex.config
def get_config():
    bs = 512
    n_epoch = 30
    train_indices = "0,1,2,"
    test_indices = "3,4,5,"
    dataset = 'shape'
    n_bottomk = 50
    n_layer = 3
    ratio = 0.5
    n_cycle = 1
    device = 'cuda'
    seed = 0
    strategy = 'DK'
    method = 'heavy_edge'
    lap= 'none'
    lr = 1e-3
    loss = 'quadratic'
    ini = False
    viz=False
    mlp=False


@ex.capture
def run(bs, n_epoch, dataset, n_bottomk, n_layer, ratio, n_cycle, train_indices, test_indices,
        device, seed, strategy, method, lap, lr, loss, ini, viz, mlp):
    f = mktemp(tmp_dir=False)
    constant_args = '--force_pos ' # --loukas_quality --viz
    if ini==True: constant_args += '--ini '
    if viz == True: constant_args += '--viz '
    if mlp == True: constant_args += '--mlp '
    print(constant_args)
    params_set = [bs, n_epoch, dataset, n_bottomk, n_layer, ratio, n_cycle, train_indices, test_indices,
        device, seed, strategy, method, lap, lr, loss, ini, viz]
    params_set = set(params_set)
    print(params_set)

    if find_duplicate(params_set, db='test', start=2):
        return

    # if viz=='T': constant_args += '--viz '

    file = f'{fname} {constant_args} ' \
        f' --bs {bs} --n_epoch {n_epoch}   ' \
        f' --dataset {dataset} --n_bottomk {n_bottomk} --ratio {ratio} --loss {loss} ' \
        f'--train_indices {train_indices} --test_indices {test_indices} --n_layer {n_layer} ' \
        f' --n_cycle {n_cycle} --device {device} --strategy {strategy} --method {method} ' \
        f'--seed {seed} --lap {lap} --lr {lr} > {f} 2>&1' # redirect everything to a file

    cmd = ' '.join([python, file])
    runcmd(cmd)
    code, out, err  = capture(['cat', f])
    return out.decode('utf-8') # code, out, err


@ex.main
def main(_run):
    out = run()
    return out # code, out, err


if __name__ == '__main__':
    ex.run_commandline()