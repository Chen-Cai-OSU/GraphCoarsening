# Created at 2020-05-29
# Summary:  sacred


from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver
from signor.ioio.dir import mktemp

from signor.utils.cli import runcmd, capture

client = MongoClient('localhost', 27017)
EXPERIMENT_NAME = 'sparsifier'
YOUR_CPU = None
DATABASE_NAME = 'sparsifier'

ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(MongoObserver.create(url=YOUR_CPU, db_name=DATABASE_NAME))

python = '~/anaconda3/envs/sparsifier/bin/python -W ignore '
fname = 'sparsenet/model/example.py '


@ex.config
def get_config():
    bs = 512
    n_epoch = 30
    train_indices = "0,1,2," # py sparsenet/test/main.py with 'train_indices="1,2,4,5,7"' # important: needs to be very careful
    test_indices = "3,4,5,"
    dataset = 'shape'
    n_bottomk = 50
    n_layer = 3
    ratio = 0.5
    n_cycle = 1
    device = 'cuda'
    seed = 0
    strategy = 'DK'
    method = ''
    lap= 'none'
    lr = 1e-3
    loss = 'quadratic'
    w_len = 5000
    offset = 0

    ini = False
    viz=False
    loukas_quality = False
    correction = False
    valeigen = False
    dynamic = False
    mlp=False
    emb_dim = 50


@ex.capture
def run(bs, n_epoch, dataset, n_bottomk, n_layer, ratio, n_cycle, train_indices, test_indices,
        device, seed, strategy, method, lap, lr, loss, w_len, offset, ini, viz,
        loukas_quality, correction, valeigen, dynamic, mlp, emb_dim):
    f = mktemp(tmp_dir=False)


    constant_args = '--force_pos ' # --loukas_quality --viz
    if ini==True: constant_args += '--ini '
    if viz==True: constant_args += '--viz '
    if dynamic == True: constant_args += '--dynamic '
    if loukas_quality == True: constant_args += '--loukas_quality '
    if correction == True: constant_args += '--correction '
    if valeigen == True: constant_args += '--valeigen '
    if mlp==True: constant_args += '--mlp '

    file = f'{fname} {constant_args} ' \
        f' --bs {bs} --n_epoch {n_epoch}   ' \
        f' --dataset {dataset} --n_bottomk {n_bottomk} --ratio {ratio} --loss {loss} ' \
        f'--train_indices {train_indices} --test_indices {test_indices} --n_layer {n_layer} --w_len {w_len} --offset {offset}' \
        f' --n_cycle {n_cycle} --device {device} --strategy {strategy} --method {method} --emb_dim {emb_dim} ' \
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