# Created at 2020-11-17
# Summary: parse train log for ICLR rebuttal
from sparsenet.util.util import pf


def train_q_of_interest(line):
    """
    from line (str) extract different losses
    line = 'INFO:root:            Graph-random_er: 6. Test(0.7):  0.397(0.015) / 0.413(0.015) / 26.6. Generalize!. Eigenloss: 0.035'
    """
    entries = line.split('/')
    assert len(entries) == 3, f'Expect entries of length 3. But get {entries}'
    assert 'Train' in entries[0]

    graph_id = entries[0].split('Idx')[1].split('-')[0]
    loss1 = entries[0].split(':')[-1].rstrip(' ').lstrip(' ').split('(')[0]  # loss with learning
    graph_id, loss1 = float(graph_id), float(loss1)
    return graph_id, loss1

import numpy as np
if __name__ == '__main__':
    for loss in np.random.random(10):
        for g in [3,4]:
            # line = f'INFO:root:            Graph-ws: {g}. Test-Val(0.3):  {loss}(0.268) / 0.519(0.432) / 1.0. Generalize!. Eigenloss: -1.0. Bl_Eigenloss: -1.0.'
            line = f'INFO:root:ws-Idx {g}-Epoch: 20. Train(0.5): {loss}(0.28) / 3.364(1.416) / 1.078. Eigenloss: -1.0. Bl_Eigenloss: 2.594'
            ret = train_q_of_interest(line, stat='train_loss')
            print(ret)