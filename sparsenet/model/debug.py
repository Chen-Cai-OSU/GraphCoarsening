# Created at 2020-05-17
# Summary: debug pytorch # https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795/4

import torch
from torch.autograd import Variable as V
from time import time
# GOAL: Do gradient descent on unrolled network that is simply:
#   y = y + w*x  -->  y(t) = y(t-1) + w*x(t)

# Random training data
from sparsenet.util.util import pf

X = V(torch.randn(100,1).cuda())
Y = V(torch.randn(100,1).cuda())

nGD = 1000        # number of gradient descent iterations
nTime = 5       # number of unrolling time steps

# Initialize things
gamma = 0.1
w = V(torch.randn(1,1).cuda(), requires_grad=True)
Yest = V((0.5*torch.ones(100,1)).cuda()) # Don't really care about values in Yest at this point, just allocating GPU memory

Yest_buffer = torch.zeros(100, 1).cuda()

for iGD in range(nGD):
    # At start of processing for each GD iteration, the
    # output estimate, Yset, should be set to an initial
    # estimate of some fixed value.  E.g., ...
    # Yest.data.zero_().add_(0.5)                # This line fails on second GD iteration.
    Yest_buffer.zero_().add_(0.5)
    Yest = V(Yest_buffer)

    # Yest = V((0.5*torch.ones(100,1)).cuda()) # This works, but I think it's allocating GPU memory every time, and thus slow.
    t0 = time()
    for iTime in range(nTime):
        Yest = Yest + w*X
    cost = torch.mean((Y - Yest)**2)
    cost.backward(retain_graph=False) # compute gradients
    w.data.sub_(gamma*w.grad.data) # Update parameters
    w.grad.data.zero_() # Reset gradients to zeros
    t1 = time()
    print(pf(t1-t0, 3))