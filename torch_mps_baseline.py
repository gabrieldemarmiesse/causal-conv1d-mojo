import time

import torch

assert torch.backends.mps.is_available()

x = torch.zeros(1, device="mps")
y = torch.zeros(1, device="mps")
torch.mps.synchronize()

# warmup
for _ in range(500):
    x.add_(y)
torch.mps.synchronize()

ITERS = 5000
t0 = time.perf_counter_ns()
for _ in range(ITERS):
    x.add_(y)
torch.mps.synchronize()
us = (time.perf_counter_ns() - t0) / 1000.0 / ITERS
print(f"torch.add_ on MPS (1-elem):      {us:7.3f}  μs/call")
