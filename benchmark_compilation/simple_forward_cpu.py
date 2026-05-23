# ruff: noqa: E402 — imports are intentionally deferred so we can time them.
import time

t0 = time.perf_counter()
import torch

t1 = time.perf_counter()

import causal_conv1d_mojo

t2 = time.perf_counter()

dtype = torch.float32
x = torch.randn(1, 16, 1024, dtype=dtype, device="cpu")
weight = torch.randn(16, 4, dtype=dtype, device="cpu")
bias = torch.randn(16, dtype=dtype, device="cpu")
t3 = time.perf_counter()

# First call triggers `mojo build` of the CPU `dispatch.mojo` (one AOT
# comptime sweep over width × has_bias × has_seq_idx × has_initial_states
# × apply_silu × dtype). Subsequent calls hit the on-disk cache.
out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias)
t4 = time.perf_counter()

out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias)
t5 = time.perf_counter()

# Same dispatch.so; different runtime branch (apply_silu=True).
out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
t6 = time.perf_counter()

out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
t7 = time.perf_counter()

print(f"import torch:                          {(t1 - t0) * 1000:8.1f} ms")
print(f"import causal_conv1d_mojo:             {(t2 - t1) * 1000:8.1f} ms")
print(f"tensor setup:                          {(t3 - t2) * 1000:8.1f} ms")
print(f"first call  variant A (AOT compile):   {(t4 - t3) * 1000:8.1f} ms")
print(f"second call variant A (warm):          {(t5 - t4) * 1000:8.1f} ms")
print(f"first call  variant B (cached):        {(t6 - t5) * 1000:8.1f} ms")
print(f"second call variant B (warm):          {(t7 - t6) * 1000:8.1f} ms")
