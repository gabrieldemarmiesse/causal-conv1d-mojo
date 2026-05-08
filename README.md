# causal-conv1d-mojo

A from-scratch [Mojo](https://www.modular.com/mojo) GPU kernel for the
depthwise causal 1-D convolution used by SSMs (Mamba, RWKV-style models),
called from Python without going through the MAX framework — `mojo build
--emit shared-lib` produces a CPython extension that PyTorch can call
directly.

The reference is Tri Dao's [`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d)
hand-tuned CUDA kernel; we benchmark against that and against a pure-PyTorch
`F.conv1d(groups=D) + F.silu` fallback.

## Performance

### Forward

![forward](docs/bench_forward.png)

Wall-clock per call, fp16 + silu + bias, 500 iters each, sync after every
call (RTX 2000 Ada Generation Laptop GPU). Lower is better; log scale.

| shape (B, D, L, W) | mojo | upstream | pure PyTorch |
| --- | ---: | ---: | ---: |
| (1, 1024, 512, 4) | 56 μs | 79 μs | 99 μs |
| (1, 1024, 2048, 4) | 87 μs | 65 μs | 191 μs |
| (1, 1024, 8192, 4) | 230 μs | 105 μs | 705 μs |
| (1, 4096, 2048, 4) | 242 μs | 112 μs | 695 μs |
| (4, 4096, 2048, 4) | 1139 μs | 922 μs | 3497 μs |
| (8, 2048, 4096, 4) | 2364 μs | 1637 μs | 9079 μs |

On heavy shapes this is **3–5× faster than pure PyTorch** and ~1.3-1.6× of
upstream's hand-tuned CUDA. On light shapes mojo is competitive or wins
(the smaller launch overhead from running directly through a CPython
extension helps). Mid-size shapes (`1×1024×8192`, `1×4096×2048`) are where
upstream's `cub::BlockLoad<WARP_TRANSPOSE>` pulls ahead — the forward
kernel still does per-element scalar global loads with bounds checks.

### Forward + backward

![backward](docs/bench_backward.png)

Same workload but `out.backward(dout)` is included in each timed iteration.
The mojo backward is a single fused kernel — grid `(dim, batch)`, walks the
seqlen in *reverse* via an inner chunk loop, exchanges the dout halo across
chunks via smem (mirroring upstream's `causal_conv1d_bwd_kernel`), then
runs one `block.sum` + atomic_add per `(channel, k)` at the end.

| shape (B, D, L, W) | mojo | upstream | pure PyTorch |
| --- | ---: | ---: | ---: |
| (1, 1024, 512, 4) | **409 μs** | 461 μs | 511 μs |
| (1, 1024, 2048, 4) | **431 μs** | 449 μs | 534 μs |
| (1, 1024, 8192, 4) | **582 μs** | 612 μs | 1780 μs |
| (1, 4096, 2048, 4) | **611 μs** | 652 μs | 1747 μs |
| (4, 4096, 2048, 4) | 2308 μs | 2144 μs | 9355 μs |
| (8, 2048, 4096, 4) | 4560 μs | 3903 μs | 22634 μs |

Mojo is now within 5% of upstream on the median shape (medium ones it
wins outright), and **3–5× faster than pure PyTorch** everywhere. The
last 15-30% on the heaviest shapes is upstream's hand-tuned cub-based
warp-transpose load + register-only halo exchange, which we don't yet
match.

The big surprise during this work was that `Atomic.fetch_add` defaults to
`Consistency.SEQUENTIAL` + system scope, lowering to
`ATOMG.E.ADD.STRONG.SYS` — a CPU-fenced atomic that drains L2 on every
call. Switching to `Atomic[scope="device"].fetch_add[ordering=
Consistency.MONOTONIC]` (matching CUDA's `atomicAdd`, which lowers to
`RED.E.ADD.STRONG.GPU`) cut backward kernel time from 14.4 ms to 1.3 ms
on the heavy shape — most of the gap to upstream lived in 5
seq-cst-system atomics in the kernel epilogue, not in the chunk loop.

For an extensive cross-shape sweep see `benchmarks/bench_backward_extensive.py`
and `benchmarks/bench_forward_extensive.py`.

## Layout

* `src/causal_conv1d_mojo/_native/causal_conv1d_native.mojo` — the GPU
  forward kernel + CPython extension entry point.
* `src/causal_conv1d_mojo/__init__.py` — Python wrapper. Wraps forward
  in a `torch.autograd.Function`; backward delegates to PyTorch autograd.
* `tests/test_native.py` — correctness for forward (incl. non-contiguous
  inputs: transposed x, sliced x, transposed weight) and backward
  (gradients vs the pytorch reference impl).
* `benchmarks/` — microbenches: kernel-time-only, wall-time forward,
  wall-time forward+backward, host-launch overhead, mojo vs upstream
  vs pure-PyTorch.

## Status / scope

Specialized for the Mamba forward path: fp16 inputs, `width=4`,
`has_bias=True`, `activation="silu"`, no `initial_states`, no
`return_final_states`. Anything outside that raises
`NotImplementedError` from the public `causal_conv1d_fn` wrapper.
Forward is end-to-end Mojo; backward goes through `torch.autograd`.

## Run it

```sh
pixi run test               # correctness
pixi run bench-vs-pytorch   # forward wall-time numbers
pixi run bench-backward     # forward + backward wall-time numbers
pixi run plot-bench         # regenerate docs/bench_forward.png + bench_backward.png
```

The Mojo source is compiled lazily on first `import causal_conv1d_mojo`
via `mojo.importer` — it runs `mojo build --emit shared-lib` and caches
the resulting `.so` under `src/causal_conv1d_mojo/_native/__mojocache__/`.
First import takes a few seconds; subsequent imports are cache hits.
No manual build step.
