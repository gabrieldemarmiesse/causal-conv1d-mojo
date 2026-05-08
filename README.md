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
| (1, 1024, 512, 4) | 58 μs | 58 μs | 77 μs |
| (1, 1024, 2048, 4) | 93 μs | 75 μs | 163 μs |
| (1, 1024, 8192, 4) | 267 μs | 110 μs | 876 μs |
| (1, 4096, 2048, 4) | 259 μs | 115 μs | 764 μs |
| (4, 4096, 2048, 4) | 1270 μs | 932 μs | 3425 μs |
| (8, 2048, 4096, 4) | 2086 μs | 1719 μs | 9561 μs |

On heavy shapes this is **3–5× faster than pure PyTorch** and ~1.2× of
upstream's hand-tuned CUDA. On light shapes upstream's launch wins because
its `causal_conv1d_fwd_kernel` is tighter than the one we currently
generate; closing that gap is one of the open todos.

### Forward + backward

![backward](docs/bench_backward.png)

Same workload but `out.backward(dout)` is included in each timed iteration.
The mojo backward delegates to PyTorch autograd (re-runs `F.conv1d + F.silu`
inside the backward and lets autograd differentiate it), which is correct
and reasonably fast but pays for the recomputed forward.

| shape (B, D, L, W) | mojo | upstream | pure PyTorch |
| --- | ---: | ---: | ---: |
| (1, 1024, 512, 4) | 492 μs | 435 μs | 441 μs |
| (1, 1024, 2048, 4) | 560 μs | 459 μs | 426 μs |
| (1, 1024, 8192, 4) | 2542 μs | 620 μs | 2086 μs |
| (1, 4096, 2048, 4) | 2276 μs | 651 μs | 1918 μs |
| (4, 4096, 2048, 4) | 11232 μs | 2234 μs | 9158 μs |
| (8, 2048, 4096, 4) | 26280 μs | 3659 μs | 21009 μs |

Upstream's hand-tuned `causal_conv1d_bwd_kernel` is currently a clear
win for backward; ours sits between pure PyTorch and upstream. A custom
Mojo backward kernel is the obvious next step if backward latency matters.

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
