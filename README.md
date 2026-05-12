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

Wall-clock per call, fp16 + silu + bias, 500 iters each, **min over
samples**, sync after every call (RTX 2000 Ada Generation Laptop GPU).
Lower is better; log scale.

| shape (B, D, L, W) | mojo | upstream | pure PyTorch |
| --- | ---: | ---: | ---: |
| (1, 1024, 512, 4) | **54 μs** | 53 μs | 71 μs |
| (1, 1024, 2048, 4) | 95 μs | 60 μs | 150 μs |
| (1, 1024, 8192, 4) | 203 μs | 93 μs | 552 μs |
| (1, 4096, 2048, 4) | 178 μs | 99 μs | 422 μs |
| (4, 4096, 2048, 4) | **807 μs** | 845 μs | 3273 μs |
| (8, 2048, 4096, 4) | **1541 μs** | 1580 μs | 8391 μs |

Mixed picture across the shape grid:
- Small + large shapes are competitive or win (`(4, 4096, 2048)` and
  `(8, 2048, 4096)` are both faster than upstream).
- **Mid-size shapes (`1×1024×8192`, `1×4096×2048`) are where upstream
  pulls ahead by ~2×** — these are memory-bound and upstream's
  `cub::BlockLoad<WARP_TRANSPOSE>` keeps loads coalesced; our forward
  kernel still does per-element scalar global loads with bounds checks.
  Fixing this is the open todo on the forward path.
- Pure PyTorch is **3-5× slower** across the board.

### Forward + backward

![backward](docs/bench_backward.png)

Same workload but `out.backward(dout)` is included in each timed iteration.
The mojo backward is a single fused kernel — grid `(dim, batch)`, walks the
seqlen in *reverse* via an inner chunk loop, exchanges the dout halo across
chunks via smem (mirroring upstream's `causal_conv1d_bwd_kernel`), then
runs one `block.sum` + atomic_add per `(channel, k)` at the end.

| shape (B, D, L, W) | mojo | upstream | pure PyTorch |
| --- | ---: | ---: | ---: |
| (1, 1024, 512, 4) | **304 μs** | 329 μs | 321 μs |
| (1, 1024, 2048, 4) | **310 μs** | 327 μs | 365 μs |
| (1, 1024, 8192, 4) | **490 μs** | 524 μs | 1369 μs |
| (1, 4096, 2048, 4) | **566 μs** | 596 μs | 1431 μs |
| (4, 4096, 2048, 4) | **2011 μs** | 2071 μs | 8953 μs |
| (8, 2048, 4096, 4) | **3347 μs** | 3909 μs | 20285 μs |

**Mojo wins on all 6 shapes**, including the heavy `(8, 2048, 4096)`
where it beats upstream by ~14%. Pure PyTorch is **3-6× slower**
everywhere.

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

### Single-step update (autoregressive decode)

![update](docs/bench_update.png)

`causal_conv1d_update(x, conv_state, weight, ...)` — the per-token
decode op. `x` is `(B, D)` (one new token per batch element),
`conv_state` is `(B, D, W-1)`, mutated in place. Same workload, 1000
iters, min over samples.

| shape (B, D) | mojo | upstream | pure PyTorch (`update_ref`) |
| --- | ---: | ---: | ---: |
| (1, 1024) | **40 μs** | 44 μs | 94 μs |
| (1, 2048) | **42 μs** | 44 μs | 98 μs |
| (1, 4096) | **42 μs** | 44 μs | 98 μs |
| (4, 1024) | **41 μs** | 46 μs | 108 μs |
| (4, 2048) | **45 μs** | 46 μs | 138 μs |
| (4, 4096) | **47 μs** | 47 μs | 199 μs |
| (16, 2048) | **43 μs** | 45 μs | 208 μs |
| (32, 4096) | **44 μs** | 47 μs | 419 μs |

Mojo is **on par or slightly faster than upstream** across all shapes.
Per-call cost is dominated by kernel launch overhead (~40 μs); both
implementations sit at the launch-overhead floor. Pure PyTorch is
**2-10× slower** since it builds a full conv every call instead of
running a single fused kernel.

## Layout

* `src/causal_conv1d_mojo/_native/` — Mojo source, one file per concern,
  matching upstream's `causal-conv1d/csrc/` layout:
    * `causal_conv1d_common.mojo` — shared constants + `_silu_f32`
      (mirrors `causal_conv1d_common.h`).
    * `causal_conv1d_fwd.mojo` — GPU forward kernel
      (mirrors `causal_conv1d_fwd.cu`).
    * `causal_conv1d_bwd.mojo` — GPU fused backward kernel + warp/block
      reductions (mirrors `causal_conv1d_bwd.cu`).
    * `causal_conv1d_update.mojo` — GPU single-step / KV-cache decode
      kernel (mirrors `causal_conv1d_update.cu`).
    * `causal_conv1d_fwd_cpu.mojo` / `causal_conv1d_bwd_cpu.mojo` /
      `causal_conv1d_update_cpu.mojo` — pure-mojo CPU paths (no
      upstream analogue; let the package run on a GPU-less machine).
    * `causal_conv1d_native.mojo` — dispatcher: 6 launchers (parses
      Python args, builds the `(dtype × width × flags)` comptime tree)
      + `PyInit_*` (mirrors `causal_conv1d.cpp`).
* `src/causal_conv1d_mojo/__init__.py` — Python wrapper. Wraps forward
  in a `torch.autograd.Function`; backward delegates to PyTorch autograd.
* `tests/test_native.py` — correctness for forward (incl. non-contiguous
  inputs: transposed x, sliced x, transposed weight) and backward
  (gradients vs the pytorch reference impl).
* `benchmarks/` — microbenches: kernel-time-only, wall-time forward,
  wall-time forward+backward, host-launch overhead, mojo vs upstream
  vs pure-PyTorch.

## Status / scope

Width: 2, 3, or 4 (Mamba uses 4). Inputs may be fp16, bf16, or fp32;
`x` / `weight` / `bias` must share a dtype. `bias` may be `None` or a
`(dim,)` tensor; `activation` may be `None`, `"silu"`, or `"swish"`
(silu/swish are the same op). `seq_idx` is fully supported on both
forward and backward — masks reads at packed-sequence boundaries,
padding rows (`seq_idx < 0`) produce zero output and zero gradient.
`initial_states` (a `(B, D, W-1)` historical context before `t=0` for
chunked stateful execution) is also forward + backward — `dinitial_states`
flows back from the conv's leftmost W-1 outputs. `return_final_states`
/ `final_states_out` is full forward + backward. Both paths go through
native Mojo kernels (GPU + CPU); the autograd `Function` plumbs all
flags through.

`causal_conv1d_update(x, conv_state, weight, ...)` provides the
single-step / KV-cache decode op: takes 1 (or a few) new tokens,
mutates `conv_state` in place, returns the conv output. Used in
autoregressive Mamba inference. Supports `cache_seqlens` (circular
buffer mode — per-batch write head, `state` becomes a ring) and
`conv_state_indices` (per-batch state-row indirection: decouple input
batch from cache slot for paged-cache servers; negative indices
denote padding tokens whose output is zeroed). Both can be combined.

## Run it

```sh
pixi run test               # correctness
pixi run bench-vs-pytorch   # forward wall-time numbers
pixi run bench-backward     # forward + backward wall-time numbers
pixi run plot-bench         # regenerate docs/bench_forward.png + bench_backward.png + bench_update.png
```

The Mojo source is compiled lazily on first `import causal_conv1d_mojo`
via `mojo.importer` — it runs `mojo build --emit shared-lib` and caches
the resulting `.so` under `src/causal_conv1d_mojo/_native/__mojocache__/`.
First import takes a few seconds; subsequent imports are cache hits.
No manual build step.


## Time to compile on the fluidstack node:

```
subpkg            compile (s)  .so size (MB)
--------------------------------------------
fwd                      2.29           42.3
bwd_full                 4.23          195.6
update                   1.72           22.9
fwd_cpu                  1.61           24.4
bwd_full_cpu             2.21           44.6
update_cpu               1.94           44.9
--------------------------------------------
TOTAL                   13.99          374.7
```
