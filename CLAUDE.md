# CLAUDE.md

Guidance for working on this repo: a Mojo port of Tri Dao's CUDA
`causal_conv1d`. The benchmark we care about is "GPU kernel time vs
upstream Tri Dao CUDA", with upstream as the moving target.

## Repository layout

- `src/causal_conv1d_mojo/`
  - `fwd/`, `bwd_full/`, `update/`: GPU kernels (one subpackage each).
    Every subpackage has `kernel.mojo` (the device function),
    `dispatch.mojo` (the Python/CPython entry, comptime dispatch tree,
    `compile_function` + `enqueue_function` calls), `common.mojo`
    (shared constants/helpers), and `__init__.py` (Python wrapper that
    lazy-imports `dispatch`). `mojo.importer` compiles each subpackage
    to a `.so` on first import; built artefacts cache under
    `<subpkg>/__mojocache__/`.
  - `fwd_cpu/`, `bwd_full_cpu/`, `update_cpu/`: CPU fallbacks. Same
    layout, used by tests as a portable reference.
  - `_fn.py`, `_update.py`, `reference.py`: Python facades + pure-PyTorch
    reference implementations.
- `tests/`: pytest suite. Run with `pixi run -e bench pytest` (the
  `bench` env brings in upstream causal-conv1d for the reference op).
- `benchmarks/`
  - `bench_gpu_kernel_time.py`: **kernel-only** GPU time via
    `torch.profiler`. Use this when iterating on kernel perf.
  - `plot_bench.py`: wall-clock end-to-end (includes Python + launch
    overhead). Produces the `docs/bench_*.png` plots.
  - `bench_vs_pytorch.py`, `bench_forward_extensive.py`, etc.:
    additional wall-clock benches.
- `causal-conv1d/`: vendored Tri Dao CUDA source (read-only reference
  for kernel patterns).
- `modular/`: vendored `modular/modular` repo (Mojo + MAX), used as a
  reference for Mojo syntax/APIs (`compile_function`, `stack_allocation`,
  `barrier`, `TileTensor.load[width=, alignment=]`, etc.).

## Running the benches

Always use the `bench` pixi env — it has the upstream Tri Dao package.

```bash
# Kernel-only GPU time per shape (uses torch.profiler CUPTI hooks)
pixi run -e bench python benchmarks/bench_gpu_kernel_time.py

# Wall-clock + plots into docs/
pixi run -e bench plot-bench
```

If you change `.mojo` source between envs, clear the cache (cached `.so`s
have the *build env's* lib path baked into RUNPATH):

```bash
find src -name __mojocache__ -type d -exec rm -rf {} +
```

## Measuring kernel performance properly

Wall-clock `time.perf_counter_ns()` around a kernel launch is dominated
by Python + cudaLaunchKernel overhead at small shapes — useless for
optimising the kernel itself. Use one of the following.

### 1. torch.profiler (CUPTI traces)

Cheapest, no extra perms needed. `bench_gpu_kernel_time.py` already
does this: wraps each impl in a `record_function` range, runs N iters,
walks `prof.events()` and sums `evt.self_device_time_total` per kernel.
This gives **per-kernel GPU time** including only the kernel's actual
execution. Use this as the primary perf signal.

Quirk: the kernel name on the GPU side is whatever the Mojo build
emits (e.g. `kernel_fwd_kernel_DType_Int6A6AcB6A6AsA6A6A_<hash>`). The
classifier `_kind(name)` in `bench_gpu_kernel_time.py` matches on
substring `fwd_kernel` and the upstream `void causal_conv1d_fwd_kernel`
prefix — update it if the Mojo build naming changes.

### 2. NSight Compute (`ncu`)

Gives the deepest metrics (memory throughput, occupancy, stall
reasons, bank conflicts, etc). Needs the kernel to actually run, and
on shared hosts often needs the `--target-processes all` flag plus the
right perf-counter permission.

```bash
# Single-shape, single-kernel runs — keep ITERS small (ncu serializes).
pixi run -e bench ncu --target-processes all --launch-skip 20 \
    --launch-count 5 \
    --metrics "sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__inst_executed.avg.per_cycle_active,\
gpu__time_duration.avg,\
launch__waves_per_multiprocessor,\
smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
dram__bytes.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_second,\
smsp__inst_executed_op_shared_st.sum,\
smsp__inst_executed_op_shared_ld.sum" \
    python benchmarks/bench_gpu_kernel_time.py
```

Common metrics to chase:

- `gpu__time_duration.avg`: per-kernel time. The ground truth.
- `launch__waves_per_multiprocessor`: <1 means the grid doesn't fill
  the GPU — small-shape regime.
- `smsp__warps_issue_stalled_*`: stall reasons. `barrier` stalls →
  smem-dance dominates; `long_scoreboard` → memory-bound.
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: count of L1 LD
  sectors. A redundant-load problem looks like ~4× upstream here.
- `smsp__inst_executed_op_shared_st.sum`, `_ld.sum`: shared-mem
  instruction count.

### 3. NSight Systems (`nsys`)

Use when you suspect a *launch* problem (kernels too small to amortise
launch overhead, missing concurrency, host stalls) rather than an
intra-kernel problem.

```bash
pixi run -e bench nsys profile --stats=true \
    -o /tmp/causal_conv1d \
    python benchmarks/bench_gpu_kernel_time.py
```

The summary table prints per-kernel total/avg time and call counts —
sanity-check against torch.profiler.

## Inspecting generated code (PTX, SASS)

The Mojo `DeviceContext.compile_function` accepts:

```mojo
ctx.compile_function[
    fwd_kernel[ ... ],
    fwd_kernel[ ... ],
    dump_asm=StaticString("/tmp/mojo_fwd_%.ptx"),   # PTX (one per variant)
    _dump_sass=StaticString("/tmp/mojo_fwd_%.sass"),# SASS (needs nvdisasm)
    _ptxas_info_verbose=True,                       # ptxas -v output (occupancy/regs)
]()
```

`%` in the path is replaced with the *module name* of that comptime
variant — so when the dispatcher emits N specialised variants, you get
N separate files. Tips:

- The `StaticString(...)` wrap is required: the `dump_asm` arg is a
  `Variant[Bool, Path, StaticString, def() capturing -> Path]` and bare
  string literals don't always coerce.
- `_dump_sass` shells out to `/usr/local/cuda/bin/nvdisasm`; install
  the CUDA toolkit if it's missing.
- Don't leave `dump_asm` on in committed code — it triggers on every
  `mojo build`, polluting `/tmp` with hundreds of files (one per
  comptime variant).

PTX features to look for when diff-ing against upstream:

- `ld.global.nc.v4.b32` (LDG.E.128): 16-byte invariant vec load.
  Missing this for fp16 → you're loading 2 bytes at a time. Fix by
  setting `kNElts = 16 // size_of[dtype]()` and using
  `tile_tensor.load[width=kNElts, alignment=16](Coord(...))`.
- `st.shared.b16` × 8 vs `st.shared.v4.b32` × 1: smem-store vectorisation.
  ptxas usually merges adjacent stores at SASS level, but at PTX level
  the `comptime for i in range(...)` pattern reads as separate stores.
  Worth checking the SASS if you suspect this is the bottleneck.
- `bar.sync`: barriers. 2 per chunk iteration is the minimum for the
  smem ring-buffer pattern (write halo / read halo / late-write carry).
- `CALL.REL.NOINC`: outlined helper calls. ptxas sometimes outlines
  `shfl.sync.bfly` into `__cuda_sm70_shflsync_bfly`; force-inline the
  intrinsic at the leaf to keep SASS flat (see `bwd_full/kernel.mojo`
  `_shfl_xor_f32`).

## Kernel-design patterns that mattered here

These were the wins on the fwd kernel rewrite (took it from 2-3× upstream
on H100 fp16 to ~1.0-1.3× on the same shapes):

1. **Grid = (dim, batch), not (chunks, dim, batch).** One block per
   (B, D); the block walks the seqlen in a chunk loop. The original
   design had each block do one chunk, so the same (B,D) reloaded the
   weights/bias N times and re-read boundary elements from global. The
   new design loads weight+bias once and shares boundary x values via
   smem.
2. **16-byte LDG.** Per thread, load `kNElts = 16 // sizeof(dtype)`
   elements as a single vec instruction — 8 for fp16/bf16, 4 for fp32.
   Set this from the dtype via `size_of[dtype]()` so all three dtypes
   pick the right width. The vec load needs `alignment=16` on the
   `TileTensor.load[]` call (the comptime inner-stride=1 promise is
   already in the Layout for `contig_inner` mode).
3. **Smem ring-buffer for the (W-1) halo.** Each thread shares its
   *last (W-1) x values* with the next thread via shared memory; the
   slot at `kNThreads-1` doubles as the inter-chunk carry. Three
   barriers per chunk: write halo, read halo, late-write the new carry
   (the third write is gated to thread `kNThreads-1` only so thread 0's
   halo read still sees the *previous chunk's* tail in the same slot).
4. **`aligned_seq` comptime gate.** When `seqlen % (kNThreads*kNElts) ==
   0`, drop the bounds-checked tail-chunk path entirely. Halves the
   compiled kernel size and avoids the predicated stores ptxas can't
   merge.
5. **Comptime dispatch tree, runtime selector.** All
   (dtype × width × has_bias × has_seq_idx × has_initial_states ×
   apply_silu × contig_inner × aligned_seq) leaves compile to their own
   cubin embedded in `.rodata`; the dispatcher walks the tree and picks
   one. The mutually-exclusive `seq_idx` + `initial_states` combo is
   `comptime if`-filtered out of the tree so we don't waste a cubin on
   it. `product` from `std.itertools` takes 2/3/4 iterables; for ≥5,
   nest the loops.
6. **`Atomic[dtype, scope="device"].fetch_add[ordering=RELAXED]`** in
   the bwd's reduce step. Default atomics on Mojo lower to
   `ATOMG.E.ADD.F32.STRONG.SYS` (system-scope, sequentially consistent
   — drains L2, sync with CPU), which added ~750ns/block on bwd. GPU-
   scope relaxed atomics are what CUDA's `atomicAdd` does.

## Where to look first when perf regresses

1. Run `bench_gpu_kernel_time.py` (kernel-only time) and compare ratios
   per shape. Wall-clock benches are noisy until shapes are large.
2. If the small-shape ratio gets worse but large-shape ratio is fine →
   launch overhead or low-occupancy regime. Check
   `launch__waves_per_multiprocessor` with `ncu`.
3. If all shapes regress → check the PTX for the relevant variant.
   Compare instruction counts (`ld.global`, `st.global`, `st.shared`,
   `ld.shared`, `bar.sync`) to a known-good version. The dispatcher's
   `dump_asm="/tmp/mojo_fwd_%.ptx"` knob is the fast way to capture
   them; remember to disable it after.
4. The vendored Tri Dao source at `causal-conv1d/csrc/` is the
   reference for every algorithmic choice (chunk size, smem layout,
   gating order). When in doubt, mirror it.

## Mojo gotchas hit while porting

- `DType` has no `.size_of()` method; use the free function
  `from std.sys import size_of` and call `size_of[dtype]()`.
- `stack_allocation[count, dtype, address_space=AddressSpace.SHARED]()`
  returns an `UnsafePointer` with **no** `.offset()` method. Use
  `ptr + i` for offsets.
- `comptime for x, y, ... in product(...)` only handles up to 4
  iterables. Nest loops or call `product` recursively.
- The `mojo build` cache (`__mojocache__/`) bakes the *build env's*
  modular-lib path into the `.so`'s `RUNPATH`. If you switch pixi envs
  the runtime loader can't find `libKGENCompilerRTShared.so`. Clear
  the cache when changing envs.
- `dump_asm` paths must be `StaticString(...)`-wrapped; bare string
  literals can fail the `Variant[Bool, Path, StaticString, ...]` coerce.

## Conventions

- Tests live under `tests/` and use the pure-PyTorch
  `causal_conv1d_ref` / `causal_conv1d_update_ref` as ground truth.
  When changing a kernel, run `pixi run -e bench pytest` (1600+ tests,
  ~20s). Don't skip it.
- The `ref` Python implementations are the *spec*; do not change them
  to chase a perf bug — fix the kernel.
- Don't commit `dump_asm=...` in `compile_function` calls. It's a
  debug-only knob.
