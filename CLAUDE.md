# CLAUDE.md

Guidance for working on this repo: a Mojo port of Tri Dao's CUDA
`causal_conv1d`. The benchmark we care about is "GPU kernel time vs
upstream Tri Dao CUDA", with upstream as the moving target.

## Repository layout

- `src/causal_conv1d_mojo/`
  - `fwd/`, `bwd_full/`, `update/`: GPU kernels (one subpackage each).
    Pure JIT-on-first-use — there is no `dispatch.mojo` and no AOT
    comptime sweep. Every subpackage has:
    - `kernel.mojo` (the device function — comptime-parameterized
      over dtype, width, has_bias, ...).
    - `common.mojo` (shared constants/helpers).
    - `launch.mojo` (`launch_<sub>[...]`: configures the
      `DeviceContext`, builds the `TileTensor` layouts, calls
      `compile_function` + `enqueue_function`, parameterised by the
      full comptime tuple).
    - `variant.mojo` (the static per-subpackage entry point. Reads
      its comptime params via `std.sys.get_defined_*` so a single
      source file covers every config — no per-variant codegen on
      disk. Exports `PyInit_variant` so the compiled `.so` is a
      loadable CPython extension).
    - `_jit.py` (Python: extracts the config tuple from the call's
      runtime args, formats a readable mod name, materialises the
      config as `-D KEY=VALUE` pairs, and delegates to the shared
      cache+compile+load helper).
    - `__init__.py` (Python wrapper that builds the args tuple and
      calls `_jit.call_<sub>(args)`).
    The shared `mojo build` → `dlopen` plumbing lives in
    `_jit_common.py` at the package root (`compile_and_load`).
    Per-variant artefacts cache under
    `$XDG_CACHE_HOME/causal_conv1d_mojo/<sub>/<backend>/<arch>/<cpu_tag>/<mod_name>.hash-<h>.so`,
    where `<backend>` is `cuda` / `rocm` / `metal`, `<arch>` is the
    GPU target (`sm89`, `gfx942`, `macos15`), `<cpu_tag>` is a short
    derivation of the host CPU brand (mojo's `-march=native` codegen
    bakes host SIMD into the `.so`'s host-side glue, so different CPUs
    must not share cache entries), and `<mod_name>` is a readable
    config string like `fp16_w4_hb0_hs0_hi0_silu0_contig1_aligned1`.
    See "Cache-key contents" below for what `<h>` covers.
  - `fwd_cpu/`, `bwd_full_cpu/`, `update_cpu/`: CPU fallbacks. Same
    JIT-on-first-use plumbing as the GPU subpackages — each (subpkg,
    config) compiles its own `.so` via `mojo build` and caches under
    `$XDG_CACHE_HOME/causal_conv1d_mojo/<sub>_cpu/cpu/<cpu_tag>/<mod_name>.hash-<h>.so`.
    No GPU `arch` subdir for CPU (obviously), but the same
    `<cpu_tag>` segment applies — host-CPU SIMD baked into the `.so`
    is the dominant factor here.
  - `_jit_common.py`: shared variant cache + compile + load helper used
    by every subpackage. Also owns the env-signature → cache-hash
    logic (see below).
  - `_fn.py`, `_update.py`, `reference.py`: Python facades + pure-PyTorch
    reference implementations.
- `tests/`: pytest suite. Run with `uv run --extra nvidia pytest` (the
  `nvidia` extra brings in upstream causal-conv1d for the reference op).
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

Always use `uv run --extra nvidia …` — the `nvidia` extra pulls in the
upstream Tri Dao causal-conv1d wheel that the benches diff against.

```bash
# Kernel-only GPU time per shape (uses torch.profiler CUPTI hooks)
uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py

# Wall-clock + plots into docs/
uv run --extra nvidia plot-bench
```

In most cases you should never need to manually clear the cache — the
hash mixes in the Python ABI tag, mojo compiler version, modular SDK
install path, ptxas version (CUDA only), and this file's own hash, so
env switches and toolchain bumps invalidate automatically. If you
suspect something stale anyway:

```bash
rm -rf ~/.cache/causal_conv1d_mojo/
```

The cache is content-addressed (`<mod_name>.hash-<h>.so`), so editing
`kernel.mojo`/`launch.mojo`/`common.mojo` also busts the cache for
every variant that depends on them on next compile.

### Production: pre-warmed cache + `CAUSAL_CONV1D_USE_CACHE_ONLY`

For containerised deploys you can pre-warm the cache on a staging
host that matches production (same Python, mojo, CPU, GPU, ptxas) by
running representative workloads, then bundle
`~/.cache/causal_conv1d_mojo/` into the production image.

In production, set `CAUSAL_CONV1D_USE_CACHE_ONLY=1`. Any cache miss
at runtime then raises `RuntimeError` instead of silently triggering
a ~1.2 s JIT compile in the request hot path. The error includes the
full env signature for the missing variant so you can see exactly
which signal diverged (CPU model, mojo version, modular path, etc.).

### Cache-key contents

`<h>` in the cached `.so` filename is sha256(…)[:16] over:
1. The full contents of the per-variant `variant.mojo` source.
2. Every `.mojo` in each `include_dirs` path (recursive but glob'd
   per dir; matches how `mojo build -I` resolves imports).
3. The `defines` dict (`-D KEY=VALUE` pairs) for that variant.
4. An env signature dict (see `_env_signature` in `_jit_common.py`)
   containing:
   - **`soabi`**: `sysconfig.get_config_var('SOABI')` — captures
     Python minor version + CPU arch + OS in one field.
   - **`mojo_version`**: `mojo --version` output, includes git hash.
   - **`modular_root`**: path to the modular SDK install — baked
     into the `.so` RUNPATH.
   - **`cpu_brand`**: full host-CPU brand string (e.g.
     `Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz`, `Apple M2 Pro`).
     Mojo's CPU codegen defaults to `-march=native`, so the produced
     `.so` contains host-specific SIMD (AVX2/AVX-512 on x86,
     NEON/SVE on ARM). Mixing CPUs in a shared cache without keying
     on this SIGILLs at first instruction. The full brand goes into
     the hash; a short tag derived from it goes into the cache
     directory path so identical-CPU hits stay clustered.
   - **`jit_common_hash`**: hash of `_jit_common.py` itself, so
     future changes to the `mojo build` invocation bust the cache.
   - **`ptxas`** (CUDA only): identifies the ptxas mojo will hand
     PTX to. Three states: `bundled` (env var unset, uses the one
     shipped with the modular SDK — subsumed by `mojo_version`),
     `cu12:<pkg-version>` (vendored `nvidia-cuda-nvcc-cu12` wheel,
     what `__init__.py` sets by default), or
     `external:<path>:<--version output>` (user-overridden).

The GPU compute capability (`sm89`, `gfx942`, …) and the host-CPU
tag are *directory* segments rather than parts of `<h>`, so one
shared cache can hold artefacts for multiple GPUs and CPUs side by
side without collision and `ls`ing the cache stays informative.

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
uv run --extra nvidia ncu --target-processes all --launch-skip 20 \
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
uv run --extra nvidia nsys profile --stats=true \
    -o /tmp/causal_conv1d \
    python benchmarks/bench_gpu_kernel_time.py
```

The summary table prints per-kernel total/avg time and call counts —
sanity-check against torch.profiler.

### 4. Apple silicon: xctrace "Metal System Trace"

There is no torch device-time hook for Metal, so the CUPTI/rocprof path
in `bench_gpu_kernel_time.py` doesn't work on Apple. The equivalent is
an Instruments "Metal System Trace" recorded around the kernel, read
back via the scriptable `metal-gpu-intervals` table.

```bash
# pre-warms the JIT cache, records a trace, prints per-encoder GPU time
scripts/xctrace_bench.sh --kind fwd --shape 1,1024,2048,4 --iters 40
scripts/xctrace_bench.sh --kind update --dtype bf16
```

`scripts/xctrace_bench.sh` drives `benchmarks/bench_metal_gpu.py` (a
mojo-only MPS workload runner — upstream causal-conv1d is CUDA-only, so
there's nothing to diff against here; the goal is precise *absolute* GPU
time) under `xctrace record --template 'Metal System Trace'`, then parses
the trace with `scripts/xctrace_gpu_intervals.py`. Our forward kernel
shows up as `Compute / Compute Command`; host<->device copies are
`Blit Command`. Notes:

- **Mojo doesn't label its Metal encoders** (`metal-object-label` is
  empty), so all compute dispatches group under one `Compute Command`
  row — fine for single-kind/single-shape runs (the row's count matches
  `iters`+warmup; its mean is the per-call GPU time). Run one shape at a
  time to keep the attribution clean.
- The export XML uses a global `id`/`ref` value dictionary; the parser
  resolves it. A row's *first* `<duration>` is the GPU time; the second
  is "CPU to GPU Latency".
- `xctrace record --launch` **intermittently crashes** (Bus/Segfault)
  while finalizing the bundle, leaving an unexportable `.trace`. The
  wrapper retries until `xctrace export` succeeds — expect a few retries
  per run; it's an Instruments bug, not ours.
- HW counters (ALU busy, bandwidth, occupancy) are **GUI-only** on Apple
  silicon — headless `xctrace export` doesn't expose them. Open the
  printed `.trace` bundle in Instruments for those.

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
5. **One cubin per (dtype × width × has_bias × has_seq_idx ×
   has_initial_states × apply_silu × contig_inner × aligned_seq) leaf,
   compiled JIT on first use.** Each leaf compiles to its own
   single-variant `.so` via `_jit_common.compile_and_load`, cached at
   `~/.cache/causal_conv1d_mojo/<sub>/<backend>/<arch>/<mod_name>.hash-<h>.so`
   (see "Cache-key contents" above). The Python-side `_jit.py`
   decides the config from runtime args and passes it as `-D
   KEY=VALUE` pairs to `mojo build`; the static `variant.mojo` reads
   the defines via `std.sys.get_defined_*` and calls
   `launch_<sub>[concrete params](...)` from `launch.mojo`. First call
   per (config, machine) pays ~1-3 s for `mojo build`; every later call
   in this or any future process hits the on-disk cache. There is no
   comptime sweep — each variant is its own translation unit.
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
   `ld.shared`, `bar.sync`) to a known-good version. Add a temporary
   `dump_asm=StaticString("/tmp/mojo_<sub>_%.ptx")` to the
   `compile_function[...]` call inside `<sub>/launch.mojo`, trigger the
   variant once (e.g. via a bench/test run), and remove the knob
   afterwards. Don't commit it.
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
- The `mojo build` cache (`~/.cache/causal_conv1d_mojo/`) bakes the
  *build env's* modular-lib path into each `.so`'s `RUNPATH`. If you
  switch uv envs the runtime loader can't find
  `libKGENCompilerRTShared.so`. Since the env signature now folds
  the modular SDK install path into the cache hash, switching envs
  auto-invalidates the affected entries — but the *files* aren't
  cleaned up. To recover disk space periodically, just nuke the
  whole cache (see the "Running the benches" section above).
- `dump_asm` paths must be `StaticString(...)`-wrapped; bare string
  literals can fail the `Variant[Bool, Path, StaticString, ...]` coerce.
- `TileTensor` has two non-obvious costs at very small kernel
  runtimes (a few microseconds total):
  1. `linear_idx_type` defaults to `DType.int64` for global-memory
     tensors with any dynamic dim, so `t[b, c, i]` lowers to
     `mul.lo.s64` (multi-op SASS) instead of `IMAD`. Passing strides
     as `UInt32` in the Layout doesn't help — Mojo widens them back
     to i64 before the multiply. Workaround: pass
     `linear_idx_type=DType.int32` explicitly.
  2. Each `TileTensor` kernarg becomes a packed `.align 8 .b8 [N]`
     blob; strides are then offsetted `ld.param.b32` loads (and for
     1-D nested layouts, register-indirect loads). Raw `.u32` stride
     kernargs are direct register loads, saving ~5-10 cycles in the
     prologue.

  For `fwd/` and `bwd_full/` (kernels that run tens to hundreds of
  μs) both costs are noise. For `update/` (decode kernel, ~2-8μs per
  call) they're measurable — that's why `update/` deliberately uses
  raw pointers + Int32 strides. See `update/kernel.mojo`'s header
  comment for the PTX-level reasoning.
