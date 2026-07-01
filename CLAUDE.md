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
- `scripts/`: all benchmark drivers, perf gates, and dev tooling live
  here. Scripts meant to be invoked directly have plain names; internal
  helpers (only imported or spawned by another script) are prefixed with
  `_`.
  - `_bench.py`: **the unified driver** (internal — `master_bench.py` is
    the entrypoint everyone should use; this is spawned by it, and is the
    `_`-prefixed single-shape measurement primitive you can still invoke
    directly for ad-hoc debugging). One CLI for every function
    (`fwd`/`bwd`/`update`), every input shape, every function-argument
    flag (`--bias`/`--seq-idx`/`--initial-states`/`--cache-seqlens`/…),
    against every impl (`--impl mojo,upstream,pytorch`), measured three
    independent ways (`--measure kernel|walltime|raw`):
    `kernel` = per-kernel GPU time via `torch.profiler`; `walltime` =
    end-to-end via `torch.utils.benchmark` (auto cpu↔gpu sync); `raw` =
    a bare synced loop for an external profiler (ncu) to wrap. Reports
    min + spread over `--runs` N≥3; `--json` for tooling. Memoizes the
    stable upstream/pytorch baselines in `scripts/baselines/`
    (gitignored; keyed on baseline version + shape + config + GPU +
    clock-lock state). `--device auto` picks cuda → mps → cpu; on Apple
    (`--device mps`) `--measure kernel` self-orchestrates an Instruments
    "Metal System Trace" (see "Apple silicon" below).
  - `plot_bench.py`: wall-clock end-to-end plots into `docs/bench_*.png`
    (uses `_baseline.py`'s JSON cache).
  - `master_bench.py`: **the autonomous, backend-agnostic perf gate** —
    a stdlib-only orchestrator that auto-detects the backend (cuda / rocm /
    metal / cpu) and runs the same a–h phase skeleton everywhere, skipping
    phases whose tooling doesn't exist on a backend (see "The master bench"
    below).
  - `strip_publish_deps.py`: CI helper (run by `.github/workflows/
    publish.yml`) that strips dev-only deps before publishing the wheel.
  - `_baseline.py`: internal JSON baseline cache used by `plot_bench.py`.
  - `_asm_tools.py`: PTX→SASS, ptxas `-v` spill canary, upstream-SASS
    extraction (`cuobjdump` on the cubin), upstream-PTX compilation
    (`nvcc -ptx` on the `.cu`, since the shipped `.so` is cubin-only —
    `nvcc` comes from `pixi exec --spec cuda-nvcc=12.8`, no PyPI wheel
    ships the driver), and side-by-side instruction-mix histograms
    (SASS *and* PTX level), using the `ptxas`/`nvdisasm`/`cuobjdump`
    shipped inside the `triton` wheel. Spawned by `master_bench.py`'s
    NVIDIA asm phase; not called directly.
  - `_apple_gpu_clock_lock.py`: forces the Apple GPU's DVFS clock to
    Maximum for xctrace recordings by binary-patching a copy of
    Instruments' `Metal System Trace.tracetemplate` (see "Apple silicon:
    forcing the GPU clock" below). Spawned by `master_bench.py`'s metal
    lock-clocks phase; not called directly.
  - `tools/`: small shell wrappers for ad-hoc dev loops (`bench`,
    `dump_isa`, `quick_test`, `rocprof_*`, the GPU `flock` wrapper, …)
    plus `check_wheel/` (containerised wheel smoke-test). See
    `scripts/tools/README.md`.
  - `assembly/nvidia/` (ours) and `reference_assembly/nvidia/`
    (upstream): PTX/SASS regenerated by the master bench. **Gitignored**
    (machine/toolchain-specific) — regenerate locally with the bench.
- `causal-conv1d/`: vendored Tri Dao CUDA source (read-only reference
  for kernel patterns).
- `modular/`: vendored `modular/modular` repo (Mojo + MAX), used as a
  reference for Mojo syntax/APIs (`compile_function`, `stack_allocation`,
  `barrier`, `TileTensor.load[width=, alignment=]`, etc.).

## Running the benches

The primary entrypoint is `scripts/master_bench.py` (see "The master bench"
below) — it auto-detects the backend and orchestrates the full a–h gate.
The examples below drive the internal `_bench.py` primitive directly, which
is handy for a single shape during a tight dev loop.

Always use `uv run --extra nvidia …` — the `nvidia` extra pulls in the
upstream Tri Dao causal-conv1d wheel that the benches diff against.

```bash
# Per-kernel GPU time, mojo vs upstream, one shape (fast inner loop)
uv run --extra nvidia python scripts/_bench.py fwd --shape 1,4096,2048,4 --impl all

# End-to-end wall-clock (torch.utils.benchmark, auto sync)
uv run --extra nvidia python scripts/_bench.py update --shape 16,2048 --measure walltime

# Wall-clock + plots into docs/
uv run --extra nvidia python scripts/plot_bench.py
```

### The master bench (backend-agnostic perf gate)

`scripts/master_bench.py` is the one autonomous, non-interactive gate to
run after a kernel edit (passwordless `sudo -n` only — never prompts). It
auto-detects the backend (NVIDIA via `nvidia-smi`, AMD via
`rocminfo`/`rocm-smi`, Apple via `sys.platform`, else CPU; override with
`--backend`) and runs the same phase skeleton on each, dispatching every
phase to that backend's tooling and **skipping cleanly where it doesn't
exist**:

- **(a) lock clocks** — cuda: `nvidia-smi`; rocm: `rocm-smi --setperflevel
  high`; metal: forces the GPU's Induced Performance State to Maximum via
  `scripts/_apple_gpu_clock_lock.py` (see "Apple silicon: forcing the GPU
  clock" below); cpu: no GPU clock, skipped.
- **(b) recompile + correctness** — clears *our* JIT cache, runs the quick
  smoke / `--full` regression suite under the backend's `uv` extra and
  device (`-k mps/cuda/cpu`). `--skip-correctness` runs the perf phases
  only (e.g. to profile a WIP kernel).
- **(c) kernel-time bench** — cuda: vs Tri Dao upstream with min+spread and
  a 3% stop-criterion (a true perf *gate*); rocm/cpu: vs the pure-PyTorch
  fallback (reported, not gated — no hand-tuned baseline exists); metal:
  *absolute* per-kernel GPU time read back from a `xctrace` Metal System
  Trace (mojo-only — upstream is CUDA-only).
- **(d) deep profiler** — cuda: `ncu` (ephemeral via `pixi exec`); metal:
  the per-encoder GPU time + clock split + duty cycle already parsed from
  the step-(c) trace; cpu: `perf stat`; rocm: skipped (`rocprofv3` can't
  instrument Mojo's `DeviceContext`).
- **(e) dump GPU asm** — cuda: PTX/SASS to `scripts/assembly/nvidia/`;
  rocm: the AMDGPU ISA to `scripts/assembly/rocm/`; metal: skipped (Mojo
  emits no textual Metal ISA — it lowers straight to a `metallib`); cpu:
  skipped.
- **(f) instruction-mix histogram** vs `scripts/reference_assembly/nvidia/`,
  at **both SASS level** (ours vs the upstream cubin) **and PTX level**
  (ours vs upstream `.cu` compiled with `nvcc -ptx` via `pixi` — a
  higher-level diff than SASS), and **(g) `ptxas -v` spill canary** —
  NVIDIA only (no counterpart elsewhere; skipped).
- **(h)** independent `torch.utils.benchmark` wall-clock run.

Steps c/d/h are deliberately separate processes.

```bash
python scripts/master_bench.py                 # QUICK tier, auto-detect (every edit)
python scripts/master_bench.py --full --fn all # FULL gate, all functions
python scripts/master_bench.py --backend cpu   # force a backend
python scripts/master_bench.py --skip-correctness   # perf phases only
python scripts/master_bench.py --refresh-reference  # re-extract upstream asm (nvidia)
```

**Dumping our kernel's PTX** is non-invasive: set
`CAUSAL_CONV1D_DUMP_ASM=<dir>` and run any kernel call — `_jit_common`
adds `-D DUMP_ASSEMBLY_INTO=<dir>/<subpkg>__<mod>.ptx` to the `mojo build`;
`variant.mojo` reads that define via `get_defined_string` and passes it to
`compile_function`'s `dump_asm=` arg, which writes the PTX at runtime (the
dump build is its own cache entry, so the perf build is unaffected). For a
one-off, set `DUMP_ASSEMBLY_INTO=<file>` directly (a literal path used
verbatim; takes precedence over the dir form). We dump only PTX, never SASS:
the stdlib's `_dump_sass`/`_ptxas_info_verbose` shell out to a hard-coded
`/usr/local/cuda/bin/{ptxas,nvdisasm}` and raise if missing (and
`MODULAR_NVPTX_COMPILER_PATH` redirects only the compiler's ptxas, not those).
SASS, the spill report, and the SASS histogram are all derived from that PTX
by `scripts/_asm_tools.py` using the triton wheel's portable `ptxas`/`nvdisasm`.

**Upstream reference at PTX level.** The shipped upstream `.so` is cubin-only
(`cuobjdump --dump-ptx` finds nothing), so the SASS histogram diffs against
SASS decoded from the cubin. For a *higher-level* diff, the master bench also
clones Tri Dao's source (pinned `UPSTREAM_REF`, cached) and compiles the
matching `.cu` with `nvcc -ptx` (`nvcc` via `pixi exec --spec cuda-nvcc=12.8`
— no PyPI wheel ships the driver), extracts the kernel matching `REF_MATCH[fn]`
into `scripts/reference_assembly/nvidia/<fn>.ptx`, and prints a second,
PTX-level histogram (ours vs upstream source). `_to_sass`/`nvcc` for *our*
kernel isn't possible here, so only PTX is dumped on our side; the comparison
is our-PTX vs upstream-PTX and our-SASS vs upstream-cubin-SASS.

`scripts/assembly/nvidia/` (ours) and `scripts/reference_assembly/nvidia/`
(upstream `.sass` + `.ptx`) are **gitignored** — both are regenerated by the
master bench on demand (the reference is rebuilt with `--refresh-reference`,
which re-clones + recompiles, or whenever it's missing), so they're not worth
tracking and never go stale against the toolchain.

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

Cheapest, no extra perms needed. `_bench.py --measure kernel` does this:
runs N iters under `torch.profiler`, walks `prof.events()` and sums
`evt.self_device_time_total` for the kernels attributed to each impl.
This gives **per-kernel GPU time** including only the kernel's actual
execution. Use this as the primary perf signal.

Quirk: the kernel name on the GPU side is whatever the Mojo build
emits (e.g. `kernel_fwd_kernel_DType_Int6A6AcB6A6AsA6A6A_<hash>`). The
classifiers in `_bench.py` (`_mojo_classifier`/`_upstream_classifier`)
match on substring `fwd_kernel` and the upstream `void
causal_conv1d_fwd_kernel` prefix — update them if the Mojo build naming
changes. The pure-pytorch impl has no single named kernel, so it sums
every CUDA event in the profiled region.

### 2. NSight Compute (`ncu`)

Gives the deepest metrics (memory throughput, occupancy, stall
reasons, bank conflicts, etc). Needs the kernel to actually run, and
on shared hosts often needs the `--target-processes all` flag plus the
right perf-counter permission. The master bench runs this automatically
(step d, ephemerally via `pixi exec --spec nsight-compute -- ncu`);
to drive it by hand, wrap `_bench.py --measure raw` (no profiler in-proc):

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
    python scripts/_bench.py fwd --shape 1,4096,2048,4 \
      --impl mojo --measure raw --iters 30 --warmup 10
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
    python scripts/_bench.py fwd --shape 1,4096,2048,4 --impl mojo --measure raw
```

The summary table prints per-kernel total/avg time and call counts —
sanity-check against torch.profiler.

### 4. Apple silicon: `_bench.py --device mps --measure kernel`

There is no torch device-time hook for Metal, so the CUPTI/rocprof path
in `_bench.py --measure kernel` can't read per-kernel time in-process on
Apple. Instead `_bench.py` **orchestrates Instruments itself**: it
pre-warms the JIT cache, records an "Metal System Trace" with `xctrace`
around a re-launch of *itself* as the traced `--measure raw` workload,
then parses the scriptable `metal-gpu-intervals` table back out and
prints per-encoder GPU time split by GPU clock state — all in one
command, findings straight to stdout. (This folds in what used to be the
separate `bench_metal_gpu.py` + `scripts/xctrace_bench.sh` +
`scripts/xctrace_gpu_intervals.py` trio.)

```bash
# pre-warms, records a trace, prints per-encoder GPU time + clock split
uv run python scripts/_bench.py fwd --device mps --shape 1,1024,2048,4
uv run python scripts/_bench.py update --device mps --dtype bf16
# on a mac with no cuda, --device defaults to auto -> mps, so just:
uv run python scripts/_bench.py fwd --shape 1,1024,2048,4
```

The mojo-only nature is intentional: upstream causal-conv1d is CUDA-only,
so there's nothing to diff against on Apple — the goal is precise
*absolute* GPU time. Our forward kernel shows up as
`Compute / Compute Command`; host<->device copies are `Blit Command`.
Implementation notes (all in `_bench.py`):

- The traced child is a re-launch of `_bench.py` with `--device mps
  --measure raw` and `CAUSAL_CONV1D_BENCH_TRACED=1` set, so it runs the
  bare loop (bracketed by `torch.mps.profiler.profile`) instead of
  recursively orchestrating another trace. `--measure walltime`/`raw`
  run in-process on mps just like cuda (via `torch.mps.synchronize`).
- **Mojo doesn't label its Metal encoders** (`metal-object-label` is
  empty), so all compute dispatches group under one `Compute Command`
  row — fine for single-kind/single-shape runs (the row's count matches
  `iters`; its median is the per-call GPU time). Run one shape at a time
  to keep the attribution clean. For `bwd`, `_bwd_callable` builds the
  autograd graph once and re-runs only `torch.autograd.grad`, so the
  traced Compute encoder attributes to the bwd kernel rather than
  blending in the fwd pass.
- The export XML uses a global `id`/`ref` value dictionary; the parser
  (`_xml_resolver`) resolves it. A row's *first* `<duration>` is the GPU
  time; the second is "CPU to GPU Latency".
- `xctrace record --launch` **intermittently crashes** (Bus/Segfault)
  while finalizing the bundle, leaving an unexportable `.trace`.
  `_record_trace` retries until `xctrace export` succeeds — expect a few
  retries per run; it's an Instruments bug, not ours.
- **Watch out for DVFS** when running `_bench.py` standalone (outside
  `master_bench.py`, which locks the clock — see below). The Mojo Metal
  launch syncs after *every* call, so Apple's GPU governor drops the clock
  to its minimum between dispatches; short kernels are frequently measured
  at a reduced clock, which is the main source of run-to-run variance (a
  kernel can read ~1.1 ms at Maximum clock and ~2.2 ms at Minimum in the
  same run). Unlocked, `_bench.py` reads `gpu-performance-state-intervals`
  and splits the per-encoder summary by GPU clock state — **trust the
  `Maximum`-clock row** as the steady-state time (the reported headline
  kernel time picks it); the rest is throttled noise. Each group's reported
  number is the *median* (robust to the bimodal DVFS distribution).

### Apple silicon: forcing the GPU clock

Unlike `nvidia-smi --lock-gpu-clocks`/`rocm-smi --setperflevel`, macOS has
no public API or CLI to pin the GPU's DVFS clock — confirmed by checking
`xcrun devicectl` (no `condition` subcommand; only sees attached
iOS/iPadOS devices, never the local Mac), Xcode's Devices & Simulators
"Device Conditions → GPU Performance State" (real, but iOS-device-only and
GUI-only), and `powermetrics` (read-only).

Instruments does have an internal "Induced GPU Performance State" knob on
its Metal System Trace / GPU Counters instrument, normally only reachable
by hand in the GUI (configure a recording, force the state, File > Save as
Template — the saved template can then be replayed headlessly via
`xctrace record --template <path>`). `scripts/_apple_gpu_clock_lock.py`
reproduces that GUI step programmatically: it binary-patches a copy of
Instruments' own `Metal System Trace.tracetemplate` (an NSKeyedArchiver
binary plist) to force `gpuperformancestate` — an undocumented, empirically
determined enum (0=Automatic, 1=Minimum, 2=Medium, 3=Maximum, verified by
recording a real GPU workload at each value and reading back
`gpu-performance-state-intervals`: value 3 held the GPU at 100% Maximum
clock across multiple independent recordings, vs. ~96% with the rest split
across Minimum/Medium when unlocked).

The patch changes ~2 bytes of the ~50 KB template (a single
`objectRefSize`-wide object-table reference, repointed at an
already-existing sibling object holding the target int) and leaves
everything else byte-identical to Apple's original — re-serializing the
whole archive via `plistlib.dump(fmt=FMT_BINARY)` instead produces a
`plutil`-valid file that `xctrace export` nonetheless rejects with
"Document Missing Template Error", so the surgical byte patch is required,
not just "load and mutate with plistlib". The patched template is cached
under `$XDG_CACHE_HOME/causal_conv1d_mojo/xctrace_templates/`,
content-addressed on the source template + patcher script + target state.

`master_bench.py`'s step (a) calls this on the metal backend, sets
`CAUSAL_CONV1D_XCTRACE_TEMPLATE` to the patched path, and `_bench.py`'s
`_record_trace` uses it instead of the plain `"Metal System Trace"` name
for every xctrace recording that run makes — eliminating the DVFS confound
at the source rather than filtering it out after the fact. This pokes at
an undocumented private format with no cross-version stability guarantee;
`_apple_gpu_clock_lock.py` wraps every step so a structural mismatch (e.g.
a future Xcode update) degrades to returning `None`, and `master_bench.py`
falls back to the pre-existing unlocked + post-hoc-clock-bucketing
behavior rather than failing the run. `_bench.py` run standalone (not via
`master_bench.py`) stays unlocked by default, matching how the nvidia/rocm
locks are also `master_bench.py`-exclusive; set
`CAUSAL_CONV1D_XCTRACE_TEMPLATE` yourself (e.g. to the output of
`python scripts/_apple_gpu_clock_lock.py Maximum`) to opt in manually.

### What you can and can't get headlessly

Confirmed by inspecting the exported tables and testing the "Game
Performance" / "Metal GPU Counters" templates:

- **Available headless** (in the Metal System Trace export): per-encoder
  GPU time (`metal-gpu-intervals`), GPU **clock/performance state** over
  time (`gpu-performance-state-intervals`, used for the clock split), the
  GPU **Active vs Idle duty cycle** (`metal-gpu-state-intervals` — surfaced
  by `_bench.py` as the "GPU duty cycle" line; low active % == the workload
  is launch/sync-bound, the single most actionable headless signal),
  command-buffer timings, residency-set events, `device-thermal-state-
  intervals`. `powermetrics --samplers gpu_power` (needs sudo) additionally
  gives aggregate GPU active residency + frequency.
- **GUI-only**: the rich per-shader counters — **occupancy %, ALU active
  %, memory throughput, stall reasons, registers/thread** (the
  `gpu-counter-value` / `gpu-shader-profiler-sample` tables). Verified:
  even forcing `xctrace record --instrument 'Metal GPU Counters'` records
  with `Counter Set: (null)` by default, and selecting a profile makes the
  GPU service reject it ("counter profile not supported on target device")
  so those tables export **empty**. The counter set must be configured in
  the Instruments GUI; once authored, a saved `.tracetemplate` *file* with
  "Induced GPU Performance State = Maximum" + a supported counter set can
  be replayed headlessly via `xctrace record --template <path>` (there is
  no CLI/env flag for either knob). So: per-kernel *time*, *clock*, and
  *duty cycle* are scriptable; *why* a kernel is slow (occupancy/stalls)
  needs the GUI, or a one-time GUI-authored template.

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

1. Run `python scripts/master_bench.py` (or `_bench.py --measure
   kernel` for one shape) and compare ratios per shape. Wall-clock
   benches are noisy until shapes are large.
2. If the small-shape ratio gets worse but large-shape ratio is fine →
   launch overhead or low-occupancy regime. Check
   `launch__waves_per_multiprocessor` with `ncu`.
3. If all shapes regress → check the PTX for the relevant variant.
   Compare the instruction-mix histogram against the upstream
   reference (the master bench prints it, step f). To dump PTX/SASS by
   hand: `CAUSAL_CONV1D_DUMP_ASM=$PWD/scripts/assembly/nvidia uv run --extra
   nvidia python scripts/_bench.py <fn> --shape <S> --impl mojo
   --measure raw --iters 1 --warmup 0 --runs 1`, then
   `scripts/_asm_tools.py sass|spill|histogram …`. No need to edit
   `launch.mojo` — the dump is a comptime define added by `_jit_common`.
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
