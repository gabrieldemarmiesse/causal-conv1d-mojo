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
      over input dtype, weight dtype, width, has_bias, ...).
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
    config string like
    `fp16_wfp32_w4_hb0_hs0_hi0_silu0_contig1_chunk161_vec161_cl0_extstr1`
    (input fp16, weight/bias fp32).
    See "Cache-key contents" below for what `<h>` covers.
  - `fwd_cpu/`, `bwd_full_cpu/`, `update_cpu/`: CPU kernels. Same
    JIT-on-first-use plumbing as the GPU subpackages — each (subpkg,
    config) compiles its own `.so` via `mojo build` and caches under
    `$XDG_CACHE_HOME/causal_conv1d_mojo/<sub>_cpu/cpu/<cpu_tag>/<mod_name>.hash-<h>.so`.
    No GPU `arch` subdir for CPU (obviously), but the same
    `<cpu_tag>` segment applies — host-CPU SIMD baked into the `.so`
    is the dominant factor here. No `launch.mojo` (nothing to launch)
    and no `common.mojo`: the kernels take raw pointers + element
    strides straight from `variant.mojo` (no TileTensor — the hot
    loops index rows manually so they can issue unaligned SIMD
    loads/stores along t). See "CPU kernel design" below for the
    vectorization/parallelism pattern and the fwd↔update bit-exactness
    contract.
  - `_jit_common.py`: shared variant cache + compile + load helper used
    by every subpackage. Also owns the env-signature → cache-hash
    logic (see below).
  - `_fn.py`, `_update.py`, `reference.py`: Python facades + pure-PyTorch
    reference implementations.
  - `causal_conv1d_varlen.py`: packed-batch trailing-state extraction.
    The primary function is a fully vectorized PyTorch gather (including
    moving CPU `cu_seqlens` to x's device), with no per-sequence host sync;
    `_ref` deliberately keeps the obvious Python loop for cross-checking.
    Both return `(batch, dim, state_len)` with dim contiguous
    (`stride(1) == 1`), ready for `causal_conv1d_update`'s `conv_state`.
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
  high`; metal: forces Induced GPU Performance State to Maximum via
  `scripts/_apple_gpu_clock_lock.py` (see "Apple silicon: forcing the GPU
  clock" below); cpu: best-effort — Linux sets the cpufreq governor to
  `performance` via `sudo -n` (restored on exit) and macOS has no
  userspace CPU-DVFS control at all, so a failed cpu lock warns and
  continues (the cpu ratchet tolerance is sized for that noise). A
  **hard gate** on cuda/rocm/metal: a failed lock exits non-zero instead
  of continuing unlocked (unlocked numbers aren't comparable across runs,
  which defeats an agentic perf loop). `--no-lock` opts out for local dev.
- **(b) recompile + correctness** — clears *our* JIT cache, runs the quick
  smoke / `--full` regression suite under the backend's `uv` extra and
  device (`-k mps/cuda/cpu`). `--skip-correctness` runs the perf phases
  only (e.g. to profile a WIP kernel).
- **(c) kernel-time bench** — cuda: vs Tri Dao upstream with min+spread and
  a 3% stop-criterion (a true perf *gate*); rocm: vs the pure-PyTorch
  fallback (reported, not gated — no hand-tuned baseline exists); metal:
  *absolute* per-kernel GPU time read back from a `xctrace` Metal System
  Trace (mojo-only — upstream is CUDA-only), **gated two ways**: a
  one-call output canary (all-zero/non-finite result fails — timing
  can't see a silently-lost dispatch) and a ratcheting regression gate
  vs this machine's own best per-shape median, persisted in
  `scripts/baselines/metal_kernel_gpu_time.json` (gitignored;
  >10% over baseline fails; faster runs lower the baseline;
  `--refresh-baseline` reseeds after an intentional change). On metal
  `--runs N` to `_bench.py` means N whole xctrace recordings (each
  contributing its foreign-encoder-filtered headline median as one run);
  the master bench records 1 in quick tier / 3 in `--full`. The
  metal *walltime* step (h) is tagged `unlocked` — the clock lock is an
  xctrace template, so it never applies outside recordings. cpu: per-call
  CPU kernel time (`_bench.py --measure kernel` on cpu times the
  synchronous call directly — median over iters; there are no device
  events to attribute) vs the pure-PyTorch fallback; the pytorch ratio is
  informational (not a tuned baseline), and the gate is the **same
  canary + ratchet pair as metal**, persisted in
  `scripts/baselines/cpu_kernel_time.json` (gitignored; >25% over this
  machine's best fails — looser than metal's 10% since there is no cpu
  clock lock on darwin; faster runs ratchet down; `--refresh-baseline`
  reseeds). Every backend also prints a **memory-roofline verdict** per
  shape (bytes moved, achieved GB/s, % of the device's peak DRAM
  bandwidth, theoretical floor time, and a regime + one-line hint) — the
  "is this number good?" an agent needs to decide whether a shape has
  headroom (`memory-bound (near-peak)` → move on; `dispatch/latency-bound`
  → amortize launch). Peak BW is keyed per device with a
  `CAUSAL_CONV1D_PEAK_GBPS` override and *printed*, so the % is never a
  black box; an unknown device omits the % rather than inventing one. SKU
  variants matter — the table distinguishes e.g. `H100 PCIe` (2.0 TB/s)
  from SXM (3.35 TB/s); longest device-name substring wins. A shape whose
  working set fits in L2/SLC can legitimately measure *above* DRAM peak —
  that reports as `memory-bound (cache-resident)` (>110% of peak) rather
  than pretending DRAM is the roofline. On Apple the same table serves
  device=cpu (unified memory: the SoC DRAM peak bounds the CPU cluster
  too; the cpu env-sig gpu string carries the brand — `cpu:Apple M4 x10`
  — so it matches); x86 CPUs have no entries and rely on the env
  override. On metal the roofline % is only trusted
  at Maximum clock — a throttled/unknown-clock run is marked `throttled`
  and the regression gate treats it as **inconclusive** (warn, don't fail,
  don't touch the baseline) so thermal noise can't spuriously fail CI.
  Shapes suffixed `+cl` in `SHAPES` run with **channel-last** x/out
  (`--channel-last`): the quick tier gates the canonical shape in both
  layouts, and the FULL tier adds three channel-last shapes, so the
  dedicated forward and backward channel-last kernels are gated against
  upstream's own channel-last CUDA kernels.
- **(d) deep profiler** — cuda: `ncu` (ephemeral via `pixi exec`); metal:
  the per-encoder GPU time + clock split + duty cycle already parsed from
  the step-(c) trace; cpu: `perf stat` on Linux, `/usr/bin/sample` on
  darwin (spawns the raw bench loop sized off the step-(c) time, samples
  it mid-loop, prints the top-of-stack leaf counts — where the time goes:
  the Mojo kernel vs Python marshalling vs the parallel runtime; full
  call tree saved under `scripts/baselines/sample_<fn>_cpu.txt`); rocm:
  skipped (`rocprofv3` can't instrument Mojo's `DeviceContext`).
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

The `summary` section ends with a machine-readable **`===AGENT-SUMMARY===`**
block (one JSON object: backend, device, tier, gate pass/fail, and a
per-shape list of `{fn, shape, dtype, channel_last, kernel_us,
achieved_gbps, pct_peak, regime, hint, ratio_over_upstream,
ratio_over_pytorch}` — the last is how rocm/cpu surface their fallback
ratio). An agent driving a perf loop can
slice that out between the delimiters instead of scraping the human
tables — it's the fastest way to see, per shape, whether there's headroom
and what to try next.

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

### Deterministic backward: `CAUSAL_CONV1D_DETERMINISTIC`

Backward selects its reduction mode on every call, matching upstream.
`CAUSAL_CONV1D_DETERMINISTIC=1` forces deterministic reduction and
`CAUSAL_CONV1D_DETERMINISTIC=0` forces the default atomic reduction;
an unset or other value follows
`torch.are_deterministic_algorithms_enabled()`. The explicit env value
therefore overrides PyTorch's process-wide setting in either direction.

The `DETERMINISTIC` comptime flag is part of both GPU and CPU bwd JIT
keys (`..._det0` / `..._det1`). Default variants keep the shared fp32
`(D, W)` / `(D,)` accumulators and relaxed float atomics. Deterministic
variants instead plain-store every row/block's partial into its unique
batch-major fp32 `(B, D, W)` / `(B, D)` workspace slot; Python reduces
those workspaces with `.sum(0)` in a fixed order before casting the
gradients back to the parameter dtype. The workspaces are zero-filled
so zero-batch/seqlen early returns also reduce to exact zeros. Peak
temporary memory is `4 * B * D * (W + has_bias)` bytes.

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

On `--device cpu` there are no device events (and the Mojo CPU kernel
isn't a torch op, so torch.profiler couldn't attribute it anyway):
`--measure kernel` instead times the synchronous kernel call per-call
and reports the median over `--iters` (robust to scheduler blips). It
includes the Python argument marshalling — an inherent per-call cost of
the CPU path — and attaches the same all-zero/non-finite output canary
the metal path has (`"canary"` in the JSON report; master_bench gates
on it). The bwd callable on cpu builds the autograd graph once and
re-runs only `torch.autograd.grad`, so the "bwd" number is backward-only
(same trick as mps, different reason: no name-based classifier to
isolate the bwd kernel from a fused fwd+bwd call).

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

macOS has no public API/CLI to pin the GPU's DVFS clock. Instruments does
have a GUI-only "Induced GPU Performance State" knob on its Metal System
Trace instrument; `scripts/_apple_gpu_clock_lock.py` reproduces it by
binary-patching a copy of `Metal System Trace.tracetemplate` (an
NSKeyedArchiver binary plist) to force `gpuperformancestate` — an
undocumented, empirically determined enum (0=Automatic, 1=Minimum,
2=Medium, 3=Maximum; verified via `gpu-performance-state-intervals`: value
3 held 100% Maximum clock across multiple recordings). Must patch the raw
binary plist directly and in place — re-serializing via
`plistlib.dump(fmt=FMT_BINARY)` produces a file `xctrace export` rejects
with "Document Missing Template Error". Patched templates are cached under
`$XDG_CACHE_HOME/causal_conv1d_mojo/xctrace_templates/`.

`master_bench.py`'s step (a) calls this on the metal backend and points
`_bench.py`'s xctrace calls (`CAUSAL_CONV1D_XCTRACE_TEMPLATE` env var) at
the patched template — same hard-gate policy as nvidia/rocm (see above).
`_bench.py` run standalone stays unlocked by default and falls back to the
post-hoc clock-state bucketing described above; set
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
2. **16-byte LDG, only after proving every row base.** Per thread, the
   aligned variants load `kNElts = 16 // sizeof(dtype)` elements as one
   vec instruction — 8 for fp16/bf16, 4 for fp32. Inner stride 1 is not
   enough: the dispatcher also requires `data_ptr % 16 == 0` and both
   `(batch_stride * sizeof(dtype)) % 16 == 0` and
   `(channel_stride * sizeof(dtype)) % 16 == 0` for every sequence tensor
   vector-accessed (fwd: x/out; bwd: x/dout/dx). A contiguous `(B,D,L)`
   tensor with odd L, or a sequence slice starting at an odd element, can
   violate that promise. Those calls compile the non-vector specialization,
   whose whole-row slice loads/stores use `alignment=size_of[dtype]()`;
   this mirrors upstream's scalar `BLOCK_LOAD_WARP_TRANSPOSE` path. Shared-
   memory vectors remain `alignment=16` because their allocation and slot
   spacing guarantee it.
3. **Smem ring-buffer for the (W-1) halo.** Each thread shares its
   *last (W-1) x values* with the next thread via shared memory; the
   slot at `kNThreads-1` doubles as the inter-chunk carry. Three
   barriers per chunk: write halo, read halo, late-write the new carry
   (the third write is gated to thread `kNThreads-1` only so thread 0's
   halo read still sees the *previous chunk's* tail in the same slot).
4. **`aligned_seq` / `vec_aligned` comptime gates.** `aligned_seq` now
   means both `seqlen % (kNThreads*kNElts) == 0` *and* the row-alignment
   proof above; `vec_aligned` is the weaker `seqlen % kNElts == 0` with
   the same proof. The former drops the bounds-checked tail-chunk path;
   the latter retains a partial chunk but knows each thread's slice is
   wholly in or out of range. Readable cache names surface these as
   `chunk16{0,1}` and `vec16{0,1}`. Backward's 8-element fp16/bf16
   `n_elts` choice is likewise allowed only after x/dout/dx pass the
   16-byte row proof; otherwise it uses the 4-element generic variant.
5. **One cubin per (dtype × wdtype × width × has_bias × has_seq_idx ×
   has_initial_states × apply_silu × contig_inner × aligned_seq ×
   vec_aligned) leaf,
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
6. **Bwd reduction mode is comptime-specialised.** The default
   `DETERMINISTIC=false` leaf uses
   `Atomic[dtype, scope="device"].fetch_add[ordering=RELAXED]` (or
   agent-scope on AMD). Default Mojo atomics lower to
   `ATOMG.E.ADD.F32.STRONG.SYS` (system-scope, sequentially consistent
   — drains L2, sync with CPU), which added ~750ns/block on bwd; GPU-
   scope relaxed atomics are what CUDA's `atomicAdd` does. In the
   current Mojo toolchain fp32 `fetch_add` itself expands to a CAS retry
   loop; that is negligible for generic bwd's one flush per `(B,D)`, but
   disastrous for channel-last bwd's flush per `(B,L-chunk,D)`, so the
   channel-last NVIDIA leaf uses inline PTX
   `atom.relaxed.gpu.global.add.f32` instead (HIP/Metal keep the
   portable relaxed atomic). The `DETERMINISTIC=true` leaf has no
   across-batch atomics: block `(b,d)` plain-stores its reduced fp32
   values into `(B,D,W)` / `(B,D)` workspaces for Python's fixed-order
   `.sum(0)`. Because that workspace scheme is generic-kernel-only,
   `_config_from_args` clears `channel_last` whenever `deterministic` is
   set, so a packed channel-last backward in deterministic mode falls
   back to the generic kernel.
7. **Channel-last fwd kernel (`fwd_channellast_kernel`), all backends.**
   When x/out have dim contiguous (`x.stride(1)==1`, the layout a
   `(B, L, D)`-contiguous activation gets after `.transpose(1, 2)` —
   upstream's `is_channel_last`), `fwd/_jit.py` dispatches a dedicated
   kernel instead of the generic strided (fully scalar) fallback: one
   thread owns `kNElts` consecutive channels (one 16-byte vector) and
   walks its block's seqlen rows, carrying the (W-1)-row halo in
   registers; loads/stores stay fully coalesced along dim. Three things
   were needed to reach upstream parity on H100 (12.9 µs vs upstream's
   12.9 µs at `(1,4096,2048,4)` fp16; was 423 µs on the scalar
   fallback, 26 µs for the naive port):
   - **Coalesced smem staging of the weight tile.** Per-thread scalar
     weight reads (`w[(c0+i)*stride + j]`, 64-byte warp stride) cost
     ~8× upstream's *total* L1 load-sector count on their own
     (`l1tex__t_sectors_..._op_ld.sum` is the tell). The block now
     cooperatively copies its `(kChunkC × W)` tile through smem
     (tap-major so read-back is a conflict-free vector per tap), and
     bias is one vector load. Each thread still owns
     `kNElts(dtype)` channels, so the weight/bias vector spans
     `kNElts(dtype) * sizeof(wdtype)` bytes: 8, 16, or 32 across the
     supported pairs.
   - **Row-walk unrolled ×4** (`kUnroll`): 4 independent x-row loads
     in flight per warp before any is consumed. The kernel is
     register-capped at ~20 warps/SM (84 regs/thread), so one
     outstanding load per warp left it `long_scoreboard`-stalled;
     4× the MLP per warp closed the last 27% gap by itself.
   - **Backend-aware rows-per-block target** (`kMinBlocksCL`): 64,
     halved down to 4 while total blocks < 1024 on CUDA/HIP
     (discrete GPUs need thousands of resident warps; halo re-reads
     mostly hit L2) vs down to 8 while < 512 on Metal. On the bench
     shape rows=16 beats rows=8 (halo traffic) and rows=32 (occupancy).
   Also 3.9× over the scalar fallback on M4 (1327→337 µs, same shape),
   within ~8% of the seqlen-contiguous kernel there. seq_idx still
   takes the generic path (upstream is the opposite: their seq_idx
   *requires* channel-last). The dispatch gate requires 16B-aligned
   x/out base pointers, a bias pointer aligned to its
   `kNElts * sizeof(wdtype)` vector width (8/16/32 bytes depending on the
   dtype pair), and x/out batch + seqlen strides divisible
   by `kNElts` elements. Because `kNElts = 16 / sizeof(dtype)`, that
   element-stride condition is exactly a 16-byte stride for fp16, bf16,
   and fp32; together with `c0` advancing by `kNElts`, it covers every
   x/out vector row. Weight staging is scalar from global then vectorized
   only in guaranteed-aligned shared memory; `initial_states` is scalar,
   so arbitrary valid outer strides remain safe. The gate also requires
   `width <= 5` — the
   unrolled walk carries the halo in `kUnroll = 4` registers, so wider
   fp16/bf16 filters (we support up to 9) stay on the generic kernel.
   within ~8% of the seqlen-contiguous kernel there. Packed seq_idx now
   uses this path: a `(W-1)` row-id halo travels beside the x register
   halo, and the four fresh ids are loaded with the four fresh x rows.
   Every lane requests the same id address, so those reads are warp
   broadcasts rather than per-channel memory traffic. On H100 this is
   12.9 µs vs upstream's 23.9 µs at `(1,4096,2048,4)` fp16 (1.08× our
   own 11.9 µs non-seq_idx time); the former generic route was 208 µs.
   The dispatch gate also requires 16B-aligned x/out/bias base pointers
   (conforming strides alone don't guarantee alignment for the 16-byte
   vector accesses) and `width <= 5` — the unrolled walk carries the
   halo in `kUnroll = 4` registers, so wider fp16/bf16 filters (we
   support up to 9) stay on the generic kernel.
   Grid is `(L-chunks, batch, C-chunks)`: L-chunks is the only axis
   that can blow past CUDA's 65535 grid.y/z cap (seqlen 4.2M+), so it
   rides grid.x. Known gap: one tiny latency-bound shape
   `(1,128,2048,4)` sits at ~1.25× upstream (3.5 vs 2.8 µs) — smaller
   rows, direct weight loads, and pre-barrier halo hoisting were all
   tried and regressed other shapes (see the Mojo gotchas below).
8. **Packed sequences plus initial states (upstream v1.7.0 parity).**
   The virtual `W-1` positions before `t=0` carry `seq_idx[b, 0]`, so
   the initial state is visible only to positions with the first packed
   sequence's id. This matters when that first fragment is shorter than
   `W-1`: the following sequence must not read the remaining state taps.
   Forward assigns that id to the negative-time seq window; backward
   applies the same gate to silu' recomputation, initial-state dweight
   terms, and `dinitial_states`. Padding ids (`seq_idx < 0`) still force
   output and `dpre` to zero. `seq_idx + return_final_states` remains
   unsupported. Channel-last seq_idx inputs are handled directly by the
   dedicated forward kernel (item 7) and backward kernel (item 10).
9. **State buffers and update vectors.** Generic fwd/bwd access
   `initial_states` / `dinitial_states` one scalar at a time, so they do
   not participate in the 16-byte dispatch proof. Update's x, conv_state,
   and out token-loop accesses are also scalar and honor their element
   strides, which makes offset columns such as `big[:, :, 5]` safe. Its
   weight vector is enabled only when width stride is 1 and the base plus
   channel stride preserve `sizeof(wdtype) * width` alignment; otherwise
   the taps are scalar-loaded with `w_w_stride`. The short linear-state
   vector load is used only for `state_l_stride == 1`; strided conv-state
   views take scalar history loads.
   unsupported. Eligible channel-last forward inputs use the dedicated
   kernel above; contiguous, wide-filter, dim-tail, or unaligned cases
   retain the generic strided GPU fallback. Backward has its own
   channel-last kernel (item 10).
10. **Channel-last bwd kernel (`bwd_channellast_kernel`), all backends.**
   This follows upstream's shared-memory transpose rather than forward's
   register-only row walk. A 128-thread block loads x/dout as 16-byte
   vectors along contiguous C into padded `(L,C)` shared-memory tiles,
   remaps threads so each owns one channel over a 32/64-row run, keeps
   only that channel's W weights and dweight partials in registers, then
   transposes dx through shared memory for coalesced vector stores. The
   tile includes W-1 x rows on both sides and W-1 dout rows on the right,
   so silu' recomputation, anti-causal dx, seq_idx gating, initial states,
   and dinitial_states all match upstream v1.7.0 (including the virtual
   `seq_idx[b,0]` id before t=0).

   The launcher selects 128 rows when the grid is saturated and a 64-row
   specialization for `L<=128` or an underfilled grid (target 1024 blocks
   on CUDA/HIP, 512 on Metal); both live in one cached semantic variant.
   `_jit.py` gates on width<=5, `dim % kNElts == 0`, channel stride one,
   width-contiguous weights, 16-byte-aligned x/dout/dx/bias bases, and
   vector-preserving batch/L strides. Anything unsafe (including wider
   fp16/bf16 filters, odd channel counts, or sliced base pointers) stays
   on the generic kernel. `_fn.py` first normalizes dout into x's layout
   family, matching upstream.

   On H100 PCIe, fp16 w4+silu+bias reaches 35.5 µs vs upstream 37.2 µs
   at `(1,4096,2048,4)` (the scalar fallback was 216.4 µs); seq_idx is
   50.5 vs 69.3 µs. bf16 is 36.3 vs 38.1 µs and 50.5 vs 69.2 µs with
   seq_idx. The small `(1,1024,2048,4)` tile is at parity (9.51 vs
   9.27 µs), while `(8,2048,4096,4)` is 273.2 vs 300.5 µs. ncu on the
   canonical fp16 kernel reports 128 registers/thread, 38.16 KiB shared
   memory/block, 1,152,768 global-load sectors (the same as upstream),
   and 9.2% long-scoreboard / 2.6% barrier stalls. The generic kernel's
   before/after PTX is byte-identical.
11. **Independent input and parameter dtypes (`dtype`, `wdtype`).** All
   six kernels template the activation/state/output type separately
   from the weight/bias type. `weight` and `bias` share `wdtype`; every
   loaded parameter value is converted to the existing fp32 accumulator
   before the FMA chain. `x`, out/state tensors, dx, and
   dinitial_states stay `dtype`; dweight/dbias accumulate in fp32 and
   Python casts them back to their parameter dtype. Width limits and
   `kNElts` remain keyed on `dtype`. The dedicated channel-last forward
   and backward kernels take the same `wdtype` for their weight/bias
   pointers; both read those scalar from global (forward stages its tile
   through smem), so no parameter vector width depends on the pair.
   When `wdtype == dtype`, pointer
   types and vector alignments resolve to the previous specialization,
   so the same-dtype instruction path and fwd↔update bit-exactness
   contract are unchanged. The MPS small-shape fast paths in `_fn.py` /
   `_update.py` are gated on `weight.dtype == x.dtype`: they hand off to
   `causal_conv1d_ref`, which mirrors upstream in rounding x through
   `weight.dtype` (`x = x.to(weight.dtype)`) before convolving, whereas
   the kernels widen x and the parameters to fp32 independently. Those
   agree only while the dtypes match, so a mixed-dtype call must not
   change answer either side of the size threshold.

## CPU kernel design (`fwd_cpu/`, `bwd_full_cpu/`, `update_cpu/`)

The CPU kernels were rewritten from scalar TileTensor loops to
raw-pointer kernels with an explicitly vectorized main body (3.4–6.5×
on fwd/bwd, up to ~140× on update at M4 decode shapes). The patterns:

As on GPU, `dtype` types activations/state/output while `wdtype` types
only weight/bias; both are cast to fp32 at load time for the existing
accumulation chain.

1. **Raw pointers + element strides, no TileTensor.** `variant.mojo`
   decodes the args tuple into `UnsafePointer` + `Int` strides; each
   row's base pointer is computed once and the hot loop indexes
   `row[t * ls]`. This is what lets the fast path issue unaligned
   vector loads/stores along t.
2. **Boundary/main split.** Per (b, d) row: t < W-1 runs a scalar
   helper that keeps the `src_t < 0` handling (initial_states /
   implicit zeros), including the `seq_idx[b, 0]` virtual id when both
   features are present; t >= W-1 is branch-free (every tap in-range).
   The main region vectorizes (kV = 32 bytes of x per tap load) when
   x/out (and dout/dx for bwd) are unit-stride along t and there is no
   seq_idx; otherwise it falls back to the same scalar helper. W
   overlapping unaligned vector loads per kV outputs all hit L1 —
   DRAM sees each element once.
3. **bwd: chunked two-pass with a stack dpre buffer.** Pass A
   recomputes `pre` from the x taps, forms `dpre` (silu' in f32),
   stores it into a `kChunk`-entry stack buffer (512, plus `width`
   slack so the cross-seam extension region fits) *and* folds the
   dweight/dbias partial sums in the same step (the x taps are already
   in registers — dweight is nearly free). Pass B computes dx from the
   buffered dpre window. The buffer extends W-1 past the chunk so pass
   B never crosses a seam. By default dweight/dbias flush per row via
   relaxed atomics into fp32 accumulators; deterministic variants use
   plain stores into row-private `(B,D,W)` / `(B,D)` workspaces instead.
4. **Task-chunked parallelism.** All three kernels deal `batch*dim`
   rows to at most `8 * num_logical_cores()` contiguous row-chunks via
   `sync_parallelize` — one task per *row* drowned small rows in
   dispatch overhead (the old update kernel spent ~24 ms on a
   (32, 4096) decode step; chunked it's ~170 µs).
5. **fwd↔update bit-exactness contract.** `test_update` pins the
   decode loop against the one-shot forward at zero tolerance for
   fp16/bf16, so every path that produces an output element — fwd
   scalar boundary, fwd vector body, fwd scalar tail, update's
   per-token loop — accumulates with the *same* explicit
   ascending-k `fma` chain in f32, then the shared `_silu_f32`
   (`_silu.mojo`, width-generic; vector callers bind `[kV]`).
   Skipped taps (fwd boundary) and zero history (update) agree because
   `fma(0, w, acc) == acc` exactly. If you touch the accumulation
   order, fma-ness, or silu of one kernel, touch all of them.
6. **Perf profile on M4** (fp32, silu, clock-unlockable): fwd large
   shapes sit at ~50 GB/s of the 120 GB/s roofline — the vector
   `exp` in silu is the limiter (~1.7 ns/elem single-core vs 0.29
   without silu), not bandwidth; bwd ~30 GB/s (double exp-ish work per
   element: silu' recompute); update is dispatch-bound below ~100 µs.
   Parallel scaling is near-ideal for ≥ 40 µs of work per task and
   collapses below ~10 µs/task (Mojo runtime wake-up latency) — don't
   shrink TASKS_PER_CORE chunks further.

## Apple/MPS interop: heap revival before every dispatch

Mojo's Metal backend only declares its *own* allocations to the compute
encoder (`useResource:` is skipped for foreign raw-address buffers), and
macOS evicts idle GPU memory after ~1-1.5 s — so a Mojo kernel
dispatched against torch MPS tensors whose heaps sat idle silently reads
zeros and drops writes (root-caused + reported on the
`mojo-cold-cache-bug-repro` branch, see its `BUG_REPORT.md`). The
`_mps.revive_heaps` helper touches every argument tensor with a tiny
batched torch op right before each dispatch (as a `pre_dispatch`
callback that runs *after* the JIT compile), gated per-`data_ptr()` to
at most one revival per 0.35 s. A busy dispatch loop cannot idle-evict
(measured), so steady-state cost is nil. `CAUSAL_CONV1D_MPS_REVIVE=
always|off` overrides. If mps outputs ever read all-zero again, suspect
this machinery first.

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

- **Mojo 1.0 split the GPU stack out of the compiler's stdlib.**
  `layout`, `DeviceContext` (`max.gpu.host`), `barrier` (`max.gpu`) and
  `sync_parallelize` (`max.algorithm`) now ship in `max-mojo-libs`, and
  accelerator codegen needs `libmax.so` from `max-core` — without it
  `mojo build` fails with "please install MAX for accelerator support".
  `block_idx`/`thread_idx`/`grid_dim`, `std.gpu.globals` and
  `std.gpu.primitives.warp` stayed in `std`; `AddressSpace` moved from
  `std.gpu.memory` to `std.memory`.
- **`Int`/`UInt` are not `DevicePassable`.** A device kernel argument
  must be fixed-width, so the kernels take `Int32` and widen to `Int`
  in the first line of the body (the launcher casts at the
  `enqueue_function` call). `update/` already used `Int32` throughout
  for the PTX reasons below.
- **SIMD lengths must be powers of two.** Several register arrays are
  sized by the filter `width` (2..9) or `width + 1`, so they are
  declared `SIMD[..., next_power_of_two(width)]`. The padding lanes stay
  zero and are never indexed, so per-lane reads and the sum reductions
  are unaffected — but a *function signature* taking one of these
  (`_reduce_channellast_grads`, `_block_sum_f32_vec`) has to be padded
  the same way or the call won't type-check. Vector *loads* can't be
  padded (they'd read out of bounds); `update/`'s trailing-history load
  reads two lanes and picks up the odd element scalar.
- `DType` has no `.size_of()` method; use the free function
  `from std.sys import size_of` and call `size_of[dtype]()`.
- `stack_allocation[count, dtype, address_space=AddressSpace.SHARED]()`
  returns an `UnsafePointer` with **no** `.offset()` method. Use
  `ptr + i` for offsets.
- `comptime for x, y, ... in product(...)` only handles up to 4
  iterables. Nest loops or call `product` recursively.
- `constrained[...]` doesn't resolve in this SDK's kernels — encode
  comptime preconditions as comments + dispatcher gates instead. A
  violated comptime bound (e.g. a negative `InlineArray` index) still
  fails the `mojo build`, but only when that variant first JITs.
- A *runtime* branch selecting between two ways to fill a register
  array (e.g. "smem-staged vs direct weight loads, picked by
  `rows_per_block`") pessimizes regalloc across the whole kernel —
  measured 1.6-2x slowdowns even on shapes that always took the same
  arm. If two load strategies are needed, specialize at comptime or
  don't bother. Similarly, hoisting independent loads *earlier* (to
  overlap a barrier) can lose by extending live ranges: this kernel
  family is register-pressure-dominated; trust ncu's
  `launch__registers_per_thread` over intuition and re-measure.
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
