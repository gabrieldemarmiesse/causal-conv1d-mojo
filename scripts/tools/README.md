# Perf tools (MI300X)

Helpers for measuring and inspecting the Mojo causal-conv1d kernels on
AMD. All commands serialise on `/tmp/.gpu_bench_lock` so it's safe to
launch them in parallel — they block until the GPU is free.

## Quick reference

| Tool | What it does |
|---|---|
| `./scripts/tools/bench <fn> [args]` | Run the unified `scripts/bench.py` driver with the GPU lock applied. `fn` is `fwd`/`bwd`/`update` (legacy `--kind` still accepted); `--impl mojo,upstream,pytorch` (or `all`/`both`), `--shape B,D,L,W` (or B,D for update), `--measure kernel,walltime,raw`, `--iters`, `--warmup`. |
| `./scripts/tools/dump_isa --sub <fwd\|bwd_full\|update> --shape <...>` | Set `CAUSAL_CONV1D_DUMP_ASM` and JIT-compile **one** variant, dumping its GPU ISA (PTX on NVIDIA, AMDGPU asm on ROCm) to `/tmp/mojo_isa/<sub>__<modname>.ptx`. Non-invasive — no `launch.mojo` patching. |
| `./scripts/tools/dump_upstream_isa [filter]` | Extract gfx942 code object from the upstream `.so`, disassemble all kernels into `/tmp/upstream_isa/upstream.s`. With a `filter` arg (regex), also writes `/tmp/upstream_isa/upstream_<filter>.s`. |
| `./scripts/tools/rocprof_kernels [bench args]` | Runs `rocprofv3 --kernel-trace --stats` against the bench. Prints per-kernel GPU time CSV. **Forces `--impl upstream`** unless you override (Mojo doesn't survive rocprof instrumentation). |
| `./scripts/tools/rocprof_pmc --kernel <regex> [bench args]` | Collect MI300X HW counters (VALU/SALU/LDS/VMEM/TCC) for kernels matching `<regex>`. Same upstream-only constraint. |
| `./scripts/tools/gpu_run <cmd...>` | Generic flock wrapper. Use for ad-hoc commands that touch the GPU. |
| `./scripts/tools/quick_test <fwd\|bwd\|update> '<pytest -k expr>'` | Run a *targeted* subset of the pytest suite — pick a `-k` expression that selects at most ~3 cases. |
| `./scripts/tools/hint <agent-name>` (stdin = body) | Append a perf hint to `/tmp/hints.md`. Use for *validated* wins so other agents can see them. |

## The fast iteration loop

```bash
# 1. Baseline: just the kernel you care about, one shape.
./scripts/tools/bench --kind fwd --impl both --shape 1,1024,2048,4 --iters 50

# 2. Edit the kernel/launch source.
# 3. Re-measure mojo only (upstream is cached via baselines/).
./scripts/tools/bench --kind fwd --impl mojo --shape 1,1024,2048,4 --iters 50

# 4. If the change looks good across a few shapes, do correctness check:
./scripts/tools/quick_test fwd 'width4 and fp16 and seqlen_2048'

# 5. If validated, write a hint so the other agents see it.
./scripts/tools/hint $YOUR_NAME <<'EOF'
### fwd: vec8 fp16 LDG buys 18% on (1,1024,2048,4)
- Before: 7.2 us/call (1.18x upstream)
- After:  6.0 us/call (0.98x upstream)
EOF
```

## When you want to read the assembly

```bash
# Mojo side — pick the variant for your shape/dtype.
./scripts/tools/dump_isa --sub fwd --shape 1,1024,2048,4
# → /tmp/mojo_isa/fwd__<modname>.ptx   (PTX on NVIDIA; AMDGPU asm on ROCm)
# On NVIDIA, turn it into SASS / a spill report:
#   uv run python scripts/_asm_tools.py sass  /tmp/mojo_isa/fwd__*.ptx out.sass
#   uv run python scripts/_asm_tools.py spill /tmp/mojo_isa/fwd__*.ptx

# Upstream side — filter to a kernel substring.
./scripts/tools/dump_upstream_isa fwd_kernel
# → /tmp/upstream_isa/upstream.s             (all kernels)
# → /tmp/upstream_isa/upstream_fwd_kernel.s  (just the fwd ones)
```

Things to compare:
- VGPR/SGPR usage (occupancy) — `grep '.num_vgpr\|.numbered_sgpr'`
- `global_load_dwordx{2,3,4}` count — vector-load width
- `ds_read_b{64,128}` count — LDS read width
- `v_pk_fma_f16` count — packed-fp16 FMAs (free 2× throughput on MI300X)
- `s_barrier` count — barrier overhead
- `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` — pipeline stalls

## Reference repos in /tmp

- `/tmp/causal-conv1d/` — Tri Dao's HIP source (read-only).
  - `csrc/causal_conv1d_fwd.cu`, `_bwd.cu`, `_update.cu` — the kernels.
  - `rocm_patch/` — HIP-specific patches.
- `/tmp/modular/` — Modular MAX/Mojo source.
  - `mojo/stdlib/std/gpu/host/device_context.mojo` — `compile_function` signature.
  - `max/` — Mojo MAX layer ops (incl. their own `causal_conv1d.mojo`).

## The GPU lock

`/tmp/.gpu_bench_lock` is shared with another agent. Every tool here
wraps its inner command in `flock /tmp/.gpu_bench_lock ...`, so calls
serialise automatically. If you write a one-off script, do the same:

```bash
flock /tmp/.gpu_bench_lock <cmd...>
```

## Caveats

- **rocprofv3 vs Mojo:** rocprofv3 intercepts HSA at process startup,
  and Mojo's `DeviceContext` then can't initialise its own HIP runtime
  ("HIP architecture query failed: Failed to initialize HSA runtime").
  So `rocprof_kernels` and `rocprof_pmc` only work with `--impl upstream`.
  For Mojo per-kernel time, `./scripts/tools/bench` (which uses `torch.profiler`)
  is the right tool — PyTorch already owns HSA, no conflict.
- **Cold cache cost:** first call per (config, machine) pays ~1-3 s for
  Mojo's `mojo build`. The cache is content-addressed under
  `~/.cache/causal_conv1d_mojo/<sub>/`, so subsequent calls are
  effectively free.
- **Hints file:** `/tmp/hints.md` is shared across agents. Only write
  *validated* perf insights (you measured before/after, the change
  reproduces). One hint per genuine improvement.
