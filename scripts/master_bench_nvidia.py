#!/usr/bin/env python3
"""master_bench_nvidia.py — the one autonomous NVIDIA perf gate.

Fully non-interactive (passwordless ``sudo -n`` only — never prompts), so it
runs unattended in CI or under an agent. It is a stdlib-only *coordinator*:
the actual work happens in subprocesses run under the project venv
(``uv run --extra nvidia python benchmarks/bench.py ...`` and
``scripts/asm_tools.py``). This script just locks the environment, sequences
the phases, parses their JSON, and gates on the results.

Phases (Phase 1 of the Measurement Protocol):
    a. lock GPU clocks (+ CPU affinity)           [sudo -n; reset in finally]
    b. clear our JIT cache, recompile, correctness suite (quick/full tiers)
    c. benchmark vs cached/refreshed baseline (min + spread, 3% stop-criterion)
    d. ncu profiler pass (ephemeral via `pixi exec`; graceful no-op if absent)
    e. dump our PTX/SASS to assembly/nvidia/        (committed path)
    f. side-by-side instruction-mix histogram vs reference_assembly/nvidia/
    g. ptxas -v spill / regalloc canary             (fails loudly on regression)
    h. independent torch.utils.benchmark (walltime) run

Steps c/d/h are three *separate* processes on purpose — torch.profiler,
torch.utils.benchmark, and ncu must not share a run.

Usage:
    python scripts/master_bench_nvidia.py                 # QUICK tier
    python scripts/master_bench_nvidia.py --full          # FULL tier (gate)
    python scripts/master_bench_nvidia.py --fn all        # fwd+bwd+update
    python scripts/master_bench_nvidia.py --refresh-baseline
    python scripts/master_bench_nvidia.py --refresh-reference
    python scripts/master_bench_nvidia.py --no-lock --no-clean --skip-ncu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UV = ["uv", "run", "--extra", "nvidia", "python"]

# Canonical (quick) shape + full sweep, per function.
SHAPES = {
    "fwd": {
        "canon": "1,4096,2048,4",
        "full": ["1,1024,512,4", "1,1024,2048,4", "1,1024,8192,4", "1,2048,2048,4",
                 "1,4096,2048,4", "4,2048,2048,4", "4,4096,2048,4", "8,2048,4096,4"],
    },
    "update": {
        "canon": "16,2048",
        "full": ["1,512", "1,2048", "4,2048", "16,2048", "32,4096"],
    },
}
SHAPES["bwd"] = SHAPES["fwd"]

SUBPKG = {"fwd": "fwd", "bwd": "bwd_full", "update": "update"}

# Upstream kernel matched to our canonical fp16/width-4 config, for the
# instruction-mix histogram. (fwd: width4, vec-load, Half/Half traits.)
REF_MATCH = {"fwd": ["Li4ELb1EN3c104HalfES2_E"], "bwd": ["Li4E"], "update": ["Li4E"]}


# --------------------------------------------------------------------------
# Small process / printing helpers.
# --------------------------------------------------------------------------

_BOLD, _RST, _YEL, _RED = "\033[1m", "\033[0m", "\033[33m", "\033[31m"


def section(msg: str) -> None:
    print(f"\n{_BOLD}===== {msg} ====={_RST}", flush=True)


def warn(msg: str) -> None:
    print(f"{_YEL}[warn]{_RST} {msg}", file=sys.stderr, flush=True)


class Gate:
    """Accumulates gate failures; the process exit code reflects it."""

    failed = False

    @classmethod
    def fail(cls, msg: str) -> None:
        cls.failed = True
        print(f"{_RED}[FAIL]{_RST} {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], *, env=None, capture=False, check=False) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command. Streams output unless captured."""
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd, env=env, check=check, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def nvidia_smi(query: str) -> str:
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""


def taskset_prefix() -> list[str]:
    """Pin workers to a fixed CPU set to bound wall-clock noise (best-effort)."""
    return ["taskset", "-c", "0-3"] if shutil.which("taskset") else []


# --------------------------------------------------------------------------
# (a) Lock the measurement environment.
# --------------------------------------------------------------------------


def lock_clocks(enabled: bool) -> tuple[str, bool]:
    """Lock SM (+ memory) clocks via passwordless sudo. Returns (clock, locked)."""
    section("(a) lock GPU clocks")
    if not enabled:
        warn("--no-lock: clocks not locked (dev mode).")
        return "unlocked", False

    def sudo(args: list[str]) -> bool:
        return subprocess.run(
            ["sudo", "-n", "nvidia-smi", *args],
            capture_output=True, text=True,
        ).returncode == 0

    max_sm = nvidia_smi("clocks.max.sm")
    if max_sm and sudo(["-pm", "1"]) and sudo([f"--lock-gpu-clocks={max_sm}"]):
        max_mem = nvidia_smi("clocks.max.mem")
        if max_mem:
            sudo([f"--lock-memory-clocks={max_mem}"])  # best-effort
        print(f"locked SM clock to {max_sm} MHz (will reset on exit)")
        return max_sm, True
    warn("passwordless sudo / clock-lock unavailable — running UNLOCKED.")
    warn("min/spread will be noisier; treat sub-spread deltas as noise.")
    return "unlocked", False


def unlock_clocks() -> None:
    for args in (["--reset-gpu-clocks"], ["--reset-memory-clocks"]):
        subprocess.run(["sudo", "-n", "nvidia-smi", *args], capture_output=True)


# --------------------------------------------------------------------------
# (b) Recompile (clear OUR JIT cache only) + correctness.
# --------------------------------------------------------------------------

# Quick smoke: one representative test per function family, cuda+fp16,
# silu+bias — catches obvious regressions in a handful of variants.
SMOKE_K = (
    "cuda and fp16 and ("
    "(test_contiguous and silu and with_bias) or "
    "(test_width_backward and with_bias) or "
    "(test_update_single_token and silu and with_bias))"
)


def correctness(tier: str, clean: bool) -> bool:
    section(f"(b) recompile + correctness ({tier} tier)")
    if clean:
        cache = Path("~/.cache/causal_conv1d_mojo").expanduser()
        print(f"clearing JIT cache {cache} (keeping mojo compiler cache)")
        shutil.rmtree(cache, ignore_errors=True)
    if tier == "quick":
        cmd = ["uv", "run", "--extra", "nvidia", "pytest", "-q", "-x",
               "tests/test_fwd.py", "tests/test_bwd.py", "tests/test_update.py",
               "-k", SMOKE_K]
    else:
        print("full regression suite (every landed feature, all devices/dtypes)")
        cmd = ["uv", "run", "--extra", "nvidia", "pytest", "-q"]
    ok = run(cmd).returncode == 0
    if not ok:
        Gate.fail(f"{tier} correctness failed")
    else:
        print("correctness OK")
    return ok


# --------------------------------------------------------------------------
# (c) Benchmark vs baseline — per-kernel GPU time (torch.profiler).
# --------------------------------------------------------------------------


def bench_kernel(fn, dtype, shapes, runs, clock, baseline_flags) -> None:
    section("(c) kernel-time bench vs upstream baseline (torch.profiler)")
    rows = []
    for shape in shapes:
        cmd = [*taskset_prefix(), *UV, str(REPO / "benchmarks" / "bench.py"), fn,
               "--shape", shape, "--dtype", dtype, "--impl", "all",
               "--measure", "kernel", "--runs", str(runs),
               "--clock-locked", clock, "--json", *baseline_flags]
        r = run(cmd, capture=True)
        if r.stderr:
            sys.stderr.write(r.stderr)
        line = (r.stdout or "").strip().splitlines()
        if r.returncode != 0 or not line:
            Gate.fail(f"bench failed on {shape}")
            continue
        try:
            rows.append(json.loads(line[-1]))
        except json.JSONDecodeError:
            Gate.fail(f"bench produced no JSON on {shape}")
    _aggregate(rows)


def _aggregate(rows: list[dict]) -> None:
    if not rows:
        Gate.fail("no parseable bench results")
        return
    print(f"\n  {'shape':>18} | {'mojo us':>9} | {'up us':>9} | {'ratio':>7} | {'spread':>7} | verdict")
    print("  " + "-" * 78)
    worst = 0.0
    for r in rows:
        res = r["results"]
        mojo = res.get("mojo", {}).get("min_us", math.nan)
        up = res.get("upstream", {}).get("min_us", math.nan)
        ratio = r["ratio_min"].get("mojo_over_upstream", math.nan)
        # Trust gate uses only the two impls in the ratio — a noisy third
        # impl (pytorch) must not mask a real mojo-vs-upstream delta.
        spread = max((res[i]["spread_pct"] for i in ("mojo", "upstream")
                      if i in res and res[i]["runs_us"]), default=0.0)
        if ratio != ratio:  # NaN — mojo and/or upstream produced no usable number
            errs = [f"{i}={res[i]['error']}" for i in ("mojo", "upstream")
                    if res.get(i, {}).get("error")]
            verdict = "NO-BASELINE"
            Gate.fail(f"no usable mojo/upstream measurement on {tuple(r['shape'])}"
                      + (f" ({'; '.join(errs)})" if errs else ""))
        else:
            gap = abs(ratio - 1) * 100
            worst = max(worst, ratio)
            if gap <= 3:
                verdict = "parity"
            elif gap < spread:
                verdict = "NOISE"
            elif ratio < 1:
                verdict = "faster"
            else:
                verdict = "SLOWER"
                Gate.fail(f"perf regression on {tuple(r['shape'])}: {ratio:.3f}x "
                          f"(gap {gap:.1f}% > spread {spread:.1f}%)")
        print(f"  {str(tuple(r['shape'])):>18} | {mojo:9.2f} | {up:9.2f} | "
              f"{ratio:6.3f}x | {spread:6.1f}% | {verdict}")
    print("  " + "-" * 78)
    if worst == 0.0:
        print("  no usable mojo/upstream comparison — see [FAIL] lines above")
    else:
        tail = ("(within 3% across all shapes — PASS)" if worst <= 1.03
                else "(REVIEW: a shape exceeds 1.03x)")
        print(f"  worst mojo/upstream ratio: {worst:.3f}x  {tail}")


# --------------------------------------------------------------------------
# (d) ncu profiler pass (ephemeral via pixi exec).
# --------------------------------------------------------------------------

_NCU_METRICS = ",".join([
    "gpu__time_duration.avg",
    "launch__waves_per_multiprocessor",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "smsp__inst_executed_op_shared_ld.sum",
])


def _ncu_cmd() -> list[str] | None:
    if shutil.which("ncu"):
        return ["ncu"]
    if shutil.which("pixi"):
        # Ephemeral — no global install; cached after first use.
        return ["pixi", "exec", "--spec", "nsight-compute", "--", "ncu"]
    return None


def profiler(fn, dtype, canon, skip) -> None:
    section("(d) ncu deep profiler")
    if skip:
        print("skipped (--skip-ncu)")
        return
    ncu = _ncu_cmd()
    if ncu is None:
        warn("ncu not found and pixi unavailable — skipping deep profiling.")
        warn("(the raw-mode driver 'bench.py ... --measure raw' is ready for it.)")
        return
    print(f"profiler: {' '.join(ncu)}")
    cmd = [*ncu, "--target-processes", "all", "--launch-skip", "10",
           "--launch-count", "5", "--metrics", _NCU_METRICS,
           *UV, str(REPO / "benchmarks" / "bench.py"), fn, "--shape", canon,
           "--dtype", dtype, "--impl", "mojo", "--measure", "raw",
           "--iters", "30", "--warmup", "10"]
    if run(cmd).returncode != 0:
        warn("ncu run failed (perf-counter perms? needs root/CAP_SYS_ADMIN)")


# --------------------------------------------------------------------------
# (e) dump PTX/SASS  +  (f) histogram  +  (g) spill canary.
# --------------------------------------------------------------------------


def assembly(fn, dtype, canon, sm, sm_a, refresh_reference) -> None:
    asm_dir = REPO / "assembly" / "nvidia"
    ref_dir = REPO / "reference_assembly" / "nvidia"
    asm_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    tools = str(REPO / "scripts" / "asm_tools.py")

    section("(e) dump our PTX/SASS -> assembly/nvidia/")
    for stale in asm_dir.glob(f"{SUBPKG[fn]}__*.ptx"):
        stale.unlink()
    env = os.environ.copy()
    env["CAUSAL_CONV1D_DUMP_ASM"] = str(asm_dir)
    dump = run(
        [*UV, str(REPO / "benchmarks" / "bench.py"), fn, "--shape", canon,
         "--dtype", dtype, "--impl", "mojo", "--measure", "raw",
         "--iters", "1", "--warmup", "0", "--runs", "1"],
        env=env, capture=True,
    )
    ptxs = sorted(asm_dir.glob(f"{SUBPKG[fn]}__*.ptx"), key=lambda p: p.stat().st_mtime)
    if dump.returncode != 0 or not ptxs:
        if dump.stderr:
            sys.stderr.write(dump.stderr)
        Gate.fail(f"PTX dump failed for {fn}")
        return
    ptx = str(ptxs[-1])
    our_sass = str(asm_dir / f"{fn}.sass")
    if run([*UV, tools, "sass", ptx, our_sass, "--arch", sm_a]).returncode != 0:
        Gate.fail("PTX->SASS failed")

    section("(g) ptxas -v spill / regalloc canary")
    if run([*UV, tools, "spill", ptx, "--arch", sm_a, "--max-spill", "0"]).returncode != 0:
        Gate.fail("spill canary: register spills detected")

    section("(f) instruction-mix histogram vs upstream reference")
    ref_sass = ref_dir / f"{fn}.sass"
    if refresh_reference or not ref_sass.exists():
        print(f"extracting upstream {fn} reference SASS")
        cmd = [*UV, tools, "upstream-sass", fn, str(ref_sass), "--arch", sm]
        for m in REF_MATCH[fn]:
            cmd += ["--match", m]
        if run(cmd).returncode != 0:
            warn(f"could not extract upstream reference ({fn}); skipping histogram")
    if ref_sass.exists():
        if run([*UV, tools, "histogram", our_sass, str(ref_sass)]).returncode != 0:
            warn("histogram diff failed")


# --------------------------------------------------------------------------
# (h) Independent wall-clock run (torch.utils.benchmark).
# --------------------------------------------------------------------------


def walltime(fn, dtype, canon, runs, clock, baseline_flags) -> None:
    section("(h) end-to-end wall-clock (torch.utils.benchmark, auto cpu<->gpu sync)")
    cmd = [*taskset_prefix(), *UV, str(REPO / "benchmarks" / "bench.py"), fn,
           "--shape", canon, "--dtype", dtype, "--impl", "all",
           "--measure", "walltime", "--runs", str(runs),
           "--clock-locked", clock, *baseline_flags]
    if run(cmd).returncode != 0:
        warn("walltime run failed")


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="full tier (task gate)")
    p.add_argument("--fn", choices=("fwd", "bwd", "update", "all"), default="fwd")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--skip-ncu", action="store_true")
    p.add_argument("--skip-asm", action="store_true")
    p.add_argument("--refresh-baseline", action="store_true")
    p.add_argument("--refresh-reference", action="store_true")
    args = p.parse_args()

    os.chdir(REPO)
    tier = "full" if args.full else "quick"
    fns = ["fwd", "bwd", "update"] if args.fn == "all" else [args.fn]
    runs = 5 if args.full else 3
    # Full tier re-seeds the baseline (authoritative); quick reuses the cache.
    baseline_flags = ["--refresh-baseline"] if (args.full or args.refresh_baseline) else []

    gpu = nvidia_smi("name")
    cc = nvidia_smi("compute_cap").replace(".", "")
    sm, sm_a = f"sm_{cc}", f"sm_{cc}a"
    print(f"master_bench_nvidia: fn={args.fn} tier={tier} dtype={args.dtype} "
          f"gpu='{gpu}' arch={sm}/{sm_a}")

    clock, locked = lock_clocks(not args.no_lock)
    if locked:
        # SIGINT/exceptions hit the finally below, but SIGTERM/SIGHUP (CI
        # cancellation, terminal close) would otherwise leave clocks locked.
        def _on_signal(signum, _frame):
            unlock_clocks()
            os._exit(128 + signum)

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)
    try:
        # (b) correctness is tier-based and covers every function family.
        if not correctness(tier, not args.no_clean):
            return 1
        for fn in fns:
            sh = SHAPES[fn]
            shapes = [sh["canon"]] if tier == "quick" else sh["full"]
            if len(fns) > 1:
                section(f"### function: {fn} ###")
            bench_kernel(fn, args.dtype, shapes, runs, clock, baseline_flags)
            profiler(fn, args.dtype, sh["canon"], args.skip_ncu)
            if not args.skip_asm:
                assembly(fn, args.dtype, sh["canon"], sm, sm_a, args.refresh_reference)
            else:
                section("(e/f/g) assembly — skipped (--skip-asm)")
            walltime(fn, args.dtype, sh["canon"], runs, clock, baseline_flags)
    finally:
        if locked:
            unlock_clocks()

    section("summary")
    if Gate.failed:
        print(f"ISSUES — one or more gates failed above (fn={args.fn} tier={tier})")
        return 1
    print(f"PASS — correctness, perf, and spill canary all green (fn={args.fn} tier={tier})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
