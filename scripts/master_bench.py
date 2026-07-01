#!/usr/bin/env python3
"""master_bench.py — the one autonomous, backend-agnostic perf gate.

Fully non-interactive (passwordless ``sudo -n`` only — never prompts), so it
runs unattended in CI or under an agent. It is a stdlib-only *coordinator*:
the actual work happens in subprocesses run under the project venv
(``uv run [--extra <backend>] python scripts/_bench.py ...`` and, on NVIDIA,
``scripts/_asm_tools.py``). This script just detects the GPU backend, locks
the environment, sequences the phases, parses their JSON, and gates on the
results.

The same phase skeleton runs on every backend; each phase dispatches to the
backend-appropriate tooling and **skips cleanly when that tooling does not
exist** (Apple has no ``ptxas -v`` spill canary, ROCm's ``rocprofv3`` can't
instrument Mojo's ``DeviceContext``, CPU has no GPU asm, …):

    a. lock clocks                          cuda: nvidia-smi · rocm: rocm-smi
                                            metal: induced GPU perf state (xctrace) ·
                                            cpu: n/a (skip)
    b. clear our JIT cache, recompile, correctness suite (quick/full tiers)
    c. kernel-time bench vs baseline        cuda: vs upstream (gate) ·
                                            rocm/cpu: vs pytorch-ref (report) ·
                                            metal: absolute GPU time (xctrace)
    d. deep profiler                        cuda: ncu · metal: xctrace stats ·
                                            cpu: perf stat · rocm: n/a (skip)
    e. dump GPU asm                         cuda: PTX/SASS · rocm: GCN ISA ·
                                            metal/cpu: n/a (skip)
    f. instruction-mix histogram vs upstream            cuda only (else skip)
    g. ptxas -v spill / regalloc canary                 cuda only (else skip)
    h. independent torch.utils.benchmark (walltime) run

┌────────────────┬────────────────────┬────────────────────────────┬───────────────────────┬─────────────────────┐
│     phase      │        cuda        │            rocm            │         metal         │         cpu         │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ a lock         │ nvidia-smi (gate)  │ rocm-smi perflevel (gate)  │ induced state (gate)  │ skip                │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ b correctness  │ -k cuda +nvidia    │ -k cuda +rocm              │ -k mps                │ -k cpu              │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ c bench        │ vs upstream (gate) │ vs pytorch (report)        │ absolute xctrace time │ walltime vs pytorch │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ d profiler     │ ncu                │ skip (rocprof breaks Mojo) │ xctrace stats         │ perf stat           │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ e asm          │ PTX/SASS           │ GCN ISA                    │ skip (no textual ISA) │ skip                │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ f/g hist+spill │ yes                │ skip                       │ skip                  │ skip                │
├────────────────┼────────────────────┼────────────────────────────┼───────────────────────┼─────────────────────┤
│ h walltime     │ yes                │ yes                        │ yes                   │ yes                 │
└────────────────┴────────────────────┴────────────────────────────┴───────────────────────┴─────────────────────┘

Backend selection is automatic (NVIDIA via ``nvidia-smi``, AMD via
``rocminfo``/``rocm-smi``, Apple via ``sys.platform``, else CPU); override
with ``--backend``.

The kernel-time baseline differs by backend: only NVIDIA has Tri Dao's
hand-tuned CUDA kernels to diff against, so only NVIDIA's step (c) is a true
perf *gate* (regression => non-zero exit). ROCm/CPU report mojo's speedup
over the pure-PyTorch fallback (informational); Apple reports absolute
per-kernel GPU time read back from a Metal System Trace (there is no
upstream to diff against — upstream is CUDA-only).

Apple has no public clock-lock API/CLI; step (a) instead reproduces
Instruments' GUI-only "Induced GPU Performance State = Maximum" setting by
binary-patching a copy of its Metal System Trace template (see
``scripts/_apple_gpu_clock_lock.py``).

Step (a) is a hard gate everywhere: an unlocked GPU makes measurements
across runs incomparable, which defeats the point of a perf gate feeding
an agentic loop. A failed lock (no passwordless sudo, or — Apple only — a
future Xcode update breaking the patch) exits non-zero rather than
silently continuing unlocked; ``--no-lock`` is the explicit opt-out for
local dev loops.

Steps c/d/h are separate processes on purpose — torch.profiler,
torch.utils.benchmark, and ncu must not share a run.

Usage:
    python scripts/master_bench.py                 # QUICK tier, auto-detect
    python scripts/master_bench.py --full          # FULL tier (gate)
    python scripts/master_bench.py --fn all        # fwd+bwd+update
    python scripts/master_bench.py --backend cpu   # force a backend
    python scripts/master_bench.py --refresh-baseline
    python scripts/master_bench.py --refresh-reference     # nvidia asm only
    python scripts/master_bench.py --skip-correctness      # perf phases only
    python scripts/master_bench.py --no-lock --no-clean --skip-ncu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent

# Canonical (quick) shape + full sweep, per function. Backend-independent —
# the shapes that exercise the kernel are the same everywhere.
SHAPES = {
    "fwd": {
        "canon": "1,4096,2048,4",
        "full": [
            "1,1024,512,4",
            "1,1024,2048,4",
            "1,1024,8192,4",
            "1,2048,2048,4",
            "1,4096,2048,4",
            "4,2048,2048,4",
            "4,4096,2048,4",
            "8,2048,4096,4",
        ],
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
# NVIDIA-only (there is no upstream kernel to match on other backends).
REF_MATCH = {"fwd": ["Li4ELb1EN3c104HalfES2_E"], "bwd": ["Li4E"], "update": ["Li4E"]}

# Upstream git tag to compile for the *PTX*-level histogram. The shipped wheel
# is cubin-only (no embedded PTX), so we clone + compile the .cu with nvcc to
# get comparable PTX. Keep in sync with the `causal-conv1d` wheel pinned in the
# nvidia extra of pyproject.toml.
UPSTREAM_REF = "v1.6.2.post1"


# --------------------------------------------------------------------------
# Backend description: every per-backend knob the phases below branch on.
# --------------------------------------------------------------------------


@dataclass
class Backend:
    """Everything the phases need to know about the active accelerator.

    One instance is built by ``detect_backend()`` (or ``--backend``) and
    threaded through every step. Fields capture *what differs* per backend;
    the phase functions stay one body each and dispatch on ``name``.
    """

    name: str  # "cuda" | "rocm" | "metal" | "cpu"
    pretty: str  # human GPU/CPU name for the banner
    arch: str  # "sm_89" / "gfx942" / "macos15" / "" (nvidia uses it for asm)
    arch_a: str = ""  # ptxas target with the 'a' suffix (nvidia only)
    device: str = "cuda"  # torch device passed to _bench.py (--device)
    test_device: str = "cuda"  # pytest -k device token
    kernel_impls: tuple[str, ...] = ("mojo",)  # impls for the kernel bench
    walltime_impls: tuple[str, ...] = ("mojo",)  # impls for the walltime run
    baseline: str | None = None  # ratio baseline impl, or None for absolute
    gate_ratio: bool = False  # does a slow ratio FAIL the gate?
    kernel_measure: str = "kernel"  # _bench.py --measure for step (c)


# --------------------------------------------------------------------------
# Small process / printing helpers.
# --------------------------------------------------------------------------

_BOLD, _RST, _YEL, _RED = "\033[1m", "\033[0m", "\033[33m", "\033[31m"

# Echo every spawned subprocess command line (the `$ ...` lines). Off by
# default to keep the phase output readable; flipped on by `--verbose`.
VERBOSE = False


def section(msg: str) -> None:
    print(f"\n{_BOLD}===== {msg} ====={_RST}", flush=True)


def warn(msg: str) -> None:
    print(f"{_YEL}[warn]{_RST} {msg}", file=sys.stderr, flush=True)


def skip(step: str, reason: str) -> None:
    """Announce a phase that does not apply to the active backend."""
    section(step)
    print(f"skipped — {reason}")


class Gate:
    """Accumulates gate failures; the process exit code reflects it."""

    failed = False

    @classmethod
    def fail(cls, msg: str) -> None:
        cls.failed = True
        print(f"{_RED}[FAIL]{_RST} {msg}", file=sys.stderr, flush=True)


def run(
    cmd: list[str], *, env=None, capture=False, check=False
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command under --verbose. Streams output
    unless captured."""
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def taskset_prefix() -> list[str]:
    """Pin workers to a fixed CPU set to bound wall-clock noise (best-effort)."""
    return ["taskset", "-c", "0-3"] if shutil.which("taskset") else []


# --------------------------------------------------------------------------
# Backend detection. Stdlib-only — we deliberately do NOT import torch here
# (that needs the right uv extra, which we haven't chosen yet); we probe the
# vendor CLIs / platform instead, then derive the arch best-effort.
# --------------------------------------------------------------------------


def nvidia_smi(query: str) -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
    except OSError:  # nvidia-smi not installed
        return ""
    return r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _sysctl(name: str) -> str:
    r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def _rocm_arch() -> str:
    """Best-effort gfx target from rocminfo (e.g. 'gfx942'); '' if unknown."""
    if not _have("rocminfo"):
        return ""
    r = subprocess.run(["rocminfo"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        line = line.strip()
        # rocminfo prints `Name:  gfx942` for the GPU agent.
        if line.startswith("Name:") and "gfx" in line:
            return line.split()[-1].split(":")[0]
    return ""


def _rocm_name() -> str:
    if _have("rocm-smi"):
        r = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True
        )
        for line in r.stdout.splitlines():
            if "Card series" in line or "Card model" in line:
                return line.split(":", 1)[-1].strip()
    return _rocm_arch() or "AMD GPU"


def _make_cuda_backend() -> Backend:
    cc = nvidia_smi("compute_cap").replace(".", "")
    return Backend(
        name="cuda",
        pretty=nvidia_smi("name") or "NVIDIA GPU",
        arch=f"sm_{cc}",
        arch_a=f"sm_{cc}a",
        device="cuda",
        test_device="cuda",
        kernel_impls=("mojo", "upstream", "pytorch"),
        walltime_impls=("mojo", "upstream", "pytorch"),
        baseline="upstream",
        gate_ratio=True,
        kernel_measure="kernel",
    )


def _make_rocm_backend() -> Backend:
    # Under ROCm, torch exposes the GPU as the "cuda" device (HIP), so both
    # the bench --device and the pytest -k token stay "cuda". There is no
    # upstream wheel for ROCm, so we report mojo's speedup over the pytorch
    # fallback rather than gating against a hand-tuned baseline.
    return Backend(
        name="rocm",
        pretty=_rocm_name(),
        arch=_rocm_arch(),
        device="cuda",
        test_device="cuda",
        kernel_impls=("mojo", "pytorch"),
        walltime_impls=("mojo", "pytorch"),
        baseline="pytorch",
        gate_ratio=False,
        kernel_measure="kernel",
    )


def _make_metal_backend() -> Backend:
    major = (platform.mac_ver()[0] or "").split(".")[0]
    chip = _sysctl("machdep.cpu.brand_string") or "Apple GPU"
    # Kernel time on Apple is an out-of-process xctrace orchestration that
    # _bench.py owns; it is mojo-only (upstream is CUDA-only) and yields an
    # *absolute* per-kernel GPU time, so there is no ratio to gate.
    return Backend(
        name="metal",
        pretty=chip,
        arch=f"macos{major}" if major else "macos",
        device="mps",
        test_device="mps",
        kernel_impls=("mojo",),
        walltime_impls=("mojo", "pytorch"),
        baseline=None,
        gate_ratio=False,
        kernel_measure="kernel",
    )


def _make_cpu_backend() -> Backend:
    # No CUDA events on CPU, so torch.profiler kernel-time is empty; step (c)
    # falls back to wall-clock. We report mojo's speedup over the pytorch
    # fallback (informational, not a gate).
    return Backend(
        name="cpu",
        pretty=platform.processor() or platform.machine() or "CPU",
        arch="",
        device="cpu",
        test_device="cpu",
        kernel_impls=("mojo", "pytorch"),
        walltime_impls=("mojo", "pytorch"),
        baseline="pytorch",
        gate_ratio=False,
        kernel_measure="walltime",
    )


_BACKEND_FACTORIES = {
    "cuda": _make_cuda_backend,
    "rocm": _make_rocm_backend,
    "metal": _make_metal_backend,
    "cpu": _make_cpu_backend,
}


def detect_backend(forced: str | None) -> Backend:
    """Pick the backend: explicit ``--backend`` wins, else auto-probe.

    Probe order mirrors _bench.py's device resolution: a real NVIDIA GPU
    (nvidia-smi reports a name), else an AMD GPU (rocminfo/rocm-smi), else
    Apple (darwin), else CPU.
    """
    if forced:
        return _BACKEND_FACTORIES[forced]()
    if _have("nvidia-smi") and nvidia_smi("name"):
        return _make_cuda_backend()
    if _have("rocminfo") or _have("rocm-smi"):
        return _make_rocm_backend()
    if sys.platform == "darwin":
        return _make_metal_backend()
    return _make_cpu_backend()


# --------------------------------------------------------------------------
# (a) Lock the measurement environment.
# --------------------------------------------------------------------------


def lock_clocks(be: Backend, enabled: bool) -> tuple[str, bool]:
    """Lock accelerator clocks. Returns (clock_tag, locked).

    The ``clock_tag`` is folded into _bench.py's baseline-cache key so cached
    numbers never mix locked and unlocked measurements.
    """
    section("(a) lock clocks")
    if not enabled:
        warn("--no-lock: clocks not locked (dev mode).")
        return "unlocked", False
    if be.name == "cuda":
        return _lock_nvidia()
    if be.name == "rocm":
        return _lock_rocm()
    if be.name == "metal":
        return _lock_metal()
    # CPU has no GPU clock to lock.
    warn(f"no headless clock lock on {be.name}; running UNLOCKED.")
    return "unlocked", False


def _fatal_lock_failure(reason: str) -> NoReturn:
    """Stop the run: a clock lock failed and --no-lock wasn't passed."""
    print(f"{_RED}[FAIL]{_RST} GPU clock lock unavailable — refusing to run unlocked.")
    print(f"{_RED}[FAIL]{_RST} ({reason})")
    print(f"{_RED}[FAIL]{_RST} pass --no-lock to run unlocked anyway (dev mode).")
    raise SystemExit(1)


def _lock_nvidia() -> tuple[str, bool]:
    def sudo(args: list[str]) -> bool:
        return (
            subprocess.run(
                ["sudo", "-n", "nvidia-smi", *args], capture_output=True, text=True
            ).returncode
            == 0
        )

    max_sm = nvidia_smi("clocks.max.sm")
    if max_sm and sudo(["-pm", "1"]) and sudo([f"--lock-gpu-clocks={max_sm}"]):
        max_mem = nvidia_smi("clocks.max.mem")
        if max_mem:
            sudo([f"--lock-memory-clocks={max_mem}"])  # best-effort
        print(f"locked SM clock to {max_sm} MHz (will reset on exit)")
        return max_sm, True
    _fatal_lock_failure(
        "passwordless sudo / nvidia-smi clock-lock unavailable — configure "
        "`sudo -n nvidia-smi -pm 1 --lock-gpu-clocks=...` without a password prompt"
    )


def _lock_rocm() -> tuple[str, bool]:
    # rocm-smi has no fixed-MHz lock like nvidia-smi; the closest headless
    # knob is forcing the highest DPM performance level.
    ok = (
        subprocess.run(
            ["sudo", "-n", "rocm-smi", "--setperflevel", "high"],
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    if ok:
        print("set ROCm perf level to 'high' (will reset to auto on exit)")
        return "high", True
    _fatal_lock_failure(
        "passwordless sudo / rocm-smi perf-level unavailable — configure "
        "`sudo -n rocm-smi --setperflevel high` without a password prompt"
    )


# Read by _bench.py's _record_trace to pick the xctrace template; must match
# _XCTRACE_TEMPLATE_ENV in _bench.py.
_XCTRACE_TEMPLATE_ENV = "CAUSAL_CONV1D_XCTRACE_TEMPLATE"


def _lock_metal() -> tuple[str, bool]:
    """Force Induced GPU Performance State to Maximum (_apple_gpu_clock_lock.py)
    and point _bench.py's xctrace calls at the patched template via env var."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "_apple_gpu_clock_lock.py"), "Maximum"],
        capture_output=True,
        text=True,
    )
    path = r.stdout.strip()
    if r.returncode == 0 and path:
        os.environ[_XCTRACE_TEMPLATE_ENV] = path
        print(f"induced GPU performance state -> Maximum ({path})")
        return "induced-maximum", True
    _fatal_lock_failure(
        "Xcode/Instruments internals may have changed; see "
        f"scripts/_apple_gpu_clock_lock.py. stderr: {r.stderr.strip()[-500:]}"
    )


def unlock_clocks(be: Backend) -> None:
    if be.name == "cuda":
        for args in (["--reset-gpu-clocks"], ["--reset-memory-clocks"]):
            subprocess.run(["sudo", "-n", "nvidia-smi", *args], capture_output=True)
    elif be.name == "rocm":
        subprocess.run(
            ["sudo", "-n", "rocm-smi", "--setperflevel", "auto"], capture_output=True
        )


# --------------------------------------------------------------------------
# (b) Recompile (clear OUR JIT cache only) + correctness.
# --------------------------------------------------------------------------


def _smoke_k(dev: str) -> str:
    """Quick smoke: one representative test per function family on `dev`.

    Mirrors the old NVIDIA selector but keyed on the backend's device token.
    fp16 is constrained for accelerators (where it's the headline dtype); CPU
    drops it so the selector still matches (CPU kernels run mainly fp32).
    """
    fam = (
        "(test_contiguous and silu and with_bias) or "
        "(test_width_backward and with_bias) or "
        "(test_update_single_token and silu and with_bias)"
    )
    dtype = "" if dev == "cpu" else "fp16 and "
    return f"{dev} and {dtype}({fam})"


def correctness(be: Backend, tier: str, clean: bool) -> bool:
    section(f"(b) recompile + correctness ({tier} tier, {be.name})")
    if clean:
        cache = Path("~/.cache/causal_conv1d_mojo").expanduser()
        if VERBOSE:
            print(f"clearing JIT cache {cache} (keeping mojo compiler cache)")
        # Keep xctrace_templates/: step (a) just wrote the clock-locked
        # template there (see _apple_gpu_clock_lock.py) and step (c) needs
        # it; wiping the whole cache root out from under it breaks the bench.
        for child in cache.glob("*"):
            if child.name == "xctrace_templates":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    if tier == "quick":
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-x",
            "tests/test_fwd.py",
            "tests/test_bwd.py",
            "tests/test_update.py",
            "-k",
            _smoke_k(be.test_device),
        ]
    else:
        print("full regression suite (every landed feature, all devices/dtypes)")
        cmd = [sys.executable, "-m", "pytest", "-q"]
    ok = run(cmd).returncode == 0
    if not ok:
        Gate.fail(f"{tier} correctness failed")
    else:
        print("correctness OK")
    return ok


# --------------------------------------------------------------------------
# (c) Benchmark vs baseline.
#
# Returns the parsed per-shape report dicts so step (d) can reuse them (the
# Apple xctrace analysis in particular — we don't want to record twice).
# --------------------------------------------------------------------------


def bench_kernel(be: Backend, fn, dtype, shapes, runs, clock, baseline_flags) -> list:
    measure = be.kernel_measure
    label = "GPU kernel time" if measure == "kernel" else "wall-clock"
    if be.name == "metal":
        section("(c) absolute GPU kernel time (xctrace Metal System Trace)")
    elif be.baseline:
        section(f"(c) {label} bench vs {be.baseline} baseline")
    else:
        section(f"(c) {label} bench")
    reps = []
    for shape in shapes:
        cmd = [
            *taskset_prefix(),
            sys.executable,
            str(REPO / "scripts" / "_bench.py"),
            fn,
            "--shape",
            shape,
            "--dtype",
            dtype,
            "--device",
            be.device,
            "--impl",
            ",".join(be.kernel_impls),
            "--measure",
            measure,
            "--runs",
            str(runs),
            "--clock-locked",
            clock,
            "--json",
            *baseline_flags,
        ]
        r = run(cmd, capture=True)
        if r.stderr:
            sys.stderr.write(r.stderr)
        line = (r.stdout or "").strip().splitlines()
        if r.returncode != 0 or not line:
            Gate.fail(f"bench failed on {shape}")
            continue
        try:
            reps.append(json.loads(line[-1]))
        except json.JSONDecodeError:
            Gate.fail(f"bench produced no JSON on {shape}")
    if be.baseline is None:
        _aggregate_absolute(reps)
    else:
        _aggregate_ratio(be, reps)
    return reps


def _aggregate_ratio(be: Backend, rows: list[dict]) -> None:
    """mojo-vs-baseline table. Gates (fails) only when ``be.gate_ratio``."""
    if not rows:
        Gate.fail("no parseable bench results")
        return
    base = be.baseline
    rkey = f"mojo_over_{base}"
    print(
        f"\n  {'shape':>18} | {'mojo us':>9} | {base[:6] + ' us':>9} | "
        f"{'ratio':>7} | {'spread':>7} | verdict"
    )
    print("  " + "-" * 78)
    worst = 0.0
    for r in rows:
        res = r["results"]
        mojo = res.get("mojo", {}).get("min_us", math.nan)
        bu = res.get(base, {}).get("min_us", math.nan)
        ratio = r["ratio_min"].get(rkey, math.nan)
        # Trust gate uses only the two impls in the ratio — a noisy third
        # impl must not mask a real mojo-vs-baseline delta.
        spread = max(
            (
                res[i]["spread_pct"]
                for i in ("mojo", base)
                if i in res and res[i]["runs_us"]
            ),
            default=0.0,
        )
        if ratio != ratio:  # NaN — mojo and/or baseline produced no number
            errs = [
                f"{i}={res[i]['error']}"
                for i in ("mojo", base)
                if res.get(i, {}).get("error")
            ]
            verdict = "NO-BASELINE"
            Gate.fail(
                f"no usable mojo/{base} measurement on {tuple(r['shape'])}"
                + (f" ({'; '.join(errs)})" if errs else "")
            )
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
                if be.gate_ratio:
                    Gate.fail(
                        f"perf regression on {tuple(r['shape'])}: {ratio:.3f}x "
                        f"(gap {gap:.1f}% > spread {spread:.1f}%)"
                    )
        print(
            f"  {str(tuple(r['shape'])):>18} | {mojo:9.2f} | {bu:9.2f} | "
            f"{ratio:6.3f}x | {spread:6.1f}% | {verdict}"
        )
    print("  " + "-" * 78)
    if worst == 0.0:
        print(f"  no usable mojo/{base} comparison — see [FAIL] lines above")
    elif be.gate_ratio:
        tail = (
            "(within 3% across all shapes — PASS)"
            if worst <= 1.03
            else "(REVIEW: a shape exceeds 1.03x)"
        )
        print(f"  worst mojo/{base} ratio: {worst:.3f}x  {tail}")
    else:
        # No hand-tuned baseline here — the ratio is mojo-vs-naive-fallback,
        # reported for context, not gated.
        print(f"  worst mojo/{base} ratio: {worst:.3f}x  (informational, not a gate)")


def _aggregate_absolute(rows: list[dict]) -> None:
    """Apple: absolute per-kernel GPU time read back from the Metal trace."""
    if not rows:
        Gate.fail("no parseable bench results")
        return
    print(f"\n  {'shape':>18} | {'kernel us':>10} | {'clock':>8} | {'duty%':>6} | note")
    print("  " + "-" * 70)
    for r in rows:
        mojo = r["results"].get("mojo", {})
        us = mojo.get("min_us", math.nan)
        a = r.get("metal_analysis") or {}
        h = a.get("headline") or {}
        clock = h.get("clock", "?")
        duty = (a.get("duty") or {}).get("busy_pct", math.nan)
        note = ""
        if mojo.get("error"):
            note = mojo["error"]
        elif clock and clock != "Maximum":
            note = "DVFS-throttled (not steady state)"
        elif duty == duty and duty < 40.0:
            note = "launch/sync-bound (low GPU residency)"
        print(
            f"  {str(tuple(r['shape'])):>18} | {us:10.2f} | {clock:>8} | "
            f"{duty:6.1f} | {note}"
        )
    print("  " + "-" * 70)
    print("  absolute GPU time (mojo-only; upstream is CUDA-only — nothing to diff)")


# --------------------------------------------------------------------------
# (d) Deep profiler. Backend-specific tool; skips when unavailable.
# --------------------------------------------------------------------------

_NCU_METRICS = ",".join(
    [
        "gpu__time_duration.avg",
        "launch__waves_per_multiprocessor",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "smsp__inst_executed_op_shared_st.sum",
        "smsp__inst_executed_op_shared_ld.sum",
    ]
)


def profiler(be: Backend, fn, dtype, canon, skip_flag, reps) -> None:
    if skip_flag:
        skip("(d) deep profiler", "--skip-ncu")
        return
    if be.name == "cuda":
        _profiler_ncu(be, fn, dtype, canon)
    elif be.name == "metal":
        _profiler_metal(reps)
    elif be.name == "cpu":
        _profiler_perf(be, fn, dtype, canon)
    else:  # rocm
        skip(
            "(d) deep profiler",
            "rocprofv3 can't instrument Mojo's DeviceContext (HSA conflict); "
            "use scripts/tools/rocprof_pmc on --impl upstream for reference counters",
        )


def _ncu_cmd() -> list[str] | None:
    if shutil.which("ncu"):
        return ["ncu"]
    if shutil.which("pixi"):
        return ["pixi", "exec", "--spec", "nsight-compute", "--", "ncu"]
    return None


_NCU_LAUNCH_COUNT = 5


def _parse_ncu_csv(text: str) -> dict[str, dict] | None:
    """Group ncu's `--csv` output (one row per launch×metric) by metric name.

    Returns {metric_name: {"unit": str, "vals": [float, ...]}} with one value
    per profiled launch, or None if no parseable CSV header was found.
    """
    lines = text.splitlines()
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if '"Metric Name"' in ln and '"Metric Value"' in ln
        ),
        None,
    )
    if start is None:
        return None
    data: dict[str, dict] = {}
    for row in csv.DictReader(lines[start:]):
        name = (row.get("Metric Name") or "").strip()
        raw = (row.get("Metric Value") or "").strip()
        if not name or not raw:
            continue
        try:
            num = float(raw.replace(",", ""))  # ncu uses ',' as a thousands sep
        except ValueError:
            continue
        d = data.setdefault(
            name, {"unit": (row.get("Metric Unit") or "").strip(), "vals": []}
        )
        d["vals"].append(num)
    return data or None


def _print_ncu_summary(data: dict[str, dict]) -> None:
    """One row per metric: mean + spread across the profiled launches."""
    hdr = (
        f"    {'metric':<54} | {'unit':>9} | {'mean':>13} | "
        f"{'min':>13} | {'max':>13} | {'spread':>7}"
    )
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    # Stable order: the order we requested metrics in, then any extras ncu added.
    ordered = list(dict.fromkeys([*_NCU_METRICS.split(","), *data]))
    for name in ordered:
        d = data.get(name)
        if not d or not d["vals"]:
            continue
        vals = d["vals"]
        mean = sum(vals) / len(vals)
        lo, hi = min(vals), max(vals)
        spread = (hi - lo) / mean * 100.0 if mean else 0.0
        print(
            f"    {name:<54} | {d['unit']:>9} | {mean:13.3f} | "
            f"{lo:13.3f} | {hi:13.3f} | {spread:6.1f}%"
        )
    n = max((len(d["vals"]) for d in data.values()), default=0)
    print(f"    (mean + spread over {n} profiled launches)")


def _profiler_ncu(be: Backend, fn, dtype, canon) -> None:
    section("(d) ncu deep profiler")
    ncu = _ncu_cmd()
    if ncu is None:
        warn("ncu not found and pixi unavailable — skipping deep profiling.")
        warn("(the raw-mode driver '_bench.py ... --measure raw' is ready for it.)")
        return
    if VERBOSE:
        print(f"profiler: {' '.join(ncu)}")
    cmd = [
        *ncu,
        "--target-processes",
        "all",
        "--launch-skip",
        "10",
        "--launch-count",
        str(_NCU_LAUNCH_COUNT),
        "--metrics",
        _NCU_METRICS,
        "--csv",
        sys.executable,
        str(REPO / "scripts" / "_bench.py"),
        fn,
        "--shape",
        canon,
        "--dtype",
        dtype,
        "--impl",
        "mojo",
        "--measure",
        "raw",
        "--iters",
        "30",
        "--warmup",
        "10",
    ]
    # Capture stdout (the CSV) but let stderr stream so the ==PROF== per-launch
    # progress bars stay visible — ncu's replay passes are slow.
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        warn("ncu run failed (perf-counter perms? needs root/CAP_SYS_ADMIN)")
        if proc.stdout:
            print(proc.stdout)
        return
    data = _parse_ncu_csv(proc.stdout or "")
    if data is None:
        warn("could not parse ncu CSV — dumping raw output below.")
        print(proc.stdout)
        return
    _print_ncu_summary(data)


def _profiler_metal(reps: list[dict]) -> None:
    """Apple's deep-profiler step: surface the headless GPU stats already read
    back from the Metal System Trace in step (c) — per-encoder GPU time split
    by DVFS clock state, plus the device-global Active/Idle duty cycle. We do
    not re-record (a trace is expensive and the data is identical)."""
    section("(d) Metal GPU profile (xctrace stats from step c)")
    analyses = [r["metal_analysis"] for r in reps if r.get("metal_analysis")]
    if not analyses:
        warn("no Metal System Trace analysis captured in step (c) — nothing to show.")
        return
    for a in analyses:
        hdr = (
            f"    {'channel':>8} | {'clock':>8} | {'count':>5} | "
            f"{'median':>9} | {'min':>9} | {'max':>9} | encoder"
        )
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for g in a.get("groups", []):
            print(
                f"    {g['channel']:>8} | {g['clock']:>8} | {g['count']:>5} | "
                f"{g['median_us']:8.2f}u | {g['min_us']:8.2f}u | "
                f"{g['max_us']:8.2f}u | {g['label']}"
            )
        d = a.get("duty")
        if d:
            print(
                f"    GPU duty cycle (device-global): {d['busy_pct']:.1f}% active "
                f"({d['active_ms']:.2f} ms active / {d['idle_ms']:.2f} ms idle)"
            )
            if d["busy_pct"] < 40.0:
                print("      -> low residency: launch/sync-bound, not compute-bound.")
        print()
    print(
        "    NOTE: occupancy / ALU% / bandwidth / stalls are GUI-only on Apple — "
        "open the trace in Instruments for those."
    )


def _profiler_perf(be: Backend, fn, dtype, canon) -> None:
    section("(d) perf stat (CPU hardware counters)")
    if not shutil.which("perf"):
        print("skipped — `perf` not found (linux perf tools); CPU has no GPU profiler.")
        return
    cmd = [
        "perf",
        "stat",
        "-d",
        sys.executable,
        str(REPO / "scripts" / "_bench.py"),
        fn,
        "--shape",
        canon,
        "--dtype",
        dtype,
        "--device",
        "cpu",
        "--impl",
        "mojo",
        "--measure",
        "raw",
        "--iters",
        "30",
        "--warmup",
        "10",
    ]
    if run(cmd).returncode != 0:
        warn("perf stat run failed (perf_event_paranoid? needs relaxed perms)")


# --------------------------------------------------------------------------
# (e) dump GPU asm  +  (f) histogram  +  (g) spill canary.
#
# The NVIDIA path is the full PTX -> SASS -> histogram-vs-upstream -> spill
# pipeline (via _asm_tools.py). ROCm/Metal can dump the device ISA for the
# record but have no ptxas/nvdisasm/upstream-SASS counterpart, so (f)/(g)
# skip. CPU has no GPU asm at all.
# --------------------------------------------------------------------------


def assembly(be: Backend, fn, dtype, canon, refresh_reference) -> None:
    if be.name == "cuda":
        _assembly_nvidia(be, fn, dtype, canon, refresh_reference)
    elif be.name == "rocm":
        _assembly_dump_only(be, fn, dtype, canon)
    elif be.name == "metal":
        # compile_function's dump_asm (our DUMP_ASSEMBLY_INTO) emits textual
        # ISA only for PTX / AMDGPU targets; Metal lowers straight to a
        # metallib with no textual ISA dump, so there is nothing to write
        # (and occupancy is GUI-only).
        skip(
            "(e/f/g) GPU asm",
            "Mojo emits no textual Metal ISA (metallib only); occupancy is GUI-only",
        )
    else:  # cpu
        skip("(e/f/g) GPU asm", "CPU build emits no GPU device code")


def _dump_ptx(be: Backend, fn, dtype, canon, asm_dir: Path) -> Path | None:
    """Run one kernel call with CAUSAL_CONV1D_DUMP_ASM set so the Mojo build
    writes the device asm for this variant. Returns the newest dumped file."""
    for stale in asm_dir.glob(f"{SUBPKG[fn]}__*.ptx"):
        stale.unlink()
    env = os.environ.copy()
    env["CAUSAL_CONV1D_DUMP_ASM"] = str(asm_dir)
    dump = run(
        [
            sys.executable,
            str(REPO / "scripts" / "_bench.py"),
            fn,
            "--shape",
            canon,
            "--dtype",
            dtype,
            "--device",
            be.device,
            "--impl",
            "mojo",
            "--measure",
            "raw",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--runs",
            "1",
        ],
        env=env,
        capture=True,
    )
    ptxs = sorted(asm_dir.glob(f"{SUBPKG[fn]}__*.ptx"), key=lambda p: p.stat().st_mtime)
    if dump.returncode != 0 or not ptxs:
        if dump.stderr:
            sys.stderr.write(dump.stderr)
        return None
    return ptxs[-1]


def _ensure_upstream_csrc() -> Path | None:
    """Shallow-clone Tri Dao's causal-conv1d at UPSTREAM_REF (cached) and return
    its csrc/ dir — the source we compile to PTX. None if git/clone fails."""
    dst = Path("~/.cache/causal_conv1d_mojo/upstream_src").expanduser() / UPSTREAM_REF
    csrc = dst / "csrc"
    if csrc.is_dir():
        return csrc
    if not shutil.which("git"):
        warn("git not found — cannot fetch upstream source for the PTX histogram")
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = run(
        [
            "git", "clone", "--depth", "1", "--branch", UPSTREAM_REF,
            "https://github.com/Dao-AILab/causal-conv1d.git", str(dst),
        ]
    )  # fmt: skip
    if r.returncode != 0 or not csrc.is_dir():
        warn(f"failed to clone upstream {UPSTREAM_REF}; skipping PTX histogram")
        return None
    return csrc


def _assembly_nvidia(be: Backend, fn, dtype, canon, refresh_reference) -> None:
    asm_dir = REPO / "scripts" / "assembly" / "nvidia"
    ref_dir = REPO / "scripts" / "reference_assembly" / "nvidia"
    asm_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    tools = str(REPO / "scripts" / "_asm_tools.py")

    section("(e) dump our PTX/SASS -> scripts/assembly/nvidia/")
    ptx_path = _dump_ptx(be, fn, dtype, canon, asm_dir)
    if ptx_path is None:
        Gate.fail(f"PTX dump failed for {fn}")
        return
    ptx = str(ptx_path)
    print(f"our PTX: {ptx}", flush=True)
    our_sass = str(asm_dir / f"{fn}.sass")
    if (
        run(
            [sys.executable, tools, "sass", ptx, our_sass, "--arch", be.arch_a]
        ).returncode
        != 0
    ):
        Gate.fail("PTX->SASS failed")

    section("(g) ptxas -v spill / regalloc canary")
    if (
        run(
            [
                sys.executable,
                tools,
                "spill",
                ptx,
                "--arch",
                be.arch_a,
                "--max-spill",
                "0",
            ]
        ).returncode
        != 0
    ):
        Gate.fail("spill canary: register spills detected")

    section("(f) instruction-mix histogram vs upstream reference")
    ref_sass = ref_dir / f"{fn}.sass"
    if refresh_reference or not ref_sass.exists():
        print(f"extracting upstream {fn} reference SASS", flush=True)
        cmd = [
            sys.executable,
            tools,
            "upstream-sass",
            fn,
            str(ref_sass),
            "--arch",
            be.arch,
        ]
        for m in REF_MATCH[fn]:
            cmd += ["--match", m]
        if run(cmd).returncode != 0:
            warn(f"could not extract upstream reference ({fn}); skipping histogram")
    if ref_sass.exists():
        print("SASS-level (ours vs upstream cubin):", flush=True)
        if (
            run(
                [sys.executable, tools, "histogram", our_sass, str(ref_sass)]
            ).returncode
            != 0
        ):
            warn("SASS histogram diff failed")

    # PTX-level histogram. The shipped upstream .so is cubin-only, so we
    # compile the .cu to PTX ourselves (nvcc via pixi) for a higher-level diff.
    ref_ptx = ref_dir / f"{fn}.ptx"
    if refresh_reference or not ref_ptx.exists():
        csrc = _ensure_upstream_csrc()
        if csrc is not None:
            print(f"compiling upstream {fn} reference PTX (nvcc via pixi)", flush=True)
            cmd = [
                sys.executable, tools, "upstream-ptx", fn, str(ref_ptx),
                "--src-dir", str(csrc), "--arch", be.arch_a,
            ]  # fmt: skip
            for m in REF_MATCH[fn]:
                cmd += ["--match", m]
            if run(cmd).returncode != 0:
                warn(f"could not compile upstream PTX ({fn}); skipping PTX histogram")
    if ref_ptx.exists():
        print("PTX-level (ours vs upstream source):", flush=True)
        if (
            run(
                [
                    sys.executable,
                    tools,
                    "histogram",
                    ptx,
                    str(ref_ptx),
                    "--format",
                    "ptx",
                ]
            ).returncode
            != 0
        ):
            warn("PTX histogram diff failed")


def _assembly_dump_only(be: Backend, fn, dtype, canon) -> None:
    """ROCm: dump the AMDGPU ISA for the record; (f)/(g) have no counterpart
    (no ptxas/nvdisasm/cuobjdump, and no upstream ROCm kernel to diff)."""
    asm_dir = REPO / "scripts" / "assembly" / be.name
    asm_dir.mkdir(parents=True, exist_ok=True)
    section(f"(e) dump our GCN ISA -> scripts/assembly/{be.name}/")
    dumped = _dump_ptx(be, fn, dtype, canon, asm_dir)
    if dumped is None:
        warn(f"GCN ISA dump failed for {fn}")
    else:
        # The dump hook always uses a .ptx extension; rename so the file name
        # isn't misleading about its contents.
        target = dumped.with_suffix(".gcn")
        dumped.replace(target)
        print(f"dumped GCN ISA -> {target}")
    skip(
        "(f/g) histogram + spill canary",
        "no ptxas/nvdisasm for ROCm and no upstream ROCm kernel to diff",
    )


# --------------------------------------------------------------------------
# (h) Independent wall-clock run (torch.utils.benchmark).
# --------------------------------------------------------------------------


def walltime(be: Backend, fn, dtype, canon, runs, clock, baseline_flags) -> None:
    section("(h) end-to-end wall-clock (torch.utils.benchmark, auto sync)")
    cmd = [
        *taskset_prefix(),
        sys.executable,
        str(REPO / "scripts" / "_bench.py"),
        fn,
        "--shape",
        canon,
        "--dtype",
        dtype,
        "--device",
        be.device,
        "--impl",
        ",".join(be.walltime_impls),
        "--measure",
        "walltime",
        "--runs",
        str(runs),
        "--clock-locked",
        clock,
        *baseline_flags,
    ]
    if run(cmd).returncode != 0:
        warn("walltime run failed")


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--full", action="store_true", help="full tier (task gate)")
    p.add_argument("--fn", choices=("fwd", "bwd", "update", "all"), default="fwd")
    p.add_argument(
        "--dtype",
        default=None,
        help="default: fp16 on a GPU backend, fp32 on cpu (no fp16 conv on CPU)",
    )
    p.add_argument(
        "--backend",
        choices=("cuda", "rocm", "metal", "cpu"),
        default=None,
        help="force a backend (default: auto-detect)",
    )
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument(
        "--skip-correctness",
        action="store_true",
        help="skip the (b) correctness gate (e.g. to profile a WIP kernel)",
    )
    p.add_argument("--skip-ncu", action="store_true", help="skip the (d) profiler step")
    p.add_argument("--skip-asm", action="store_true")
    p.add_argument("--refresh-baseline", action="store_true")
    p.add_argument("--refresh-reference", action="store_true")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="echo every spawned subprocess command line (the `$ ...` lines)",
    )
    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    os.chdir(REPO)
    be = detect_backend(args.backend)
    # CPU torch has no fp16 conv1d, so fall back to fp32 there by default.
    dtype = args.dtype or ("fp32" if be.name == "cpu" else "fp16")
    tier = "full" if args.full else "quick"
    fns = ["fwd", "bwd", "update"] if args.fn == "all" else [args.fn]
    runs = 5 if args.full else 3
    # Full tier re-seeds the baseline (authoritative); quick reuses the cache.
    baseline_flags = (
        ["--refresh-baseline"] if (args.full or args.refresh_baseline) else []
    )

    if be.arch_a:
        arch = f" arch={be.arch}/{be.arch_a}"
    elif be.arch:
        arch = f" arch={be.arch}"
    else:
        arch = ""
    print(
        f"master_bench: backend={be.name} fn={args.fn} tier={tier} "
        f"dtype={dtype} device='{be.pretty}'{arch}"
    )

    clock, locked = lock_clocks(be, not args.no_lock)
    if locked:
        # SIGINT/exceptions hit the finally below, but SIGTERM/SIGHUP (CI
        # cancellation, terminal close) would otherwise leave clocks locked.
        def _on_signal(signum, _frame):
            unlock_clocks(be)
            os._exit(128 + signum)

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)
    try:
        # (b) correctness is tier-based and covers every function family.
        if args.skip_correctness:
            skip("(b) correctness", "--skip-correctness")
        elif not correctness(be, tier, not args.no_clean):
            return 1
        for fn in fns:
            sh = SHAPES[fn]
            shapes = [sh["canon"]] if tier == "quick" else sh["full"]
            if len(fns) > 1:
                section(f"### function: {fn} ###")
            reps = bench_kernel(be, fn, dtype, shapes, runs, clock, baseline_flags)
            profiler(be, fn, dtype, sh["canon"], args.skip_ncu, reps)
            if not args.skip_asm:
                assembly(be, fn, dtype, sh["canon"], args.refresh_reference)
            else:
                skip("(e/f/g) assembly", "--skip-asm")
            walltime(be, fn, dtype, sh["canon"], runs, clock, baseline_flags)
    finally:
        if locked:
            unlock_clocks(be)

    section("summary")
    if Gate.failed:
        print(f"ISSUES — one or more gates failed above (fn={args.fn} tier={tier})")
        return 1
    print(f"PASS — all gates green (backend={be.name} fn={args.fn} tier={tier})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
