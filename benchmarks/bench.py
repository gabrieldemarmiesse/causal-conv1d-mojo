"""Unified benchmark driver for causal_conv1d_mojo.

One CLI for every function (`fwd` / `bwd` / `update`), every input
shape, every function-argument flag, against every implementation
(`mojo` / `upstream` / `pytorch`), measured three *independent* ways:

  * ``--measure kernel``   — per-kernel GPU time via ``torch.profiler``
                             (CUPTI). The headline "GPU kernel time vs
                             upstream" signal.
  * ``--measure walltime`` — end-to-end wall-clock via
                             ``torch.utils.benchmark`` (auto CPU<->GPU
                             sync). Captures Python + launch overhead.
  * ``--measure raw``      — a tight, synchronized loop with *no*
                             profiler attached, so an external profiler
                             (``ncu`` / NSight Compute) can wrap the
                             whole process and attribute clean metrics.

Apple silicon (``--device mps``)
--------------------------------
Metal has no torch device-time hook (no CUPTI/rocprof), so
``--measure kernel`` on ``mps`` can't read per-kernel time in-process.
Instead the driver *orchestrates Instruments itself*: it pre-warms the
JIT cache, records a "Metal System Trace" with ``xctrace`` around a
re-launch of *this same script* as the traced workload (the timed loop
is bracketed by ``torch.mps.profiler``), then parses the
``metal-gpu-intervals`` table back out and prints per-encoder GPU time
split by GPU clock state — the Instruments findings land in stdout
automatically. ``--measure walltime`` / ``raw`` run in-process on mps
just like cuda (using ``torch.mps.synchronize``). Upstream Tri Dao is
CUDA-only, so mps comparisons are mojo-only.

This file replaces the old pile of ``bench_*.py`` scripts (including the
Apple-only ``bench_metal_gpu.py`` + ``scripts/xctrace_bench.sh`` +
``scripts/xctrace_gpu_intervals.py``, now folded in here): every shape
is one invocation, every comparison is a flag. The
``scripts/master_bench_nvidia.py`` orchestrator drives it across
shapes/tiers and adds clock-locking, correctness gating, ncu, and
assembly diffing.

Examples
--------
    # mojo-only fwd kernel time on one shape (fast inner loop)
    python benchmarks/bench.py fwd --shape 1,4096,2048,4 --impl mojo

    # mojo vs upstream vs pytorch, kernel time, 5 runs, JSON for tooling
    python benchmarks/bench.py fwd --shape 1,1024,2048,4 \
        --impl all --runs 5 --json

    # end-to-end wall-clock (torch.utils.benchmark) for the update kernel
    python benchmarks/bench.py update --shape 16,2048 --measure walltime

    # raw loop for ncu to wrap (no profiler overhead)
    python benchmarks/bench.py bwd --shape 4,4096,2048,4 \
        --impl mojo --measure raw
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

import causal_conv1d_mojo
from causal_conv1d_mojo.reference import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)

# ---------------------------------------------------------------------------
# Implementations.
#
# `mojo`     — our native Mojo kernels (the moving target).
# `upstream` — Tri Dao's hand-tuned CUDA kernels (the baseline). Imported
#              lazily so `--impl mojo` works without the `nvidia` extra.
# `pytorch`  — the pure-PyTorch reference (F.conv1d + F.silu), the
#              fallback you'd write with no custom op at all.
# ---------------------------------------------------------------------------

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ---------------------------------------------------------------------------
# Device helpers. One place that knows how to pick a device, how to drain it,
# and (on Apple) how to name the SoC for the env signature.
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> str:
    """`auto` -> cuda if present, else mps (Apple), else cpu; else verbatim."""
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> Callable[[], None]:
    """The right device-drain for `device` (no-op on cpu)."""
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def _mac_chip() -> str:
    """Best-effort Apple SoC brand for the env signature (e.g. 'Apple M2 Pro').

    On Apple silicon the GPU is integrated, so the CPU brand identifies the
    GPU too. Falls back to a generic tag if sysctl is unavailable.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return f"mps:{out.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    return "mps"


# Set in the child process that `xctrace --launch`es as the traced workload,
# so it runs the bare `--measure raw` loop instead of recursively orchestrating
# another trace.
_TRACED_ENV = "CAUSAL_CONV1D_BENCH_TRACED"


def _is_traced_workload() -> bool:
    return os.environ.get(_TRACED_ENV) == "1"


def _upstream_module():
    try:
        import causal_conv1d  # noqa: PLC0415

        return causal_conv1d
    except ImportError as e:  # pragma: no cover - depends on extras
        raise SystemExit(
            "the `upstream` implementation needs the Tri Dao causal-conv1d "
            "wheel; run benches with `uv run --extra nvidia ...`"
        ) from e


# ---------------------------------------------------------------------------
# Kernel-name classifiers for `--measure kernel`.
#
# Mojo's `mojo build` mangles comptime params into the kernel name
# (e.g. `..._fwd_kernel_..._<hash>`); upstream CUDA kernels are
# `void causal_conv1d_<fn>_kernel<...>`. We attribute profiler CUDA
# events to an impl by matching its name. The pure-PyTorch path has no
# single named kernel, so it sums *every* CUDA event in the profiled
# region (it is the only thing running there).
# ---------------------------------------------------------------------------


def _mojo_classifier(fn: str) -> Callable[[str], bool]:
    if fn == "fwd":
        return lambda n: "fwd_kernel" in n and not n.startswith("void")
    if fn == "bwd":
        return lambda n: "bwd" in n and "kernel" in n and not n.startswith("void")
    if fn == "update":
        return lambda n: "update_kernel" in n and not n.startswith("void")
    raise ValueError(fn)


def _upstream_classifier(fn: str) -> Callable[[str], bool]:
    # Matches both the standard `void causal_conv1d_<fn>_kernel` and the
    # channel-last `void causal_conv1d_channellast_<fn>_kernel` (upstream
    # dispatches to the latter for channel-last x).
    needle = f"{fn}_kernel"
    return lambda n: n.startswith("void causal_conv1d_") and needle in n


def _classifier(impl: str, fn: str) -> Callable[[str], bool] | None:
    if impl == "mojo":
        return _mojo_classifier(fn)
    if impl == "upstream":
        return _upstream_classifier(fn)
    # pytorch: sum every CUDA event (None sentinel = "match all").
    return None


def _parse_impls(spec: str) -> list[str]:
    """`all` -> all three; `both` (legacy) -> mojo+upstream; else comma list."""
    if spec == "all":
        return ["mojo", "upstream", "pytorch"]
    if spec == "both":  # legacy alias from the old bench CLI / tools wrappers
        return ["mojo", "upstream"]
    return [s for s in spec.split(",") if s]


# ---------------------------------------------------------------------------
# Input construction.
# ---------------------------------------------------------------------------


@dataclass
class Config:
    fn: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    width: int
    state_len: int
    activation: str | None
    has_bias: bool
    has_seq_idx: bool
    has_initial_states: bool
    return_final_states: bool
    has_cache_seqlens: bool
    has_conv_state_indices: bool
    channel_last: bool = False
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Stable, JSON-serialisable config used as a cache key component."""
        return {
            "fn": self.fn,
            "dtype": self.dtype,
            "device": self.device,
            "width": self.width,
            "state_len": self.state_len,
            "activation": self.activation or "none",
            "bias": self.has_bias,
            "seq_idx": self.has_seq_idx,
            "initial_states": self.has_initial_states,
            "return_final_states": self.return_final_states,
            "cache_seqlens": self.has_cache_seqlens,
            "conv_state_indices": self.has_conv_state_indices,
            "channel_last": self.channel_last,
        }


def _seq_idx_tensor(batch: int, seqlen: int, device) -> torch.Tensor:
    """Two contiguous segments per row (a realistic packed-sequence layout)."""
    idx = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)
    if seqlen > 1:
        idx[:, seqlen // 2 :] = 1
    return idx


def _channel_last(t: torch.Tensor) -> torch.Tensor:
    """(B, D, L)-shaped tensor with the channel dim contiguous (stride(1)==1).

    Upstream's seq_idx / initial_states kernels are channel-last only;
    our Mojo kernels handle either layout. Build by laying out (B, L, D)
    contiguously and transposing the last two axes.
    """
    b, d, l = t.shape
    return t.transpose(1, 2).contiguous().transpose(1, 2)


def _make_fwd_inputs(cfg: Config) -> dict[str, torch.Tensor | None]:
    b, d, l, w = cfg.shape
    dtype = _DTYPES[cfg.dtype]
    dev = torch.device(cfg.device)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    x = torch.randn(b, d, l, generator=g).to(dev, dtype)
    initial_states = (
        torch.randn(b, d, w - 1, generator=g).to(dev, dtype)
        if cfg.has_initial_states
        else None
    )
    if cfg.channel_last:
        x = _channel_last(x)
        if initial_states is not None:
            initial_states = _channel_last(initial_states)
    out = {
        "x": x,
        "weight": torch.randn(d, w, generator=g).to(dev, dtype),
        "bias": torch.randn(d, generator=g).to(dev, dtype) if cfg.has_bias else None,
        "seq_idx": _seq_idx_tensor(b, l, dev) if cfg.has_seq_idx else None,
        "initial_states": initial_states,
    }
    return out


def _make_update_inputs(cfg: Config) -> dict[str, torch.Tensor | None]:
    b, d = cfg.shape[0], cfg.shape[1]
    w, state_len = cfg.width, cfg.state_len
    dtype = _DTYPES[cfg.dtype]
    dev = torch.device(cfg.device)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    out = {
        "x": torch.randn(b, d, generator=g).to(dev, dtype),
        "conv_state": torch.randn(b, d, state_len, generator=g).to(dev, dtype),
        "weight": torch.randn(d, w, generator=g).to(dev, dtype),
        "bias": torch.randn(d, generator=g).to(dev, dtype) if cfg.has_bias else None,
        "cache_seqlens": (
            torch.zeros(b, dtype=torch.int32, device=dev)
            if cfg.has_cache_seqlens
            else None
        ),
        "conv_state_indices": (
            torch.arange(b, dtype=torch.int32, device=dev)
            if cfg.has_conv_state_indices
            else None
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Per-(impl, fn) callables. Each returns a 0-arg closure that performs one
# unit of work (forward, fwd+backward, or update).
# ---------------------------------------------------------------------------


def _supports(impl: str, cfg: Config) -> bool:
    """Which (impl, config) combos are runnable on this device.

    - The upstream Tri Dao wheel is CUDA-only, so it can't run on mps/cpu.
    - The pure-PyTorch references don't cover packed sequences / paged
      caches; skip those combinations rather than crash.
    """
    if impl == "upstream" and cfg.device != "cuda":
        return False
    if impl != "pytorch":
        return True
    if cfg.has_seq_idx or cfg.has_conv_state_indices:
        return False
    return True


def _fwd_callable(impl: str, cfg: Config) -> Callable[[], Any]:
    t = _make_fwd_inputs(cfg)
    kw = dict(
        bias=t["bias"],
        seq_idx=t["seq_idx"],
        initial_states=t["initial_states"],
        return_final_states=cfg.return_final_states,
        activation=cfg.activation,
    )
    if impl == "mojo":
        fn = causal_conv1d_mojo.causal_conv1d_fn
    elif impl == "upstream":
        fn = _upstream_module().causal_conv1d_fn
    else:  # pytorch
        fn = causal_conv1d_ref
        kw.pop("seq_idx")  # ref has no seq_idx
    return lambda: fn(t["x"], t["weight"], **kw)


def _bwd_callable(impl: str, cfg: Config) -> Callable[[], Any]:
    """fwd + backward, rebuilding the autograd graph each call."""
    t = _make_fwd_inputs(cfg)
    b, d, l, w = cfg.shape
    dout = torch.randn(
        b, d, l, generator=torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    ).to(torch.device(cfg.device), _DTYPES[cfg.dtype])

    if impl == "mojo":
        fwd = causal_conv1d_mojo.causal_conv1d_fn
    elif impl == "upstream":
        fwd = _upstream_module().causal_conv1d_fn
    else:
        fwd = causal_conv1d_ref

    use_seq_idx = cfg.has_seq_idx and impl != "pytorch"

    if cfg.device == "mps":
        # Apple trace path: Mojo doesn't label its Metal encoders, so a
        # fwd+bwd-per-call closure would collapse both kernels' Compute
        # encoders into one trace group, blending two kernels' times. Build
        # the autograd graph ONCE and re-run only backward each call (via
        # torch.autograd.grad, not .backward(), so no AccumulateGrad nodes
        # fire and add stray elementwise kernels). This leaves the traced
        # Compute Command attributable to the bwd kernel.
        x_ = t["x"].detach().requires_grad_()
        w_ = t["weight"].detach().requires_grad_()
        b_ = t["bias"].detach().requires_grad_() if t["bias"] is not None else None
        kw = dict(bias=b_, initial_states=t["initial_states"], activation=cfg.activation)
        if use_seq_idx:
            kw["seq_idx"] = t["seq_idx"]
        out = fwd(x_, w_, **kw)
        if isinstance(out, tuple):
            out = out[0]
        inputs = [v for v in (x_, w_, b_) if v is not None]
        return lambda: torch.autograd.grad(out, inputs, dout, retain_graph=True)

    # cuda/cpu: rebuild the graph each call and run the full fwd+bwd. The
    # profiler classifier isolates the bwd kernel by name, so blending the
    # fwd kernel into the same call is harmless here.
    def call():
        x_ = t["x"].detach().requires_grad_()
        w_ = t["weight"].detach().requires_grad_()
        b_ = t["bias"].detach().requires_grad_() if t["bias"] is not None else None
        kw = dict(bias=b_, initial_states=t["initial_states"], activation=cfg.activation)
        if use_seq_idx:
            kw["seq_idx"] = t["seq_idx"]
        out = fwd(x_, w_, **kw)
        if isinstance(out, tuple):
            out = out[0]
        out.backward(dout)

    return call


def _update_callable(impl: str, cfg: Config) -> Callable[[], Any]:
    t = _make_update_inputs(cfg)
    kw = dict(bias=t["bias"], activation=cfg.activation, cache_seqlens=t["cache_seqlens"])
    if impl == "mojo":
        fn = causal_conv1d_mojo.causal_conv1d_update
        kw["conv_state_indices"] = t["conv_state_indices"]
    elif impl == "upstream":
        fn = _upstream_module().causal_conv1d_update
        kw["conv_state_indices"] = t["conv_state_indices"]
    else:  # pytorch ref: no conv_state_indices
        fn = causal_conv1d_update_ref
    return lambda: fn(t["x"], t["conv_state"], t["weight"], **kw)


def make_callable(impl: str, cfg: Config) -> Callable[[], Any]:
    if cfg.fn == "fwd":
        return _fwd_callable(impl, cfg)
    if cfg.fn == "bwd":
        return _bwd_callable(impl, cfg)
    if cfg.fn == "update":
        return _update_callable(impl, cfg)
    raise ValueError(cfg.fn)


# ---------------------------------------------------------------------------
# Measurement back-ends. Each returns microseconds-per-call (one number per
# call to the function; for kernel mode that is summed GPU time).
# ---------------------------------------------------------------------------


def _mps_profiler_ctx(device: str):
    """Bracket the timed loop in `torch.mps.profiler` when we are the traced
    Apple workload, so the Metal System Trace gets OS-signpost-scoped
    intervals around the benchmark. No-op everywhere else.
    """
    if device == "mps" and _is_traced_workload():
        try:
            import torch.mps.profiler as mps_profiler  # noqa: PLC0415

            return mps_profiler.profile(mode="interval", wait_until_completed=False)
        except Exception:  # noqa: BLE001 — profiler is best-effort scoping
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def measure_kernel(
    call: Callable[[], Any],
    classify: Callable[[str], bool] | None,
    *,
    iters: int,
    warmup: int,
    device: str,
) -> float:
    """Mean per-call GPU time (µs) of the kernels matched by `classify`.

    `classify is None` sums every CUDA event (the pure-PyTorch path).
    """
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    sync = _sync(device)
    for _ in range(warmup):
        call()
    sync()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            call()
        sync()
    total = 0.0
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if classify is None or classify(evt.name):
            total += evt.self_device_time_total
    return total / iters


def measure_walltime(
    call: Callable[[], Any],
    *,
    min_run_time: float = 0.5,
    warmup: int = 10,
    device: str = "cuda",
) -> float:
    """End-to-end per-call wall-clock (µs) via torch.utils.benchmark.

    `Timer.blocked_autorange` handles CPU<->GPU synchronization and picks
    an iteration count that amortises timer overhead. We warm up first so
    a cold-cache JIT compile (~1 s on the first call) or one-time per-impl
    setup never lands inside the timed window.
    """
    import torch.utils.benchmark as tbench  # noqa: PLC0415

    for _ in range(warmup):
        call()
    _sync(device)()
    timer = tbench.Timer(stmt="_call()", globals={"_call": call})
    m = timer.blocked_autorange(min_run_time=min_run_time)
    return m.median * 1e6


def measure_raw(
    call: Callable[[], Any], *, iters: int, warmup: int, device: str
) -> float:
    """Tight synchronized loop, *no* profiler — for ncu/NSight to wrap.

    Returns a coarse wall-clock per-call number (not the measurement of
    record; the external profiler's per-kernel metrics are). The point is
    to drive the kernel with zero profiling overhead in-process.

    This is also the mode the Apple `xctrace` orchestrator re-launches as
    its traced workload (`--device mps --measure raw`); on mps the timed
    loop is bracketed by `torch.mps.profiler` (see `_mps_profiler_ctx`).
    """
    sync = _sync(device)
    for _ in range(warmup):
        call()
    sync()
    with _mps_profiler_ctx(device):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            call()
        sync()
    return (time.perf_counter_ns() - t0) / iters / 1e3


# ---------------------------------------------------------------------------
# Baseline cache.
#
# The mojo kernel is the moving target; upstream + pytorch are stable on a
# pinned baseline with locked clocks, so we measure them once and reuse.
# Key: (baseline_tag, fn, shape, config, measure, env signature). The env
# signature folds in the GPU name and the clock-lock state, so unlocking
# the GPU or moving to different hardware auto-invalidates the cache (same
# discipline as the JIT cache). Stored under benchmarks/baselines/ (which
# is .gitignored — these numbers are machine + clock specific).
# ---------------------------------------------------------------------------

_BASELINE_DIR = Path(__file__).resolve().parent / "baselines"


@dataclass
class Measurement:
    runs_us: list[float] = field(default_factory=list)
    from_cache: bool = False
    error: str | None = None

    @property
    def min_us(self) -> float:
        return min(self.runs_us) if self.runs_us else float("nan")

    @property
    def spread_pct(self) -> float:
        if len(self.runs_us) < 2 or self.min_us == 0:
            return 0.0
        return (max(self.runs_us) - min(self.runs_us)) / self.min_us * 100.0


class BaselineCache:
    """Memoizes stable baseline (upstream/pytorch) measurements to JSON.

    The version tag is keyed *per impl*: the `upstream` records carry the
    upstream wheel version (so an upstream bump invalidates them), while
    `pytorch` records carry a stable constant — otherwise a pytorch entry
    measured during an `--impl all` run (tag=upstream version) would miss
    on a later `--impl mojo,pytorch` run (no upstream → different tag).
    """

    def __init__(self, *, fn: str, measure: str, env_sig: dict[str, str], tags: dict[str, str]):
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _BASELINE_DIR / f"{fn}_{measure}.json"
        self.env_sig = env_sig
        self.tags = tags
        self._records: list[dict] = []
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text()).get("records", [])
            except json.JSONDecodeError:
                self._records = []

    def _key(self, impl: str, shape: tuple, config: dict) -> tuple:
        return (
            self.tags.get(impl, "n/a"),
            impl,
            tuple(shape),
            tuple(sorted(config.items())),
            tuple(sorted(self.env_sig.items())),
        )

    def get(self, impl: str, shape: tuple, config: dict) -> Measurement | None:
        key = self._key(impl, shape, config)
        for r in self._records:
            if tuple(r["key"]) == _jsonable(key):
                m = Measurement(runs_us=r["runs_us"], from_cache=True)
                return m
        return None

    def put(self, impl: str, shape: tuple, config: dict, m: Measurement) -> None:
        key = _jsonable(self._key(impl, shape, config))
        self._records = [r for r in self._records if tuple(r["key"]) != key]
        self._records.append({"key": list(key), "runs_us": m.runs_us})
        self.path.write_text(
            json.dumps({"records": self._records}, indent=2) + "\n"
        )


def _jsonable(key: tuple) -> tuple:
    """Round-trip a key through JSON-native types so cached and live keys
    compare equal (JSON turns tuples into lists, ints stay ints)."""
    return tuple(json.loads(json.dumps(list(key))))


# ---------------------------------------------------------------------------
# Apple-silicon kernel time via xctrace "Metal System Trace".
#
# Metal has no torch device-time hook, so `--measure kernel` on mps can't
# read per-kernel GPU time in-process the way CUPTI does on cuda. Instead we
# orchestrate Instruments: record a Metal System Trace around a re-launch of
# this script as the traced workload, then read the per-encoder GPU intervals
# back out of the `metal-gpu-intervals` table. This folds in what used to be
# `bench_metal_gpu.py` + `scripts/xctrace_bench.sh` +
# `scripts/xctrace_gpu_intervals.py`.
# ---------------------------------------------------------------------------

_XCTRACE_SCHEMA = "metal-gpu-intervals"
_CLOCK_SCHEMA = "gpu-performance-state-intervals"


def _shape_str(shape: tuple[int, ...]) -> str:
    return ",".join(str(x) for x in shape)


def _workload_argv(cfg: Config, args, *, iters: int, warmup: int) -> list[str]:
    """Argv that re-launches *this* script as the mojo-only mps workload.

    `--measure raw` is a bare synchronized loop (bracketed by
    `torch.mps.profiler` when the traced env var is set); it's the thing
    xctrace records, and also what we run un-traced to pre-warm the JIT
    cache so `mojo build` never lands inside the trace.
    """
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        cfg.fn,
        "--device", "mps",
        "--measure", "raw",
        "--impl", "mojo",
        "--dtype", cfg.dtype,
        "--width", str(cfg.width),
        "--activation", cfg.activation or "none",
        "--iters", str(iters),
        "--warmup", str(warmup),
        "--runs", "1",
        "--seed", str(cfg.seed),
        "--shape", _shape_str(cfg.shape),
    ]
    if not cfg.has_bias:
        argv.append("--no-bias")
    if cfg.has_seq_idx:
        argv.append("--seq-idx")
    if cfg.has_initial_states:
        argv.append("--initial-states")
    if cfg.return_final_states:
        argv.append("--return-final-states")
    if cfg.channel_last:
        argv.append("--channel-last")
    if cfg.has_cache_seqlens:
        argv.append("--cache-seqlens")
    if cfg.has_conv_state_indices:
        argv.append("--conv-state-indices")
    if cfg.fn == "update" and cfg.state_len:
        argv += ["--state-len", str(cfg.state_len)]
    return argv


def _xctrace_export(trace: str, schema: str) -> bytes:
    """`xctrace export` for one trace table; returns XML (raises on failure)."""
    out = subprocess.run(
        [
            "xctrace", "export", "--input", trace,
            "--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        ],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    return out.stdout


def _xml_resolver(root):
    """id/ref value-dictionary resolver for an exported table.

    The export uses a global id/ref dictionary: the first use of a value
    carries `id=` + the data, later uses are `<tag ref=.../>`. Return a fn
    mapping any element to its definition.
    """
    registry = {el.get("id"): el for el in root.iter() if el.get("id")}

    def resolve(el):
        ref = el.get("ref")
        return registry.get(ref, el) if ref else el

    return resolve


def _load_clock_timeline(trace: str) -> list[tuple[int, int, str]]:
    """Sorted (start_ns, end_ns, state) GPU clock windows from the trace.

    Apple's DVFS drops the GPU clock between the synchronized per-call
    dispatches, so short kernels are often measured downclocked; tagging
    each interval with its clock lets us trust the 'Maximum'-clock rows as
    steady state. Empty if the table is absent.
    """
    try:
        root = ET.fromstring(_xctrace_export(trace, _CLOCK_SCHEMA))
    except RuntimeError:
        return []
    resolve = _xml_resolver(root)
    windows: list[tuple[int, int, str]] = []
    for row in root.iter("row"):
        start = dur = state = None
        for k in row:
            if k.tag == "start-time":
                start = int(resolve(k).text)
            elif k.tag == "duration":
                dur = int(resolve(k).text)
            elif k.tag == "gpu-performance-state":
                state = resolve(k).get("fmt", "")
        if start is not None and state:
            windows.append((start, start + (dur or 0), state))
    windows.sort()
    return windows


def _clock_at(windows: list[tuple[int, int, str]], t: int) -> str:
    """GPU clock state active at timestamp `t` (ns); '' if unknown."""
    for s, e, st in windows:
        if s <= t < e:
            return st
    return ""


def _gpu_duty_cycle(trace: str) -> dict | None:
    """GPU Active vs Idle residency over the trace, from
    `metal-gpu-state-intervals`.

    This is device-global (the GPU state machine, not per-process), but
    during a headless bench run our workload is the only meaningful GPU
    user, so it reads as the kernel's duty cycle. A low active fraction
    means the work is launch/sync-bound rather than compute-bound — the
    most actionable headless signal Instruments gives us on Apple, since
    the per-shader counters (ALU%, occupancy, bandwidth) need the GUI.
    Returns None if the table is absent.
    """
    try:
        root = ET.fromstring(_xctrace_export(trace, "metal-gpu-state-intervals"))
    except RuntimeError:
        return None
    resolve = _xml_resolver(root)
    by_state: dict[str, int] = defaultdict(int)
    for row in root.iter("row"):
        state = None
        dur = 0
        for k in row:
            if k.tag == "gpu-state":
                state = resolve(k).get("fmt", "")
            elif k.tag == "duration":
                txt = resolve(k).text
                dur = int(txt) if txt else 0
        if state:
            by_state[state] += dur
    total = sum(by_state.values())
    if not total:
        return None
    active = by_state.get("Active", 0)
    return {
        "active_ms": active / 1e6,
        "idle_ms": by_state.get("Idle", 0) / 1e6,
        "busy_pct": active / total * 100.0,
        "states_ms": {k: v / 1e6 for k, v in sorted(by_state.items())},
    }


def _normalize_label(lbl: str) -> str:
    """Collapse per-iteration encoder labels so they group across calls.

    `Command Buffer 7:Compute Command 0` -> `Compute Command`.
    """
    lbl = re.sub(r"Command Buffer \d+:", "", lbl)
    lbl = re.sub(r" \d+$", "", lbl)
    return lbl.strip() or "?"


def _parse_intervals(xml: bytes, process: str):
    """Yield (channel, label, start_ns, duration_ns) per GPU encoder.

    Each row's GPU duration is its *first* `<duration>` child — the second
    is the "CPU to GPU Latency" column, also typed `duration`. The
    `metal-gpu-intervals` table lumps every process's GPU work together
    (WindowServer compositing dominates the row count), so filter by
    `process` (the workload is `python`).
    """
    root = ET.fromstring(xml)
    resolve = _xml_resolver(root)
    for row in root.iter("row"):
        kids = list(row)
        proc = ""
        durations = []
        channel = ""
        flabel = None
        for k in kids:
            if k.tag == "process" and not proc:
                proc = resolve(k).get("fmt", "")
            elif k.tag == "duration":
                durations.append(resolve(k))
            elif k.tag == "gpu-channel-name" and not channel:
                channel = resolve(k).get("fmt", "")
            elif k.tag == "formatted-label" and flabel is None:
                flabel = resolve(k)
        if process and process not in proc:
            continue
        if not durations or durations[0].text is None:
            continue
        label = "?"
        if flabel is not None:
            s = flabel.find("string")
            if s is not None:
                label = resolve(s).get("fmt", "") or "?"
        start_ns = 0
        for k in kids:
            if k.tag == "start-time":
                start_ns = int(resolve(k).text)
                break
        yield channel, _normalize_label(label), start_ns, int(durations[0].text)


def _record_trace(child_argv: list[str], trace: str, *, attempts: int = 6) -> None:
    """Record a Metal System Trace around `child_argv`, retrying flaky runs.

    `xctrace record --launch` intermittently crashes (Bus/Segfault) while
    finalizing the bundle, leaving an unexportable `.trace`. It's flaky, not
    deterministic, so retry until the intervals table actually exports back.
    The traced child inherits `_TRACED_ENV=1` so it runs the bare workload
    loop instead of recursively orchestrating another trace.
    """
    env = {**os.environ, _TRACED_ENV: "1"}
    for attempt in range(1, attempts + 1):
        if os.path.exists(trace):
            shutil.rmtree(trace, ignore_errors=True)
        subprocess.run(
            [
                "xctrace", "record", "--template", "Metal System Trace",
                "--output", trace, "--launch", "--", *child_argv,
            ],
            env=env,
        )  # ignore returncode: xctrace crashes on finalize but may still write
        try:
            _xctrace_export(trace, _XCTRACE_SCHEMA)
            return
        except RuntimeError:
            print(f"  (attempt {attempt}/{attempts}: unreadable trace, retrying)")
    raise SystemExit(
        f"xctrace failed to produce a readable trace after {attempts} attempts"
    )


# Apple GPU performance states, lowest to highest clock. Used to pick the
# least-throttled group as the headline when full Maximum clock never landed.
_CLOCK_RANK = {"Minimum": 0, "Low": 1, "Medium": 2, "High": 3, "Maximum": 4}


def _pick_headline(groups: list[dict]) -> dict | None:
    """The conv kernel's group: the `Compute Command` encoder, at the highest
    GPU clock state observed.

    Mojo doesn't label its Metal encoders, so the kernel lands under the
    `Compute Command` encoder label; host<->device copies are `Blit Command`
    (often on the same `Compute` GPU channel, so we must match on the
    *label*, not the channel). We trust the Maximum-clock group (DVFS steady
    state); if the governor never got there, we take the highest clock seen
    as the least-throttled estimate (and warn). Ties break by interval count.
    """
    kernel = [g for g in groups if "compute command" in g["label"].lower()]
    if not kernel:
        # Fall back to any compute-channel encoder that isn't a copy.
        kernel = [
            g for g in groups
            if "compute" in g["channel"].lower() and "blit" not in g["label"].lower()
        ]
    if not kernel:
        return None
    return max(kernel, key=lambda g: (_CLOCK_RANK.get(g["clock"], -1), g["count"]))


def _summarize_trace(trace: str, *, process: str = "python") -> tuple[dict, list[float]]:
    """Parse the trace into a structured Instruments analysis.

    Returns (analysis_dict, headline_durs_us) where the analysis lists every
    GPU encoder grouped by (channel, label, clock) and headline_durs_us is
    the per-interval µs list for the conv kernel (for min/spread reporting).
    """
    windows = _load_clock_timeline(trace)
    bucket: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    n = 0
    grand_ns = 0
    for channel, label, start_ns, ns in _parse_intervals(
        _xctrace_export(trace, _XCTRACE_SCHEMA), process
    ):
        clock = _clock_at(windows, start_ns) or "?"
        bucket[(channel, label, clock)].append(ns)
        n += 1
        grand_ns += ns

    groups: list[dict] = []
    headline_by_key: dict[tuple, list[int]] = {}
    for (channel, label, clock), durs in bucket.items():
        groups.append(
            {
                "channel": channel,
                "label": label,
                "clock": clock,
                "count": len(durs),
                # Median, not mean: with DVFS the distribution is bimodal and
                # the median of a single-clock group is a robust estimate.
                "median_us": statistics.median(durs) / 1e3,
                "min_us": min(durs) / 1e3,
                "max_us": max(durs) / 1e3,
                "total_us": sum(durs) / 1e3,
            }
        )
        headline_by_key[(channel, label, clock)] = durs
    groups.sort(key=lambda g: -g["total_us"])

    headline = _pick_headline(groups)
    headline_durs = []
    if headline is not None:
        key = (headline["channel"], headline["label"], headline["clock"])
        headline_durs = [d / 1e3 for d in headline_by_key[key]]

    analysis = {
        "trace": trace,
        "process": process,
        "clock_windows": bool(windows),
        "groups": groups,
        "headline": headline,
        "intervals": n,
        "total_gpu_ms": grand_ns / 1e6,
        "duty": _gpu_duty_cycle(trace),
    }
    return analysis, headline_durs


def run_metal_kernel(cfg: Config, args, env_sig: dict, tag: str) -> dict:
    """Orchestrate the Apple kernel-time measurement and assemble the report.

    Pre-warms the JIT cache (un-traced subprocess), records a Metal System
    Trace around the workload, parses the per-encoder GPU intervals, and
    returns a report carrying both the headline kernel time (mojo) and the
    full Instruments analysis (`metal_analysis`).
    """
    if shutil.which("xctrace") is None:
        raise SystemExit(
            "--measure kernel on mps needs Apple `xctrace` (install the Xcode "
            "command line tools), or use --measure walltime / raw instead"
        )

    # 1) Pre-warm in a separate, un-traced process so `mojo build` lands in the
    #    on-disk JIT cache and the traced run only dlopen()s the cached .so.
    #    Capture its output: the workload's own report is noise here, but we
    #    surface it if the pre-warm fails (e.g. a JIT compile error).
    print("=== pre-warm (fills JIT cache; not traced) ===")
    prewarm = _workload_argv(cfg, args, iters=1, warmup=3)
    warm = subprocess.run(prewarm, capture_output=True, text=True)
    if warm.returncode != 0:
        raise SystemExit(
            f"pre-warm workload failed (exit {warm.returncode}):\n"
            f"{warm.stdout}\n{warm.stderr}"
        )

    # 2) Record the Metal System Trace around a re-launch of the workload.
    trace_dir = tempfile.mkdtemp(prefix="ccv_metal_")
    trace = os.path.join(trace_dir, f"ccv_{cfg.fn}.trace")
    child = _workload_argv(cfg, args, iters=args.iters, warmup=1)
    print(f"\n=== xctrace record (Metal System Trace) -> {trace} ===")
    _record_trace(child, trace)

    # 3) Read per-encoder GPU intervals back out.
    analysis, headline_durs = _summarize_trace(trace)
    m = Measurement(runs_us=headline_durs)
    if not headline_durs:
        m.error = "no Compute GPU intervals found in trace"

    rep = _assemble_report(cfg, args, env_sig, tag, {"mojo": m})
    rep["metal_analysis"] = analysis
    return rep


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _env_signature(args, device: str) -> dict[str, str]:
    if device == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    elif device == "mps":
        gpu = _mac_chip()
    else:
        gpu = f"cpu:{os.cpu_count()}"
    # The master script passes the locked clock (MHz) so the cache key
    # tracks the measurement environment; "unlocked" otherwise.
    return {"gpu": gpu, "clock": args.clock_locked or "unlocked"}


def run(args) -> dict:
    device = _resolve_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device required for --device cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS (Apple GPU) device required for --device mps")

    shape = _parse_shape(args.shape, args.fn, args.width)
    if args.fn in ("fwd", "bwd"):
        width = shape[3]
    else:
        width = args.width
    state_len = args.state_len if args.state_len else width - 1

    cfg = Config(
        fn=args.fn,
        shape=shape,
        dtype=args.dtype,
        device=device,
        width=width,
        state_len=state_len,
        activation=None if args.activation == "none" else args.activation,
        has_bias=not args.no_bias,
        has_seq_idx=args.seq_idx,
        has_initial_states=args.initial_states,
        return_final_states=args.return_final_states,
        has_cache_seqlens=args.cache_seqlens,
        has_conv_state_indices=args.conv_state_indices,
        channel_last=args.channel_last,
        seed=args.seed,
    )

    impls = _parse_impls(args.impl)
    env_sig = _env_signature(args, device)
    upstream_tag = args.baseline_tag or (
        getattr(_upstream_module(), "__version__", "unknown")
        if ("upstream" in impls and args.measure != "raw" and device == "cuda")
        else "n/a"
    )
    # Per-impl cache tags (see BaselineCache): upstream tracks the wheel
    # version; pytorch is our own ref, so a stable constant.
    tags = {"upstream": upstream_tag, "pytorch": "pytorch-ref"}
    tag = upstream_tag  # headline tag shown in the report

    # Apple: no in-process device-time hook, so kernel-time measurement is an
    # out-of-process xctrace orchestration (mojo-only — upstream is CUDA-only).
    if device == "mps" and args.measure == "kernel" and not _is_traced_workload():
        if "mojo" not in impls:
            raise SystemExit(
                "mps kernel-time tracing is mojo-only (upstream is CUDA-only); "
                "pass --impl mojo, or use --measure walltime/raw for other impls"
            )
        return run_metal_kernel(cfg, args, env_sig, tag)

    cache = None
    if args.measure in ("kernel", "walltime") and not args.no_baseline_cache:
        cache = BaselineCache(
            fn=args.fn, measure=args.measure, env_sig=env_sig, tags=tags
        )

    results: dict[str, Measurement] = {}
    for impl in impls:
        if not _supports(impl, cfg):
            continue

        # Baselines (upstream/pytorch) are cacheable; mojo is always measured.
        is_baseline = impl in ("upstream", "pytorch")
        if (
            cache is not None
            and is_baseline
            and not args.refresh_baseline
            and (hit := cache.get(impl, shape, cfg.as_dict())) is not None
        ):
            results[impl] = hit
            continue

        m = _measure_impl(impl, cfg, args)
        results[impl] = m
        if cache is not None and is_baseline:
            cache.put(impl, shape, cfg.as_dict(), m)

    return _assemble_report(cfg, args, env_sig, tag, results)


def _measure_impl(impl: str, cfg: Config, args) -> Measurement:
    classify = _classifier(impl, cfg.fn)
    m = Measurement()
    # Isolate per-impl failures: e.g. upstream rejects standard-layout
    # seq_idx/initial_states ("only supported for channel last layout") —
    # that must not crash the whole comparison. Record it and move on.
    try:
        call = make_callable(impl, cfg)
        for _ in range(args.runs):
            if args.measure == "kernel":
                us = measure_kernel(
                    call, classify, iters=args.iters, warmup=args.warmup,
                    device=cfg.device,
                )
            elif args.measure == "walltime":
                us = measure_walltime(
                    call, min_run_time=args.min_run_time, warmup=args.warmup,
                    device=cfg.device,
                )
            else:  # raw
                us = measure_raw(
                    call, iters=args.iters, warmup=args.warmup, device=cfg.device
                )
            m.runs_us.append(us)
    except Exception as e:  # noqa: BLE001 — surface, don't abort the comparison
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        m.error = msg[:120]
    return m


def _assemble_report(cfg, args, env_sig, tag, results: dict[str, Measurement]) -> dict:
    out_results = {
        impl: {
            "runs_us": m.runs_us,
            "min_us": m.min_us,
            "spread_pct": m.spread_pct,
            "from_cache": m.from_cache,
            "error": m.error,
        }
        for impl, m in results.items()
    }
    ratios = {}
    if "mojo" in results and results["mojo"].runs_us:
        for base in ("upstream", "pytorch"):
            if base in results and results[base].min_us > 0:
                ratios[f"mojo_over_{base}"] = results["mojo"].min_us / results[base].min_us
    return {
        "fn": cfg.fn,
        "shape": list(cfg.shape),
        "measure": args.measure,
        "iters": args.iters,
        "warmup": args.warmup,
        "runs": args.runs,
        "config": cfg.as_dict(),
        "env": env_sig,
        "baseline_tag": tag,
        "results": out_results,
        "ratio_min": ratios,
    }


# ---------------------------------------------------------------------------
# Output formatting.
# ---------------------------------------------------------------------------


def _print_human(rep: dict) -> None:
    cfg = rep["config"]
    flags = [k for k in ("bias", "seq_idx", "initial_states", "return_final_states",
                         "cache_seqlens", "conv_state_indices") if cfg.get(k)]
    print(
        f"{rep['fn'].upper()} | shape={tuple(rep['shape'])} | dtype={cfg['dtype']} "
        f"| act={cfg['activation']} | {'+'.join(flags) or 'no-flags'} "
        f"| measure={rep['measure']} | runs={rep['runs']}"
    )
    print(f"  env: {rep['env']}  baseline_tag={rep['baseline_tag']}")
    hdr = f"  {'impl':>9} | {'min (us)':>10} | {'spread':>7} | {'cache':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for impl, r in rep["results"].items():
        if r.get("error"):
            print(f"  {impl:>9} | {'error':>10} | {'':>7} | {'':>5}   ({r['error']})")
            continue
        cache = "hit" if r["from_cache"] else "-"
        print(
            f"  {impl:>9} | {r['min_us']:10.2f} | {r['spread_pct']:6.1f}% | {cache:>5}"
        )
    for name, ratio in rep["ratio_min"].items():
        base = name.replace("mojo_over_", "")
        verdict = _verdict(ratio, rep, ("mojo", base))
        print(f"  ratio {name}: {ratio:.3f}x  {verdict}")
    if rep.get("metal_analysis"):
        _print_metal_analysis(rep["metal_analysis"])


def _print_metal_analysis(a: dict) -> None:
    """Print the per-encoder GPU-time breakdown read back from the trace."""
    print()
    print(f"  METAL INSTRUMENTS ANALYSIS  (Metal System Trace, process={a['process']!r})")
    if not a["clock_windows"]:
        print("    (no gpu-performance-state-intervals in trace; clock state unknown)")
    hdr = (
        f"    {'channel':>8} | {'clock':>8} | {'count':>5} | {'median':>9} | "
        f"{'min':>9} | {'max':>9} | encoder"
    )
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for g in a["groups"]:
        print(
            f"    {g['channel']:>8} | {g['clock']:>8} | {g['count']:>5} | "
            f"{g['median_us']:8.2f}u | {g['min_us']:8.2f}u | {g['max_us']:8.2f}u | "
            f"{g['label']}"
        )
    h = a["headline"]
    if h:
        print(
            f"    headline kernel time (Compute Command @ {h['clock']} clock, "
            f"median): {h['median_us']:.2f} us  (count={h['count']})"
        )
        if a["clock_windows"] and h["clock"] != "Maximum":
            print(
                f"    WARNING: GPU never reached Maximum clock (best observed: "
                f"{h['clock']}); this is a DVFS-throttled number, not steady "
                f"state. The per-call sync lets the governor downclock between "
                f"dispatches — re-run, or drive a larger/longer workload."
            )
    else:
        print("    (no Compute Command encoder intervals found — nothing to attribute)")
    print(
        f"    total GPU time: {a['total_gpu_ms']:.3f} ms across "
        f"{a['intervals']} encoder intervals"
    )
    d = a.get("duty")
    if d:
        print(
            f"    GPU duty cycle (device-global): {d['busy_pct']:.1f}% active  "
            f"({d['active_ms']:.2f} ms active / {d['idle_ms']:.2f} ms idle)"
        )
        if d["busy_pct"] < 40.0:
            print(
                "          -> low residency: launch/sync-bound, not compute-bound "
                "(the per-call sync starves the GPU and holds the clock down)."
            )
    if a["clock_windows"]:
        print("    NOTE: trust the 'Maximum'-clock rows; lower states are DVFS noise.")
    print("    NOTE: occupancy / ALU% / bandwidth / stalls are GUI-only on Apple —")
    print(f"          open the trace in Instruments for those: {a['trace']}")


def _verdict(ratio: float, rep: dict, impls: tuple[str, ...]) -> str:
    """Trust gate: a win/loss is only real when it clears the spread.

    Only the two impls *in this ratio* gate it — a third noisy impl
    (e.g. pytorch) must not mask a real mojo-vs-upstream delta.
    """
    spreads = [
        rep["results"][i]["spread_pct"]
        for i in impls
        if i in rep["results"] and rep["results"][i]["runs_us"]
    ]
    worst_spread = max(spreads) if spreads else 0.0
    margin_pct = abs(ratio - 1.0) * 100.0
    if margin_pct <= 3.0:
        return "(within 3% — parity)"
    if margin_pct < worst_spread:
        return f"(±{worst_spread:.1f}% spread > {margin_pct:.1f}% gap — NOISE)"
    return "(faster)" if ratio < 1.0 else "(SLOWER)"


def _parse_shape(s: str | None, fn: str, width: int) -> tuple[int, ...]:
    if s is None:
        # Sensible canonical defaults so the tool is usable standalone.
        return (1, 4096, 2048, 4) if fn in ("fwd", "bwd") else (16, 2048)
    parts = tuple(int(x) for x in s.replace("x", ",").split(",") if x)
    if fn in ("fwd", "bwd") and len(parts) == 3:
        parts = (*parts, width)  # allow B,D,L with --width
    return parts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("fn", choices=("fwd", "bwd", "update"), nargs="?", default=None,
                   help="function to benchmark")
    p.add_argument("--kind", choices=("fwd", "bwd", "update"), default=None,
                   help="legacy alias for the positional function name")
    p.add_argument(
        "--shape",
        help="B,D,L,W for fwd/bwd (W optional, use --width); B,D for update",
    )
    p.add_argument("--dtype", choices=tuple(_DTYPES), default="fp16")
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                   help="auto: cuda if present, else mps (Apple), else cpu")
    p.add_argument("--width", type=int, default=4, help="conv width (update; fwd/bwd take it from --shape)")
    p.add_argument("--state-len", type=int, default=0, help="update conv_state length (default width-1)")
    p.add_argument("--activation", choices=("none", "silu", "swish"), default="silu")
    p.add_argument("--no-bias", action="store_true")
    # fwd/bwd function-argument flags
    p.add_argument("--seq-idx", action="store_true", help="packed-sequence mask (fwd/bwd)")
    p.add_argument("--initial-states", action="store_true", help="prepend conv state (fwd/bwd)")
    p.add_argument("--return-final-states", action="store_true", help="emit final states (fwd)")
    p.add_argument("--channel-last", action="store_true",
                   help="channel-last x layout (required by upstream for seq-idx/initial-states)")
    # update function-argument flags
    p.add_argument("--cache-seqlens", action="store_true", help="circular conv_state (update)")
    p.add_argument("--conv-state-indices", action="store_true", help="paged conv_state (update)")
    # impls + measurement
    p.add_argument("--impl", default="mojo",
                   help="comma list of {mojo,upstream,pytorch}, or 'all' / 'both' (legacy)")
    p.add_argument("--measure", choices=("kernel", "walltime", "raw"), default="kernel")
    p.add_argument("--iters", type=int, default=100, help="inner iters (kernel/raw)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--runs", type=int, default=3, help="repeat measurements (>=3 for min+spread)")
    p.add_argument("--min-run-time", type=float, default=0.5, help="walltime autorange budget (s)")
    p.add_argument("--seed", type=int, default=0)
    # baseline cache
    p.add_argument("--no-baseline-cache", "--no-cache", action="store_true",
                   help="always re-measure baselines (--no-cache is a legacy alias)")
    p.add_argument("--refresh-baseline", action="store_true", help="re-measure + re-seed baselines")
    p.add_argument("--baseline-tag", default=None, help="override baseline identity (default: upstream version)")
    p.add_argument("--clock-locked", default=None, help="locked clock MHz (folded into cache key)")
    # output
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Resolve the function from the positional arg or the legacy --kind alias.
    args.fn = args.fn or args.kind
    if args.fn is None:
        parser.error("a function is required: pass it positionally (e.g. `fwd`) or via --kind")
    rep = run(args)
    # The traced Apple workload exists only to drive the kernel under
    # xctrace; its own report is noise inside the recording. Stay silent
    # (the orchestrator parses the trace and prints the real findings).
    if _is_traced_workload() and not args.json:
        return
    if args.json:
        print(json.dumps(rep))
    else:
        _print_human(rep)


if __name__ == "__main__":
    main()
