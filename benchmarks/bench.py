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

This file replaces the old pile of ``bench_*.py`` scripts: every shape
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
import json
import os
import time
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
    """The pure-PyTorch references don't cover packed sequences / paged
    caches; skip those (impl, config) combinations rather than crash."""
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

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
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
    call: Callable[[], Any], *, min_run_time: float = 0.5, warmup: int = 10
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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer = tbench.Timer(stmt="_call()", globals={"_call": call})
    m = timer.blocked_autorange(min_run_time=min_run_time)
    return m.median * 1e6


def measure_raw(
    call: Callable[[], Any], *, iters: int, warmup: int, device: str
) -> float:
    """Tight synchronized loop, *no* profiler — for ncu/NSight to wrap.

    Returns a coarse wall-clock per-call number (not the measurement of
    record; the external profiler's per-kernel metrics are). The point
    is to drive the kernel with zero profiling overhead in-process.
    """
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        call()
    sync()
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
# Driver.
# ---------------------------------------------------------------------------


def _env_signature(args) -> dict[str, str]:
    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else f"cpu:{os.cpu_count()}"
    )
    # The master script passes the locked clock (MHz) so the cache key
    # tracks the measurement environment; "unlocked" otherwise.
    return {"gpu": gpu, "clock": args.clock_locked or "unlocked"}


def run(args) -> dict:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device required for --device cuda")

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
    env_sig = _env_signature(args)
    upstream_tag = args.baseline_tag or (
        getattr(_upstream_module(), "__version__", "unknown")
        if ("upstream" in impls and args.measure != "raw")
        else "n/a"
    )
    # Per-impl cache tags (see BaselineCache): upstream tracks the wheel
    # version; pytorch is our own ref, so a stable constant.
    tags = {"upstream": upstream_tag, "pytorch": "pytorch-ref"}
    tag = upstream_tag  # headline tag shown in the report

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
                    call, min_run_time=args.min_run_time, warmup=args.warmup
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
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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
    if args.json:
        print(json.dumps(rep))
    else:
        _print_human(rep)


if __name__ == "__main__":
    main()
