"""Per-kernel GPU-time benchmark using torch.profiler (CUPTI / rocprof).

Wraps each implementation in a `record_function` range, calls it N
times under `torch.profiler`, walks `prof.events()` and sums per-kernel
GPU time. Reports `mojo (us/call)`, `upstream (us/call)`, and the ratio.

Supports running a single shape and/or a single implementation for fast
iteration:

    # all impls (mojo + upstream), all shapes — original behaviour
    python benchmarks/bench_gpu_kernel_time.py

    # only mojo on one shape — fast feedback while editing the kernel
    python benchmarks/bench_gpu_kernel_time.py --shape 1,1024,2048,4 --impl mojo

    # only fwd (skip update kernel grid), only update grid, etc.
    python benchmarks/bench_gpu_kernel_time.py --kind fwd
    python benchmarks/bench_gpu_kernel_time.py --kind update

    # bwd shapes (autograd graph)
    python benchmarks/bench_gpu_kernel_time.py --kind bwd

    # tighten iter count for quicker passes
    python benchmarks/bench_gpu_kernel_time.py --iters 20 --warmup 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

import causal_conv1d_mojo

from causal_conv1d import causal_conv1d_fn as upstream_fn
from causal_conv1d import causal_conv1d_update as upstream_update_fn
from _baseline import BaselineCache


# Default shape grids.
FWD_SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 8192, 4),
    (1, 2048, 2048, 4),
    (1, 4096, 2048, 4),
    (4, 2048, 2048, 4),
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]
BWD_SHAPES = FWD_SHAPES
UPDATE_SHAPES = [
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 1024),
    (4, 2048),
    (4, 4096),
    (16, 2048),
    (32, 4096),
]


# Classifiers — these match kernel names that show up in torch.profiler
# events. Mojo's `mojo build` mangles comptime params into kernel names
# (e.g. `kernel_fwd_kernel_DType_..._<hash>`); upstream CUDA kernels are
# `void causal_conv1d_*_kernel<...>`.
def _is_mojo_fwd(name: str) -> bool:
    return "fwd_kernel" in name and not name.startswith("void")


def _is_mojo_bwd(name: str) -> bool:
    return "bwd" in name and "kernel" in name and not name.startswith("void")


def _is_mojo_update(name: str) -> bool:
    return "update_kernel" in name and not name.startswith("void")


def _is_upstream_fwd(name: str) -> bool:
    return name.startswith("void causal_conv1d_fwd_kernel")


def _is_upstream_bwd(name: str) -> bool:
    return name.startswith("void causal_conv1d_bwd_kernel")


def _is_upstream_update(name: str) -> bool:
    return name.startswith("void causal_conv1d_update_kernel")


def _sum_cuda_us(prof, predicate) -> float:
    total = 0.0
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if predicate(evt.name):
            total += evt.self_device_time_total
    return total


def _bench(
    fn: Callable[[], None],
    predicate: Callable[[str], bool],
    iters: int,
    warmup: int,
) -> float:
    """Mean per-call GPU time, μs, attributed to kernels matched by `predicate`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return _sum_cuda_us(prof, predicate) / iters


# ----------------------------- workload builders -----------------------------


def _make_fwd_call(impl: str, shape, *, dtype, activation, device, g):
    b, d, l, w = shape
    x = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, w, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)

    if impl == "mojo":
        return lambda: causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, bias=bias, activation=activation
        )
    if impl == "upstream":
        return lambda: upstream_fn(x, weight, bias=bias, activation=activation)
    raise ValueError(f"unknown fwd impl: {impl}")


def _make_bwd_call(impl: str, shape, *, dtype, activation, device, g):
    """Returns a 0-arg callable that does fwd + bwd once."""
    b, d, l, w = shape
    x = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, w, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)
    dout = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)

    fwd = (
        causal_conv1d_mojo.causal_conv1d_fn
        if impl == "mojo"
        else upstream_fn
    )

    def call():
        x_ = x.detach().requires_grad_()
        w_ = weight.detach().requires_grad_()
        b_ = bias.detach().requires_grad_()
        out = fwd(x_, w_, bias=b_, activation=activation)
        out.backward(dout)

    return call


def _make_update_call(impl: str, shape, *, dtype, activation, device, g, width=4):
    b, d = shape
    state_len = width - 1
    x = torch.randn(b, d, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, width, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)
    state = torch.randn(b, d, state_len, generator=g).to(device=device, dtype=dtype)

    if impl == "mojo":
        return lambda: causal_conv1d_mojo.causal_conv1d_update(
            x, state, weight, bias=bias, activation=activation
        )
    if impl == "upstream":
        return lambda: upstream_update_fn(
            x, state, weight, bias=bias, activation=activation
        )
    raise ValueError(f"unknown update impl: {impl}")


# ----------------------------- per-kind drivers -----------------------------


def run_fwd(args, shapes, device, dtype, g) -> None:
    print(
        f"FWD kernel: GPU={torch.cuda.get_device_name(0)} | dtype={args.dtype} "
        f"| activation={args.activation} | bias=True | iters={args.iters}"
    )
    header = (
        f"{'shape (B,D,L,W)':>22} | {'mojo (us/call)':>15} | "
        f"{'upstream (us/call)':>19} | {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    cache = BaselineCache(__file__) if not args.no_cache else None
    cfg = {
        "dtype": args.dtype,
        "activation": args.activation,
        "bias": True,
        "iters": args.iters,
    }
    want = args.impl

    for shape in shapes:
        mojo_us = up_us = float("nan")

        if want in ("mojo", "both"):
            call = _make_fwd_call(
                "mojo", shape,
                dtype=dtype, activation=args.activation, device=device, g=g,
            )
            mojo_us = _bench(call, _is_mojo_fwd, args.iters, args.warmup)

        if want in ("upstream", "both"):
            def run_up():
                call = _make_fwd_call(
                    "upstream", shape,
                    dtype=dtype, activation=args.activation, device=device, g=g,
                )
                return _bench(call, _is_upstream_fwd, args.iters, args.warmup)

            up_us = (
                cache.get_or_run(impl="upstream", shape=shape, config=cfg, run=run_up)
                if cache is not None else run_up()
            )

        ratio = (mojo_us / up_us) if (want == "both" and up_us) else float("nan")
        ratio_str = f"{ratio:6.2f}x" if ratio == ratio else "    -"
        mojo_str = f"{mojo_us:15.1f}" if mojo_us == mojo_us else f"{'-':>15}"
        up_str = f"{up_us:19.1f}" if up_us == up_us else f"{'-':>19}"
        print(f"{shape!s:>22} | {mojo_str} | {up_str} | {ratio_str}")


def run_bwd(args, shapes, device, dtype, g) -> None:
    print(
        f"BWD kernel: GPU={torch.cuda.get_device_name(0)} | dtype={args.dtype} "
        f"| activation={args.activation} | bias=True | iters={args.iters}"
    )
    header = (
        f"{'shape (B,D,L,W)':>22} | {'mojo (us/call)':>15} | "
        f"{'upstream (us/call)':>19} | {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    cache = BaselineCache(__file__) if not args.no_cache else None
    cfg = {
        "dtype": args.dtype,
        "activation": args.activation,
        "bias": True,
        "iters": args.iters,
        "kind": "bwd",
    }
    want = args.impl

    for shape in shapes:
        mojo_us = up_us = float("nan")

        if want in ("mojo", "both"):
            call = _make_bwd_call(
                "mojo", shape,
                dtype=dtype, activation=args.activation, device=device, g=g,
            )
            mojo_us = _bench(call, _is_mojo_bwd, args.iters, args.warmup)

        if want in ("upstream", "both"):
            def run_up():
                call = _make_bwd_call(
                    "upstream", shape,
                    dtype=dtype, activation=args.activation, device=device, g=g,
                )
                return _bench(call, _is_upstream_bwd, args.iters, args.warmup)

            up_us = (
                cache.get_or_run(impl="upstream_bwd", shape=shape, config=cfg, run=run_up)
                if cache is not None else run_up()
            )

        ratio = (mojo_us / up_us) if (want == "both" and up_us) else float("nan")
        ratio_str = f"{ratio:6.2f}x" if ratio == ratio else "    -"
        mojo_str = f"{mojo_us:15.1f}" if mojo_us == mojo_us else f"{'-':>15}"
        up_str = f"{up_us:19.1f}" if up_us == up_us else f"{'-':>19}"
        print(f"{shape!s:>22} | {mojo_str} | {up_str} | {ratio_str}")


def run_update(args, shapes, device, dtype, g) -> None:
    print(
        f"UPDATE kernel: GPU={torch.cuda.get_device_name(0)} | dtype={args.dtype} "
        f"| activation={args.activation} | bias=True | seqlen=1 | state_len=3 "
        f"| iters={args.iters}"
    )
    header = (
        f"{'shape (B,D)':>14} | {'mojo (us/call)':>15} | "
        f"{'upstream (us/call)':>19} | {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    cache = BaselineCache(__file__) if not args.no_cache else None
    cfg = {
        "dtype": args.dtype,
        "activation": args.activation,
        "bias": True,
        "iters": args.iters,
        "mode": "update",
        "width": 4,
        "state_len": 3,
    }
    want = args.impl

    for shape in shapes:
        mojo_us = up_us = float("nan")

        if want in ("mojo", "both"):
            call = _make_update_call(
                "mojo", shape,
                dtype=dtype, activation=args.activation, device=device, g=g,
            )
            mojo_us = _bench(call, _is_mojo_update, args.iters, args.warmup)

        if want in ("upstream", "both"):
            def run_up():
                call = _make_update_call(
                    "upstream", shape,
                    dtype=dtype, activation=args.activation, device=device, g=g,
                )
                return _bench(call, _is_upstream_update, args.iters, args.warmup)

            up_us = (
                cache.get_or_run(impl="upstream_update", shape=shape, config=cfg, run=run_up)
                if cache is not None else run_up()
            )

        ratio = (mojo_us / up_us) if (want == "both" and up_us) else float("nan")
        ratio_str = f"{ratio:6.2f}x" if ratio == ratio else "    -"
        mojo_str = f"{mojo_us:15.2f}" if mojo_us == mojo_us else f"{'-':>15}"
        up_str = f"{up_us:19.2f}" if up_us == up_us else f"{'-':>19}"
        print(f"{shape!s:>14} | {mojo_str} | {up_str} | {ratio_str}")


# ----------------------------- arg parsing -----------------------------


def _parse_shape(s: str) -> tuple[int, ...]:
    parts = [int(x) for x in s.replace("x", ",").split(",") if x]
    return tuple(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kind",
        choices=("fwd", "bwd", "update", "all"),
        default="all",
        help="Which kernel(s) to bench. Default `all` runs fwd + update "
             "(bwd needs the autograd graph and is opt-in).",
    )
    p.add_argument(
        "--impl",
        choices=("mojo", "upstream", "both"),
        default="both",
        help="Which implementation to time. `mojo` is the fast-feedback mode "
             "when iterating on the Mojo kernel.",
    )
    p.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Single shape, comma- or x-separated. fwd/bwd: B,D,L,W; update: B,D",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    p.add_argument("--activation", default="silu")
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip the JSON baseline cache (always re-measure upstream).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / ROCm device required")
    device = torch.device("cuda")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.dtype
    ]
    g = torch.Generator(device="cpu").manual_seed(0)

    def shapes_for(kind: str):
        if args.shape is not None:
            return [_parse_shape(args.shape)]
        return {"fwd": FWD_SHAPES, "bwd": BWD_SHAPES, "update": UPDATE_SHAPES}[kind]

    kinds = ("fwd", "update") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        if kind == "fwd":
            run_fwd(args, shapes_for("fwd"), device, dtype, g)
        elif kind == "bwd":
            run_bwd(args, shapes_for("bwd"), device, dtype, g)
        else:
            run_update(args, shapes_for("update"), device, dtype, g)
        print()


if __name__ == "__main__":
    main()
