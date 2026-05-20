"""`causal-conv1d-mojo` console script.

Single subcommand: `bench` — runs forward, backward, and update on one
large shape and prints the speedup vs a pure-PyTorch reference.

    uvx causal-conv1d-mojo bench         # auto-detect cuda / mps / cpu
    uvx causal-conv1d-mojo bench --gpu   # force cuda or mps (fail if none)
    uvx causal-conv1d-mojo bench --cpu   # force cpu
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from causal_conv1d_mojo import (
    causal_conv1d_fn,
    causal_conv1d_ref,
    causal_conv1d_update,
    causal_conv1d_update_ref,
)


# One "big" shape per device class. CUDA gets the largest workload, MPS
# a medium one (smaller GPUs / unified memory), CPU the smallest so the
# bench finishes in seconds.
SHAPES = {
    "cuda": dict(B=4, D=4096, L=2048, W=4),
    "mps": dict(B=2, D=2048, L=2048, W=4),
    "cpu": dict(B=1, D=512, L=1024, W=4),
}
DTYPE = torch.float16


def _pick_device(force: str | None) -> str:
    if force == "cpu":
        return "cpu"
    if force == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        print("error: --gpu requested but no CUDA or MPS device found", file=sys.stderr)
        sys.exit(1)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("no accelerator found, falling back to CPU")
    return "cpu"


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _time(fn, device: str, iters: int, warmup: int) -> float:
    """Return mean wall-clock seconds per call after warmup, syncing
    around the timed region."""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters


def _bench_forward(device, dtype, shape, iters, warmup):
    B, D, L, W = shape["B"], shape["D"], shape["L"], shape["W"]
    x = torch.randn(B, D, L, device=device, dtype=dtype)
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)
    mojo = _time(
        lambda: causal_conv1d_fn(x, weight, bias, activation="silu"),
        device,
        iters,
        warmup,
    )
    ref = _time(
        lambda: causal_conv1d_ref(x, weight, bias, activation="silu"),
        device,
        iters,
        warmup,
    )
    return mojo, ref


def _bench_backward(device, dtype, shape, iters, warmup):
    B, D, L, W = shape["B"], shape["D"], shape["L"], shape["W"]
    x = torch.randn(B, D, L, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(D, W, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(D, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(B, D, L, device=device, dtype=dtype)

    def mojo_step():
        for t in (x, weight, bias):
            t.grad = None
        out = causal_conv1d_fn(x, weight, bias, activation="silu")
        out.backward(dout)

    def ref_step():
        for t in (x, weight, bias):
            t.grad = None
        out = causal_conv1d_ref(x, weight, bias, activation="silu")
        out.backward(dout)

    mojo = _time(mojo_step, device, iters, warmup)
    ref = _time(ref_step, device, iters, warmup)
    return mojo, ref


def _bench_update(device, dtype, shape, iters, warmup):
    B, D, W = shape["B"], shape["D"], shape["W"]
    x = torch.randn(B, D, device=device, dtype=dtype)
    state_mojo = torch.randn(B, D, W - 1, device=device, dtype=dtype)
    state_ref = state_mojo.clone()
    weight = torch.randn(D, W, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)
    mojo = _time(
        lambda: causal_conv1d_update(x, state_mojo, weight, bias, activation="silu"),
        device,
        iters,
        warmup,
    )
    ref = _time(
        lambda: causal_conv1d_update_ref(x, state_ref, weight, bias, activation="silu"),
        device,
        iters,
        warmup,
    )
    return mojo, ref


def _device_label(device: str) -> str:
    if device == "cuda":
        return f"GPU ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "MPS (Apple GPU)"
    return "CPU"


def _print_row(name: str, mojo_s: float, ref_s: float) -> None:
    speedup = ref_s / mojo_s if mojo_s > 0 else float("inf")
    print(
        f"  {name:<9}  mojo {mojo_s * 1e6:8.1f} µs   "
        f"pytorch {ref_s * 1e6:8.1f} µs   "
        f"speedup {speedup:5.2f}×"
    )


def bench(args: argparse.Namespace) -> int:
    device = _pick_device("cpu" if args.cpu else "gpu" if args.gpu else None)
    print(f"device: {_device_label(device)}")

    shape = SHAPES[device]
    print(
        f"shape: B={shape['B']} D={shape['D']} L={shape['L']} W={shape['W']} "
        f"dtype={DTYPE}"
    )

    iters = 20 if device == "cpu" else 100
    warmup = 3 if device == "cpu" else 20
    print(f"warmup={warmup} iters={iters}\n")

    print("forward:")
    fwd_m, fwd_r = _bench_forward(device, DTYPE, shape, iters, warmup)
    _print_row("forward", fwd_m, fwd_r)

    print("backward:")
    bwd_m, bwd_r = _bench_backward(device, DTYPE, shape, iters, warmup)
    _print_row("backward", bwd_m, bwd_r)

    print("update:")
    upd_m, upd_r = _bench_update(device, DTYPE, shape, iters, warmup)
    _print_row("update", upd_m, upd_r)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causal-conv1d-mojo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bench", help="benchmark mojo vs pure-pytorch on one shape")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--cpu", action="store_true", help="force CPU backend")
    grp.add_argument("--gpu", action="store_true", help="force GPU (CUDA or MPS)")
    args = parser.parse_args(argv)
    if args.cmd == "bench":
        return bench(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
