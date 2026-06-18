"""Apple-silicon GPU workload driver, launched *under* xctrace.

There is no torch device-time hook for Metal (unlike CUPTI/rocprof on
NVIDIA/AMD), so `bench_gpu_kernel_time.py`'s profiler path doesn't work
here. Instead we run the Mojo Metal kernel in a tight, synchronized loop
inside a process that Instruments records as a "Metal System Trace", and
read the per-encoder GPU intervals back with
`scripts/xctrace_gpu_intervals.py`. `scripts/xctrace_bench.sh` wires the
two together.

This driver is mojo-only on purpose: the upstream Tri Dao causal-conv1d
wheel is CUDA-only, so there is nothing to diff against on Apple — the
goal here is precise *absolute* GPU time for our kernel, not a ratio.

Run it directly (no xctrace) for a quick wall-clock sanity check:

    uv run python benchmarks/bench_metal_gpu.py --kind fwd --shape 1,1024,2048,4

Under xctrace it is invoked the same way; the wall-clock numbers it
prints are dominated by per-iter `torch.mps.synchronize()` and launch
overhead and are NOT the measurement — the trace intervals are.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

import causal_conv1d_mojo


FWD_SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 2048, 2048, 4),
    (4, 2048, 2048, 4),
    (8, 2048, 4096, 4),
]
UPDATE_SHAPES = [
    (1, 512),
    (1, 2048),
    (4, 2048),
    (16, 2048),
    (32, 4096),
]


def _make_fwd_call(shape, *, dtype, activation, device, g):
    b, d, l, w = shape
    x = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, w, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)
    return lambda: causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )


def _make_bwd_call(shape, *, dtype, activation, device, g):
    b, d, l, w = shape
    x = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, w, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)
    dout = torch.randn(b, d, l, generator=g).to(device=device, dtype=dtype)

    def call():
        x_ = x.detach().requires_grad_()
        w_ = weight.detach().requires_grad_()
        b_ = bias.detach().requires_grad_()
        out = causal_conv1d_mojo.causal_conv1d_fn(
            x_, w_, bias=b_, activation=activation
        )
        out.backward(dout)

    return call


def _make_update_call(shape, *, dtype, activation, device, g, width=4):
    b, d = shape
    state_len = width - 1
    x = torch.randn(b, d, generator=g).to(device=device, dtype=dtype)
    weight = torch.randn(d, width, generator=g).to(device=device, dtype=dtype)
    bias = torch.randn(d, generator=g).to(device=device, dtype=dtype)
    state = torch.randn(b, d, state_len, generator=g).to(device=device, dtype=dtype)
    return lambda: causal_conv1d_mojo.causal_conv1d_update(
        x, state, weight, bias=bias, activation=activation
    )


def _run(label: str, call: Callable[[], None], iters: int, warmup: int) -> None:
    # Warmup also fills the JIT cache on a cold machine; pre-warming via the
    # wrapper keeps the trace free of `mojo build` noise (see xctrace_bench.sh).
    for _ in range(warmup):
        call()
    torch.mps.synchronize()

    t0 = time.perf_counter_ns()
    for _ in range(iters):
        call()
        torch.mps.synchronize()
    dt = (time.perf_counter_ns() - t0) / iters / 1e3  # us/call (wall-clock)
    print(f"  {label:>22}: {iters} iters, {dt:8.1f} us/call (wall-clock, not the measurement)")


def _parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.replace("x", ",").split(",") if x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=("fwd", "bwd", "update"), default="fwd")
    p.add_argument("--shape", type=str, default=None,
                   help="Single shape, comma/x-separated. fwd/bwd: B,D,L,W; update: B,D")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    p.add_argument("--activation", default="silu")
    args = p.parse_args()
    activation = None if args.activation in ("none", "identity") else args.activation

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS (Apple GPU) device required")
    device = torch.device("mps")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    g = torch.Generator(device="cpu").manual_seed(0)

    if args.shape is not None:
        shapes = [_parse_shape(args.shape)]
    else:
        shapes = {"fwd": FWD_SHAPES, "bwd": FWD_SHAPES,
                  "update": UPDATE_SHAPES}[args.kind]

    builder = {"fwd": _make_fwd_call, "bwd": _make_bwd_call,
               "update": _make_update_call}[args.kind]

    print(f"{args.kind.upper()} on MPS | dtype={args.dtype} "
          f"| activation={args.activation} | iters={args.iters}")
    for shape in shapes:
        call = builder(shape, dtype=dtype, activation=activation,
                       device=device, g=g)
        _run(str(shape), call, args.iters, args.warmup)


if __name__ == "__main__":
    main()
