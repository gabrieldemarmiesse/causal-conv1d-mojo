"""GPU-kernel-only bench for the backward pass: mojo vs upstream vs pure PyTorch.

Measures per-call GPU time of `out.backward(dout)` on a fresh autograd
graph each iteration. Forward kernels are included in each sample
because the graph needs to exist before backward; reported numbers sum
every CUDA event launched during (forward + backward), via
torch.profiler (CUPTI) — Python and cudaLaunchKernel overhead are
excluded.

mojo:     causal_conv1d_mojo.causal_conv1d_fn (native fwd + custom bwd)
upstream: causal_conv1d.causal_conv1d_fn (Tri Dao CUDA fwd + bwd)
pytorch:  pure F.conv1d(groups=D)+F.silu, autograd-driven backward
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import causal_conv1d_mojo
from _baseline import BaselineCache

# Optional dep — install with `pip install causal-conv1d==1.6.1` (or
# `pixi run pip install -e .[bench]`). The package is a C++ extension
# whose source-build takes minutes; we only need it for upstream-vs-Mojo
# benchmark comparisons.
try:
    from causal_conv1d import causal_conv1d_fn as upstream_fn
except ImportError as e:
    raise SystemExit(
        "this benchmark compares against upstream causal-conv1d; "
        'run `pip install causal-conv1d==1.6.1` (or `pixi run pip install -e ".[bench]"`) first'
    ) from e


SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 8192, 4),
    (1, 4096, 2048, 4),
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]
WARMUP = 20
ITERS = 200


def _make(B, D, L, W):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(B, D, L, generator=g).to("cuda", torch.float16).requires_grad_()
    weight = torch.randn(D, W, generator=g).to("cuda", torch.float16).requires_grad_()
    bias = torch.randn(D, generator=g).to("cuda", torch.float16).requires_grad_()
    dout = torch.randn(B, D, L, generator=g).to("cuda", torch.float16)
    return x, weight, bias, dout


def _pytorch_fwd(x, weight, bias):
    D, W = weight.shape
    L = x.shape[-1]
    return F.silu(
        F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :L]
    )


def bench_one(make_call) -> float:
    """make_call() must rebuild the autograd graph and return (out, dout).

    Returns mean GPU-kernel time per (forward + backward) call, μs, via
    torch.profiler (CUPTI). All CUDA events emitted between calls are
    summed; Python and launch overhead are excluded.
    """
    for _ in range(WARMUP):
        out, dout = make_call()
        out.backward(dout)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(ITERS):
            out, dout = make_call()
            out.backward(dout)
        torch.cuda.synchronize()
    total_us = 0.0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            total_us += evt.self_device_time_total
    return total_us / ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | iters={ITERS} (forward + backward) | "
        f"kernel time via torch.profiler\n"
    )

    cache = BaselineCache(__file__)
    cfg = {
        "dtype": "fp16",
        "activation": "silu",
        "bias": True,
        "iters": ITERS,
        "mode": "fwd+bwd",
    }

    h = f"{'shape (B,D,L,W)':>22} | {'mojo':>10} | {'upstream':>10} | {'pytorch':>10}"
    print(h)
    print("-" * len(h))

    for B, D, L, W in SHAPES:
        x, weight, bias, dout = _make(B, D, L, W)
        kw = dict(bias=bias, activation="silu")

        def call_mojo():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                weight.detach().requires_grad_(),
                bias.detach().requires_grad_(),
            )
            out = causal_conv1d_mojo.causal_conv1d_fn(x_, w_, **{**kw, "bias": b_})
            return out, dout

        def call_upstream():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                weight.detach().requires_grad_(),
                bias.detach().requires_grad_(),
            )
            out = upstream_fn(x_, w_, **{**kw, "bias": b_})
            return out, dout

        def call_pytorch():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                weight.detach().requires_grad_(),
                bias.detach().requires_grad_(),
            )
            out = _pytorch_fwd(x_, w_, b_)
            return out, dout

        m = bench_one(call_mojo)
        u = cache.get_or_run(
            impl="upstream",
            shape=(B, D, L, W),
            config=cfg,
            run=lambda: bench_one(call_upstream),
        )
        p = cache.get_or_run(
            impl="pytorch",
            shape=(B, D, L, W),
            config=cfg,
            run=lambda: bench_one(call_pytorch),
        )
        print(f"{(B, D, L, W)!s:>22} | {m:9.1f}u | {u:9.1f}u | {p:9.1f}u")


if __name__ == "__main__":
    main()
