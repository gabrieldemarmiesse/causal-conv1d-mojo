"""Wall-time bench for the backward pass: mojo vs upstream vs pure PyTorch.

Measures per-call time of `out.backward(dout)` on a fresh autograd graph
each iteration. Forward time is included in each sample because the graph
needs to exist before backward; reported numbers are total wall-clock per
(forward + backward) call to keep apples-to-apples.

mojo:     causal_conv1d_mojo.causal_conv1d_fn (native fwd + custom bwd)
upstream: causal_conv1d.causal_conv1d_fn (Tri Dao CUDA fwd + bwd)
pytorch:  pure F.conv1d(groups=D)+F.silu, autograd-driven backward
"""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


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
    Returns median per-iter wall time in us covering forward + backward + sync.
    """
    for _ in range(WARMUP):
        out, dout = make_call()
        out.backward(dout)
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        out, dout = make_call()
        out.backward(dout)
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | iters={ITERS} (forward + backward)\n"
    )

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
        u = bench_one(call_upstream)
        p = bench_one(call_pytorch)
        print(f"{(B, D, L, W)!s:>22} | {m:9.1f}u | {u:9.1f}u | {p:9.1f}u")


if __name__ == "__main__":
    main()
