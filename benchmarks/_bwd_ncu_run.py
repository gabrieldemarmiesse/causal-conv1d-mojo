"""Minimal driver for `ncu` to profile the bwd kernel on one shape.

`ncu` serializes kernel launches, so we keep things tight: one bench
shape, one forward + backward per profiled iteration, no upstream
comparison (we just want metrics on the Mojo bwd kernel).

Usage:
    ncu --launch-skip 20 --launch-count 5 --set full \
        --kernel-name regex:'bwd_full_kernel' \
        python benchmarks/_bwd_ncu_run.py
"""

from __future__ import annotations

import os
import sys

import torch
import causal_conv1d_mojo


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    # Default to (1, 4096, 2048, 4) — the worst-ratio shape on master
    # (was 1.46x). Override via the bench shape via env vars.
    B = int(os.environ.get("BENCH_B", 4))
    D = int(os.environ.get("BENCH_D", 4096))
    L = int(os.environ.get("BENCH_L", 2048))
    W = int(os.environ.get("BENCH_W", 4))

    g = torch.Generator(device="cpu").manual_seed(0)
    x = (
        torch.randn(B, D, L, generator=g)
        .to("cuda", torch.float16)
        .requires_grad_()
    )
    weight = (
        torch.randn(D, W, generator=g)
        .to("cuda", torch.float16)
        .requires_grad_()
    )
    bias = (
        torch.randn(D, generator=g)
        .to("cuda", torch.float16)
        .requires_grad_()
    )
    dout = torch.randn(B, D, L, generator=g).to("cuda", torch.float16)

    for _ in range(5):
        x_ = x.detach().requires_grad_()
        w_ = weight.detach().requires_grad_()
        b_ = bias.detach().requires_grad_()
        out = causal_conv1d_mojo.causal_conv1d_fn(
            x_, w_, bias=b_, activation="silu"
        )
        out.backward(dout)
    torch.cuda.synchronize()

    iters = int(os.environ.get("BENCH_ITERS", 30))
    for _ in range(iters):
        x_ = x.detach().requires_grad_()
        w_ = weight.detach().requires_grad_()
        b_ = bias.detach().requires_grad_()
        out = causal_conv1d_mojo.causal_conv1d_fn(
            x_, w_, bias=b_, activation="silu"
        )
        out.backward(dout)
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main())
