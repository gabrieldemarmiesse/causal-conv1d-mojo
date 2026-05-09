"""Compare three forward paths on identical workloads:

  * "mojo"     -- causal_conv1d_mojo.causal_conv1d_fn (native Mojo kernel
                  via direct CPython extension).
  * "upstream" -- causal_conv1d.causal_conv1d_fn (Tri Dao's hand-tuned
                  CUDA kernel via torch.library.custom_op).
  * "pytorch"  -- a pure-PyTorch reference using F.conv1d + F.silu, the
                  fallback you'd write if you didn't have a custom op
                  at all.

Reports two numbers per call: wall-clock per call (sync after each
call) and host-only submit time (one sync at the end). Both at fp16
with bias and silu, the bench config our native path specializes for.
"""

from __future__ import annotations

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
WARMUP = 30
ITERS = 500


def call_pytorch(x, weight, bias) -> torch.Tensor:
    """Pure-PyTorch causal_conv1d_fn equivalent.

    x: (B, D, L), weight: (D, W), bias: (D,). Returns (B, D, L).
    """
    seqlen = x.shape[-1]
    D, W = weight.shape
    out = F.conv1d(
        x,
        weight.unsqueeze(1),  # (D, 1, W) for groups=D depthwise
        bias,
        padding=W - 1,
        groups=D,
    )[..., :seqlen]
    return F.silu(out)


def bench_wall(fn) -> float:
    # Min over samples: kernel time is "this many cycles + possible
    # interference"; min picks the run that wasn't disturbed. Mean and
    # median are both shifted upward by system noise; min is the
    # tightest noise-free estimate.
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return min(samples) / 1_000.0


def bench_host(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        fn()
    elapsed = time.perf_counter_ns() - t0
    torch.cuda.synchronize()
    return elapsed / ITERS / 1_000.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | iters={ITERS}\n"
    )

    h = (
        f"{'shape (B,D,L,W)':>22} | "
        f"{'mojo wall':>10} | {'mojo host':>10} | "
        f"{'up wall':>10} | {'up host':>10} | "
        f"{'pt wall':>10} | {'pt host':>10}"
    )
    print(h)
    print("-" * len(h))

    for batch, dim, seqlen, width in SHAPES:
        x = torch.randn(batch, dim, seqlen, generator=g).to(
            device=device, dtype=torch.float16
        )
        weight = torch.randn(dim, width, generator=g).to(
            device=device, dtype=torch.float16
        )
        bias = torch.randn(dim, generator=g).to(device=device, dtype=torch.float16)

        kw = dict(bias=bias, activation="silu")
        m_wall = bench_wall(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw)
        )
        m_host = bench_host(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw)
        )
        u_wall = bench_wall(lambda: upstream_fn(x, weight, **kw))
        u_host = bench_host(lambda: upstream_fn(x, weight, **kw))
        p_wall = bench_wall(lambda: call_pytorch(x, weight, bias))
        p_host = bench_host(lambda: call_pytorch(x, weight, bias))

        print(
            f"{(batch, dim, seqlen, width)!s:>22} | "
            f"{m_wall:9.1f}u | {m_host:9.1f}u | "
            f"{u_wall:9.1f}u | {u_host:9.1f}u | "
            f"{p_wall:9.1f}u | {p_host:9.1f}u"
        )


if __name__ == "__main__":
    main()
