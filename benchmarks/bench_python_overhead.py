"""Compare host-side (Python + framework dispatch) overhead.

`bench_gpu_kernel_time.py` reports pure GPU kernel time. This script
reports two more numbers per call:

  * "wall (us/call)": wall-clock per call, GPU sync after each call.
    Captures everything: Python wrapper + framework dispatch + kernel launch
    + GPU execution + sync. This is what an interactive caller sees.

  * "host (us/call)": time to *submit* the call back-to-back without
    waiting on the GPU. Approximates the host-only cost (Python +
    framework + cudaLaunchKernel). Subtracting this from "wall" leaves
    the GPU compute time. We pick a workload large enough that the GPU
    is the bottleneck, so the launch queue stays full and back-to-back
    submits don't pile up beyond what the GPU can drain.

Mojo path = causal_conv1d_mojo (MAX CustomOpLibrary -> Mojo kernel).
Upstream  = causal_conv1d.causal_conv1d_fn (torch.library.custom_op -> CUDA).
"""

from __future__ import annotations

import statistics
import time

import torch

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 4096, 2048, 4),
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]
WARMUP = 30
ITERS = 500


def bench_wall(fn, *args, **kwargs) -> float:
    """Median per-call wall time in us, with a cudaSync after each call."""
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0  # ns -> us


def bench_host(fn, *args, **kwargs) -> float:
    """Per-submit host time in us. No per-call sync; only one sync at the
    very end. With a slow-enough GPU workload, the launch queue fills and
    the loop's pace tracks the GPU's drain rate, so we'd actually still
    measure GPU. Use shapes where launch >> kernel for cleaner numbers,
    or compare against `bench_wall` to reason about it."""
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        fn(*args, **kwargs)
    elapsed = time.perf_counter_ns() - t0
    torch.cuda.synchronize()
    return elapsed / ITERS / 1_000.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.float16
    g = torch.Generator(device="cpu").manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | activation=silu | bias=True | iters={ITERS}\n"
    )

    h = (
        f"{'shape (B,D,L,W)':>22} | "
        f"{'mojo wall':>10} | {'mojo host':>10} | "
        f"{'up wall':>10} | {'up host':>10} | "
        f"{'wall ratio':>10}"
    )
    print(h)
    print("-" * len(h))

    for batch, dim, seqlen, width in SHAPES:
        x = torch.randn(batch, dim, seqlen, generator=g).to(device=device, dtype=dtype)
        weight = torch.randn(dim, width, generator=g).to(device=device, dtype=dtype)
        bias = torch.randn(dim, generator=g).to(device=device, dtype=dtype)
        kw = dict(bias=bias, activation="silu")

        m_wall = bench_wall(causal_conv1d_mojo.causal_conv1d_fn, x, weight, **kw)
        m_host = bench_host(causal_conv1d_mojo.causal_conv1d_fn, x, weight, **kw)
        u_wall = bench_wall(upstream_fn, x, weight, **kw)
        u_host = bench_host(upstream_fn, x, weight, **kw)
        ratio = m_wall / u_wall if u_wall else float("inf")
        print(
            f"{(batch, dim, seqlen, width)!s:>22} | "
            f"{m_wall:9.1f}u | {m_host:9.1f}u | "
            f"{u_wall:9.1f}u | {u_host:9.1f}u | "
            f"{ratio:9.2f}x"
        )


if __name__ == "__main__":
    main()
