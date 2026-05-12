"""Compare three forward paths on identical workloads:

  * "mojo"     -- causal_conv1d_mojo.causal_conv1d_fn (native Mojo kernel
                  via direct CPython extension).
  * "upstream" -- causal_conv1d.causal_conv1d_fn (Tri Dao's hand-tuned
                  CUDA kernel via torch.library.custom_op).
  * "pytorch"  -- a pure-PyTorch reference using F.conv1d + F.silu, the
                  fallback you'd write if you didn't have a custom op
                  at all.

Reports GPU-kernel-only time per call (μs) for each impl, measured via
`torch.profiler` (CUPTI traces). Python + cudaLaunchKernel + sync round-
trip overhead is excluded — only the kernel's own GPU execution time is
summed. Both at fp16 with bias and silu, the bench config our native
path specializes for.
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


def bench_kernel(fn) -> float:
    """Mean GPU-kernel time per call, μs, via torch.profiler (CUPTI).

    Warmup runs outside the profiler scope; ITERS calls inside; we sum
    `self_device_time_total` (μs) over every CUDA event in the trace
    and divide by ITERS. Captures ALL kernels launched by `fn` (for the
    PyTorch reference that includes the conv1d + silu kernels).
    """
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            total_us += evt.self_device_time_total
    return total_us / ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | iters={ITERS} | "
        f"GPU kernel time via torch.profiler\n"
    )

    cache = BaselineCache(__file__)
    cfg = {"dtype": "fp16", "activation": "silu", "bias": True, "iters": ITERS}

    h = (
        f"{'shape (B,D,L,W)':>22} | "
        f"{'mojo':>10} | {'upstream':>10} | {'pytorch':>10}"
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
        shape = (batch, dim, seqlen, width)
        m = bench_kernel(lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw))
        u = cache.get_or_run(
            impl="upstream",
            shape=shape,
            config=cfg,
            run=lambda: bench_kernel(lambda: upstream_fn(x, weight, **kw)),
        )
        p = cache.get_or_run(
            impl="pytorch",
            shape=shape,
            config=cfg,
            run=lambda: bench_kernel(lambda: call_pytorch(x, weight, bias)),
        )

        print(
            f"{shape!s:>22} | "
            f"{m:9.1f}u | {u:9.1f}u | {p:9.1f}u"
        )


if __name__ == "__main__":
    main()
