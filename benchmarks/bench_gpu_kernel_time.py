"""Per-kernel GPU-time benchmark using torch.profiler (CUPTI traces).

torch.profiler reports GPU time per kernel without needing the
NVIDIA performance-counter permission ncu requires. We run warmup
outside the profiler scope, then for each shape call our Mojo-backed
impl and the upstream CUDA op N times each, wrapped in NVTX-style
record_function ranges. The profiler then reports per-kernel cumulative
GPU time, which we group back into "mojo" / "upstream" buckets.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import causal_conv1d_mojo

from causal_conv1d import causal_conv1d_fn as upstream_fn
from _baseline import BaselineCache


SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 8192, 4),
    (1, 2048, 2048, 4),
    (1, 4096, 2048, 4),
    (4, 2048, 2048, 4),
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]
ITERS = 100


def _is_mojo(name: str) -> bool:
    """Return True if `name` looks like a Mojo-emitted CUDA kernel.

    Mojo emits names like `kernel_fwd_kernel_DType_..._<hash>` (the `mojo build`
    backend mangles the comptime parameters into the name).
    """
    return "fwd_kernel" in name and not name.startswith("void")


def _sum_cuda_us(prof) -> float:
    total = 0.0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            total += evt.self_device_time_total
    return total


def _bench_kernel(fn) -> float:
    """Mean per-call GPU kernel time, μs, via torch.profiler (CUPTI)."""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    return _sum_cuda_us(prof) / ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.float16
    activation = "silu"
    g = torch.Generator(device="cpu").manual_seed(0)

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | activation=silu | bias=True | iters={ITERS}\n"
    )
    header = f"{'shape (B,D,L,W)':>22} | {'mojo (us/call)':>15} | {'upstream (us/call)':>19} | {'ratio':>7}"
    print(header)
    print("-" * len(header))

    cache = BaselineCache(__file__)
    cfg = {
        "dtype": "fp16",
        "activation": activation,
        "bias": True,
        "iters": ITERS,
    }

    # Debug the mojo kernel name on first shape so the user sees what name
    # the build produced (was useful when iterating on the comptime tree).
    first_shape = SHAPES[0]
    dumped_debug = False

    for batch, dim, seqlen, width in SHAPES:
        x = torch.randn(batch, dim, seqlen, generator=g).to(device=device, dtype=dtype)
        weight = torch.randn(dim, width, generator=g).to(device=device, dtype=dtype)
        bias = torch.randn(dim, generator=g).to(device=device, dtype=dtype)
        shape = (batch, dim, seqlen, width)

        # Mojo: always re-measure. We also do one debug pass on the first
        # shape to print kernel names emitted by mojo build.
        if shape == first_shape and not dumped_debug:
            for _ in range(20):
                causal_conv1d_mojo.causal_conv1d_fn(
                    x, weight, bias=bias, activation=activation
                )
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(ITERS):
                    causal_conv1d_mojo.causal_conv1d_fn(
                        x, weight, bias=bias, activation=activation
                    )
                torch.cuda.synchronize()
            counts: dict[str, int] = defaultdict(int)
            mojo_total = 0.0
            for evt in prof.events():
                if evt.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                counts[evt.name] += 1
                if _is_mojo(evt.name):
                    mojo_total += evt.self_device_time_total
            print("DEBUG mojo kernels on first shape (counts over ITERS):")
            for n, c in sorted(counts.items()):
                print(f"  {c:5d}  {n}")
            print()
            mojo_us = mojo_total / ITERS
            dumped_debug = True
        else:
            mojo_us = _bench_kernel(
                lambda: causal_conv1d_mojo.causal_conv1d_fn(
                    x, weight, bias=bias, activation=activation
                )
            )

        up_us = cache.get_or_run(
            impl="upstream",
            shape=shape,
            config=cfg,
            run=lambda: _bench_kernel(
                lambda: upstream_fn(x, weight, bias=bias, activation=activation)
            ),
        )

        ratio = mojo_us / up_us if up_us else float("inf")
        print(
            f"{shape!s:>22} | {mojo_us:15.1f} | {up_us:19.1f} | {ratio:6.2f}x"
        )


if __name__ == "__main__":
    main()
