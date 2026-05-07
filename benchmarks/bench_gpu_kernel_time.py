"""Per-kernel GPU-time benchmark using torch.profiler (CUPTI traces).

torch.profiler reports GPU time per kernel without needing the
NVIDIA performance-counter permission ncu requires. We run warmup
outside the profiler scope, then for each shape call our Mojo-backed
impl and the upstream CUDA op N times each, wrapped in NVTX-style
record_function ranges. The profiler then reports per-kernel cumulative
GPU time, which we group back into "mojo" / "upstream" buckets.
"""
from __future__ import annotations

import re
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


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


def _kind(name: str) -> str:
    """Classify a CUDA kernel symbol as mojo (our op) or upstream (Tri Dao's).

    Mojo's `foreach` lowers to symbols like `std_algorithm_backend_gpu_el<hash>`;
    the upstream Tri Dao op is `causal_conv1d_fwd_kernel`. We also count the
    small Fill kernel from the placeholder `torch.zeros(1,1,1)` we currently
    pass for absent `initial_states` -- it runs once per call inside our wrapper.
    """
    if "causal_conv1d_fwd_kernel" in name:
        return "upstream"
    if name.startswith("std_algorithm_backend_gpu_el"):
        return "mojo"
    if "FillFunctor" in name:
        return "mojo"
    return ""


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.float16
    activation = "silu"
    g = torch.Generator(device="cpu").manual_seed(0)

    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | activation=silu | bias=True | iters={ITERS}\n")
    header = f"{'shape (B,D,L,W)':>22} | {'mojo (us/call)':>15} | {'upstream (us/call)':>19} | {'ratio':>7}"
    print(header)
    print("-" * len(header))

    for batch, dim, seqlen, width in SHAPES:
        x = torch.randn(batch, dim, seqlen, generator=g).to(device=device, dtype=dtype)
        weight = torch.randn(dim, width, generator=g).to(device=device, dtype=dtype)
        bias = torch.randn(dim, generator=g).to(device=device, dtype=dtype)

        # Warmup: not under the profiler.
        for _ in range(20):
            causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation=activation)
            upstream_fn(x, weight, bias=bias, activation=activation)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        ) as prof:
            for _ in range(ITERS):
                with record_function("mojo"):
                    causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation=activation)
                with record_function("upstream"):
                    upstream_fn(x, weight, bias=bias, activation=activation)
            torch.cuda.synchronize()

        if (batch, dim, seqlen, width) == SHAPES[0]:
            counts: dict[str, int] = defaultdict(int)
            for evt in prof.events():
                if evt.device_type == torch.autograd.DeviceType.CUDA:
                    counts[evt.name] += 1
            print("DEBUG kernels on first shape (counts over both impls x ITERS):")
            for n, c in sorted(counts.items()):
                print(f"  {c:5d}  {n}")
            print()

        # key_averages is per-op rolled up; use events for per-kernel attribution.
        totals: dict[str, int] = defaultdict(int)
        for evt in prof.events():
            if evt.device_type != torch.autograd.DeviceType.CUDA:
                continue
            kind = _kind(evt.name)
            if not kind:
                continue
            totals[kind] += evt.self_device_time_total

        mojo_us = totals["mojo"] / ITERS
        up_us = totals["upstream"] / ITERS
        ratio = mojo_us / up_us if up_us else float("inf")
        print(f"{(batch, dim, seqlen, width)!s:>22} | {mojo_us:15.1f} | {up_us:19.1f} | {ratio:6.2f}x")


if __name__ == "__main__":
    main()
