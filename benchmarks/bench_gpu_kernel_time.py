"""Per-kernel GPU-time benchmark using torch.profiler.

Mirrors causal-conv1d-mojo's `bench_gpu_kernel_time.py`. Wraps each
implementation in a `record_function` range, calls it N times under
`torch.profiler`, walks `prof.events()` and sums per-kernel GPU time.
Reports `mojo (us/call)`, `upstream (us/call)`, and the ratio.

The mojo kernel currently only supports the simplest envelope (fp16,
head_dim=64, non-causal); the shape grid below stays inside that
envelope until more variants land. Once we expand support, more
shapes / dtypes / causal modes get added here.

    # bench all shapes
    uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py

    # one shape, faster iteration
    uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py \\
        --shape 1,512,8,64 --iters 20 --warmup 5
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

import flash_attn_mojo
from flash_attn import flash_attn_func as upstream_fn


# (batch, seqlen, nheads, head_dim). head_dim is fixed at 64 (the
# only thing the current mojo kernel supports).
FWD_SHAPES = [
    (1, 128, 8, 64),
    (1, 512, 8, 64),
    (1, 1024, 8, 64),
    (1, 2048, 8, 64),
    (2, 1024, 8, 64),
    (4, 1024, 8, 64),
    (8, 1024, 8, 64),
]


def _is_mojo_fwd(name: str) -> bool:
    # Mojo build mangles the kernel name; the substring `fwd_kernel`
    # survives. Filter out upstream's `void flash_fwd_kernel<...>` since
    # it also contains the substring.
    return "fwd_kernel" in name and not name.startswith("void")


def _is_upstream_fwd(name: str) -> bool:
    return name.startswith("void flash_fwd_kernel") or name.startswith(
        "void flash::flash_fwd_kernel"
    )


def _sum_cuda_us(prof, predicate) -> float:
    total = 0.0
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if predicate(evt.name):
            total += evt.self_device_time_total
    return total


def _bench(
    fn: Callable[[], None],
    predicate: Callable[[str], bool],
    iters: int,
    warmup: int,
) -> float:
    """Mean per-call GPU time, μs, attributed to kernels matched by `predicate`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return _sum_cuda_us(prof, predicate) / iters


def _make_fwd_call(impl: str, shape, *, dtype, device, g):
    b, l, h, d = shape
    q = torch.randn(b, l, h, d, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(b, l, h, d, generator=g).to(device=device, dtype=dtype)
    v = torch.randn(b, l, h, d, generator=g).to(device=device, dtype=dtype)

    if impl == "mojo":
        return lambda: flash_attn_mojo.flash_attn_func(q, k, v)
    if impl == "upstream":
        return lambda: upstream_fn(q, k, v)
    raise ValueError(f"unknown impl: {impl}")


def run_fwd(args, shapes, device, dtype) -> None:
    print(
        f"FWD kernel: GPU={torch.cuda.get_device_name(0)} | dtype={args.dtype} "
        f"| iters={args.iters}"
    )
    header = (
        f"{'shape (B,L,H,D)':>20} | {'mojo (us/call)':>15} | "
        f"{'upstream (us/call)':>19} | {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    g = torch.Generator(device="cpu").manual_seed(0)
    for shape in shapes:
        mojo_us = _bench(
            _make_fwd_call("mojo", shape, dtype=dtype, device=device, g=g),
            _is_mojo_fwd, args.iters, args.warmup,
        )
        up_us = _bench(
            _make_fwd_call("upstream", shape, dtype=dtype, device=device, g=g),
            _is_upstream_fwd, args.iters, args.warmup,
        )
        ratio = mojo_us / up_us if up_us > 0 else float("inf")
        shape_str = "(" + ",".join(str(s) for s in shape) + ")"
        print(
            f"{shape_str:>20} | {mojo_us:>15.2f} | {up_us:>19.2f} | "
            f"{ratio:>6.2f}x"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Single shape `B,L,H,D` (e.g. `1,512,8,64`).",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", choices=["fp16"], default="fp16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — bench requires GPU")
    device = "cuda"
    dtype = {"fp16": torch.float16}[args.dtype]

    if args.shape is not None:
        shapes = [tuple(int(x) for x in args.shape.split(","))]
    else:
        shapes = FWD_SHAPES

    run_fwd(args, shapes, device, dtype)


if __name__ == "__main__":
    main()
