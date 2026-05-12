"""Per-kernel GPU-time benchmark for the BACKWARD kernel.

Mirror of `bench_gpu_kernel_time.py` but measures bwd kernel time only,
classifying by name substring `bwd_kernel` (Mojo) vs upstream's
`void causal_conv1d_bwd_kernel`.

The autograd graph is rebuilt for every iteration (so backward runs);
we walk profiler events and only sum events whose name contains
`bwd_kernel`. Forward kernels are excluded from the report by name.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

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
WARMUP = 20
ITERS = 100


def _kind(name: str) -> str:
    if name.startswith("void causal_conv1d_bwd_kernel"):
        return "upstream"
    if "bwd" in name and "kernel" in name and not name.startswith("void"):
        return "mojo"
    return ""


def _bench_bwd(make_call, mojo: bool) -> float:
    """Mean per-call GPU time of the BWD kernel only, μs.

    make_call() returns (out, dout); we call out.backward(dout).
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
    total = 0.0
    want = "mojo" if mojo else "upstream"
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if _kind(evt.name) == want:
            total += evt.self_device_time_total
    return total / ITERS


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    dtype = torch.float16
    activation = "silu"
    g = torch.Generator(device="cpu").manual_seed(0)

    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | iters={ITERS}\n"
    )
    header = (
        f"{'shape (B,D,L,W)':>22} | {'mojo (us/call)':>15} | "
        f"{'upstream (us/call)':>19} | {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    first_shape = SHAPES[0]
    dumped_debug = False

    for batch, dim, seqlen, width in SHAPES:
        x = (
            torch.randn(batch, dim, seqlen, generator=g)
            .to(device=device, dtype=dtype)
            .requires_grad_()
        )
        weight = (
            torch.randn(dim, width, generator=g)
            .to(device=device, dtype=dtype)
            .requires_grad_()
        )
        bias = (
            torch.randn(dim, generator=g)
            .to(device=device, dtype=dtype)
            .requires_grad_()
        )
        dout = torch.randn(batch, dim, seqlen, generator=g).to(
            device=device, dtype=dtype
        )

        def call_mojo():
            x_ = x.detach().requires_grad_()
            w_ = weight.detach().requires_grad_()
            b_ = bias.detach().requires_grad_()
            out = causal_conv1d_mojo.causal_conv1d_fn(
                x_, w_, bias=b_, activation=activation
            )
            return out, dout

        def call_upstream():
            x_ = x.detach().requires_grad_()
            w_ = weight.detach().requires_grad_()
            b_ = bias.detach().requires_grad_()
            out = upstream_fn(x_, w_, bias=b_, activation=activation)
            return out, dout

        shape = (batch, dim, seqlen, width)
        if shape == first_shape and not dumped_debug:
            for _ in range(WARMUP):
                out, _d = call_mojo()
                out.backward(dout)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(ITERS):
                    out, _d = call_mojo()
                    out.backward(dout)
                torch.cuda.synchronize()
            counts: dict[str, int] = defaultdict(int)
            mojo_total = 0.0
            for evt in prof.events():
                if evt.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                counts[evt.name] += 1
                if _kind(evt.name) == "mojo":
                    mojo_total += evt.self_device_time_total
            print("DEBUG mojo bwd kernels on first shape (counts over ITERS):")
            for n, c in sorted(counts.items()):
                if "bwd" in n or "fwd" in n or "kernel" in n:
                    print(f"  {c:5d}  {n}")
            print()
            mojo_us = mojo_total / ITERS
            dumped_debug = True
        else:
            mojo_us = _bench_bwd(call_mojo, mojo=True)

        up_us = _bench_bwd(call_upstream, mojo=False)

        ratio = mojo_us / up_us if up_us else float("inf")
        print(
            f"{shape!s:>22} | {mojo_us:15.1f} | {up_us:19.1f} | {ratio:6.2f}x"
        )


if __name__ == "__main__":
    main()
