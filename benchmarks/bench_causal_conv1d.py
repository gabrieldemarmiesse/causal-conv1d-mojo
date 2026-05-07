"""GPU latency comparison: causal_conv1d_mojo vs upstream causal_conv1d.

Both implementations compute the same depthwise causal 1D conv; we time
forward-only on CUDA with torch.cuda.Event over ``ITERS`` iterations
after ``WARMUP`` warm-up calls, and report the median per-call latency.
"""
import argparse
import statistics

import torch

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


WARMUP = 20
ITERS = 200


# (batch, dim, seqlen, width) -- shapes representative of Mamba/SSM workloads.
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


def _bench(fn, *args, **kwargs) -> float:
    """Median per-call latency in microseconds."""
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
    for start, end in events:
        start.record()
        fn(*args, **kwargs)
        end.record()
    torch.cuda.synchronize()
    return statistics.median((s.elapsed_time(e) * 1000 for s, e in events))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--activation", choices=["none", "silu"], default="silu")
    parser.add_argument("--bias", action="store_true", default=True)
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    activation = None if args.activation == "none" else args.activation

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype={args.dtype} | activation={args.activation} | bias={args.bias}\n")

    header = f"{'shape (B,D,L,W)':>22} | {'mojo (us)':>10} | {'upstream (us)':>14} | {'mojo/upstream':>14}"
    print(header)
    print("-" * len(header))

    g = torch.Generator(device="cpu").manual_seed(0)
    for batch, dim, seqlen, width in SHAPES:
        x = torch.randn(batch, dim, seqlen, generator=g).to(device=device, dtype=dtype)
        weight = torch.randn(dim, width, generator=g).to(device=device, dtype=dtype)
        bias = torch.randn(dim, generator=g).to(device=device, dtype=dtype) if args.bias else None

        kwargs = dict(bias=bias, activation=activation)
        t_mojo = _bench(causal_conv1d_mojo.causal_conv1d_fn, x, weight, **kwargs)
        t_up = _bench(upstream_fn, x, weight, **kwargs)
        ratio = t_mojo / t_up
        print(f"{(batch, dim, seqlen, width)!s:>22} | {t_mojo:10.1f} | {t_up:14.1f} | {ratio:13.2f}x")


if __name__ == "__main__":
    main()
