"""Extensive forward-only bench: mojo vs upstream vs pure PyTorch."""
import statistics
import time

import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


SHAPES = [
    (1, 256, 256, 4),
    (1, 256, 1024, 4),
    (1, 256, 4096, 4),
    (1, 1024, 256, 4),
    (1, 1024, 512, 4),
    (1, 1024, 1024, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 4096, 4),
    (1, 1024, 8192, 4),
    (1, 1024, 16384, 4),
    (1, 2048, 1024, 4),
    (1, 2048, 2048, 4),
    (1, 2048, 4096, 4),
    (1, 2048, 8192, 4),
    (1, 4096, 1024, 4),
    (1, 4096, 2048, 4),
    (1, 4096, 4096, 4),
    (1, 4096, 8192, 4),
    (4, 1024, 2048, 4),
    (4, 2048, 2048, 4),
    (4, 4096, 1024, 4),
    (4, 4096, 2048, 4),
    (4, 4096, 4096, 4),
    (8, 1024, 2048, 4),
    (8, 2048, 2048, 4),
    (8, 2048, 4096, 4),
    (8, 4096, 2048, 4),
    (16, 1024, 2048, 4),
    (16, 2048, 2048, 4),
    (32, 1024, 1024, 4),
    (32, 2048, 1024, 4),
]
WARMUP = 25
ITERS = 500


def _make(B, D, L, W):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(B, D, L, generator=g).to("cuda", torch.float16)
    weight = torch.randn(D, W, generator=g).to("cuda", torch.float16)
    bias = torch.randn(D, generator=g).to("cuda", torch.float16)
    return x, weight, bias


def _pytorch_fwd(x, weight, bias):
    D, W = weight.shape
    L = x.shape[-1]
    return F.silu(F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :L])


def bench_one(call) -> float:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def fmt_us(t):
    if t >= 1000:
        return f"{t/1000:>7.2f}ms"
    return f"{t:>7.1f}μs"


def main() -> None:
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | dtype=fp16 | "
        f"activation=silu | bias=True | width=4 | iters={ITERS} (forward only)\n"
    )
    h = (
        f"{'shape (B, D, L)':>20} | "
        f"{'mojo':>10} | {'upstream':>10} | {'pytorch':>10} | "
        f"{'mojo/up':>7} | {'mojo/pt':>7}"
    )
    print(h)
    print("-" * len(h))

    rows = []
    for B, D, L, W in SHAPES:
        x, weight, bias = _make(B, D, L, W)
        m = bench_one(lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu"))
        u = bench_one(lambda: upstream_fn(x, weight, bias=bias, activation="silu"))
        p = bench_one(lambda: _pytorch_fwd(x, weight, bias))
        rows.append((B, D, L, m, u, p))
        print(f"{(B, D, L)!s:>20} | {fmt_us(m)} | {fmt_us(u)} | {fmt_us(p)} | {m/u:>6.2f}x | {m/p:>6.2f}x")

    ratios_up = [m / u for _, _, _, m, u, _ in rows]
    ratios_pt = [m / p for _, _, _, m, _, p in rows]
    print()
    print(
        f"summary: {len(rows)} shapes — "
        f"mojo/upstream median {statistics.median(ratios_up):.2f}x "
        f"(min {min(ratios_up):.2f}, max {max(ratios_up):.2f}); "
        f"mojo/pytorch median {statistics.median(ratios_pt):.2f}x "
        f"(min {min(ratios_pt):.2f}, max {max(ratios_pt):.2f})"
    )


if __name__ == "__main__":
    main()
