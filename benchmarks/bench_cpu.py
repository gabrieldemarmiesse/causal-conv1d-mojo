"""CPU bench: mojo vs causal_conv1d_ref vs pure-PyTorch.

`causal_conv1d_fn` is CUDA-only on the upstream package; the only
upstream code path that runs on CPU is `causal_conv1d_ref` (pure
pytorch). We compare both (and a hand-written `F.conv1d + F.silu`).
"""

import statistics
import time

import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


SHAPES = [
    (1, 256, 256, 4),
    (1, 256, 1024, 4),
    (1, 1024, 256, 4),
    (1, 1024, 1024, 4),
    (1, 1024, 2048, 4),
    (1, 2048, 1024, 4),
    (1, 2048, 2048, 4),
    (4, 1024, 1024, 4),
    (4, 2048, 1024, 4),
    (8, 1024, 1024, 4),
]
WARMUP = 5
ITERS_FWD = 50
ITERS_BWD = 25


def _make(B, D, L, W):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(B, D, L, generator=g).to(torch.float16)
    weight = torch.randn(D, W, generator=g).to(torch.float16)
    bias = torch.randn(D, generator=g).to(torch.float16)
    dout = torch.randn(B, D, L, generator=g).to(torch.float16)
    return x, weight, bias, dout


def pytorch_fwd(x, weight, bias):
    D, W = weight.shape
    L = x.shape[-1]
    return F.silu(
        F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :L]
    )


def bench_fwd(call) -> float:
    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(ITERS_FWD):
        t0 = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def bench_fwd_bwd(make_call) -> float:
    for _ in range(WARMUP):
        out, dout = make_call()
        out.backward(dout)
    samples = []
    for _ in range(ITERS_BWD):
        t0 = time.perf_counter_ns()
        out, dout = make_call()
        out.backward(dout)
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def fmt(t):
    if t >= 1000:
        return f"{t / 1000:>7.2f}ms"
    return f"{t:>7.1f}μs"


def main():
    print(
        f"CPU: {torch.get_num_threads()} threads | dtype=fp16 | "
        f"activation=silu | bias=True | width=4\n"
    )

    print("=== FORWARD (50 iters) ===")
    h = (
        f"{'shape (B, D, L)':>20} | {'mojo':>10} | {'ref':>10} | {'pytorch':>10}"
        f" | {'mojo/ref':>9} | {'mojo/pt':>9}"
    )
    print(h)
    print("-" * len(h))
    fwd_rows = []
    for B, D, L, W in SHAPES:
        x, w, b, _ = _make(B, D, L, W)
        m = bench_fwd(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, w, bias=b, activation="silu")
        )
        r = bench_fwd(lambda: causal_conv1d_ref(x, w, bias=b, activation="silu"))
        p = bench_fwd(lambda: pytorch_fwd(x, w, b))
        fwd_rows.append((m, r, p))
        print(
            f"{(B, D, L)!s:>20} | {fmt(m)} | {fmt(r)} | {fmt(p)}"
            f" | {m / r:>8.2f}x | {m / p:>8.2f}x"
        )
    ratios_r = [m / r for m, r, _ in fwd_rows]
    ratios_p = [m / p for m, _, p in fwd_rows]
    print(
        f"\nfwd summary — mojo/ref median {statistics.median(ratios_r):.2f}x "
        f"(min {min(ratios_r):.2f}, max {max(ratios_r):.2f}); "
        f"mojo/pytorch median {statistics.median(ratios_p):.2f}x"
    )

    print("\n=== FORWARD + BACKWARD (25 iters) ===")
    print(h)
    print("-" * len(h))
    bwd_rows = []
    for B, D, L, W in SHAPES:
        x, w, b, dout = _make(B, D, L, W)

        def make_call_mojo():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                w.detach().requires_grad_(),
                b.detach().requires_grad_(),
            )
            out = causal_conv1d_mojo.causal_conv1d_fn(
                x_, w_, bias=b_, activation="silu"
            )
            return out, dout

        def make_call_ref():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                w.detach().requires_grad_(),
                b.detach().requires_grad_(),
            )
            out = causal_conv1d_ref(x_, w_, bias=b_, activation="silu")
            return out, dout

        def make_call_pt():
            x_, w_, b_ = (
                x.detach().requires_grad_(),
                w.detach().requires_grad_(),
                b.detach().requires_grad_(),
            )
            out = pytorch_fwd(x_, w_, b_)
            return out, dout

        m = bench_fwd_bwd(make_call_mojo)
        r = bench_fwd_bwd(make_call_ref)
        p = bench_fwd_bwd(make_call_pt)
        bwd_rows.append((m, r, p))
        print(
            f"{(B, D, L)!s:>20} | {fmt(m)} | {fmt(r)} | {fmt(p)}"
            f" | {m / r:>8.2f}x | {m / p:>8.2f}x"
        )
    ratios_r = [m / r for m, r, _ in bwd_rows]
    ratios_p = [m / p for m, _, p in bwd_rows]
    print(
        f"\nfwd+bwd summary — mojo/ref median {statistics.median(ratios_r):.2f}x "
        f"(min {min(ratios_r):.2f}, max {max(ratios_r):.2f}); "
        f"mojo/pytorch median {statistics.median(ratios_p):.2f}x"
    )


if __name__ == "__main__":
    main()
