"""Compare three forward paths on the same workload:

  * "mojo (max)"   -- via max.experimental.torch.CustomOpLibrary
                      (causal_conv1d_mojo.causal_conv1d_fn).
  * "mojo (native)" -- direct Python -> Mojo extension that launches the
                       same GPU kernel via std.gpu.host.DeviceContext,
                       no MAX in the middle.
  * "upstream"     -- causal_conv1d.causal_conv1d_fn (Tri Dao's CUDA
                       kernel via torch.library.custom_op).

For each: report wall-clock per call (with sync after each call) and
host-only submit time (one sync at the end). The native path is
specialized for fp16 / width=4 / has_bias=True / activation="silu".
"""
from __future__ import annotations

import statistics
import time

import torch

import causal_conv1d_mojo
from causal_conv1d_mojo._native import causal_conv1d_native as native_mod
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


def call_native(x, weight, bias, out) -> None:
    native_mod.causal_conv1d_fwd_fp16_w4_silu_bias(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        torch.cuda.current_stream().cuda_stream,
    )


def bench_wall(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


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
        f"{'max wall':>9} | {'max host':>9} | "
        f"{'nat wall':>9} | {'nat host':>9} | "
        f"{'up wall':>9} | {'up host':>9}"
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
        bias = torch.randn(dim, generator=g).to(
            device=device, dtype=torch.float16
        )
        out = torch.empty_like(x)

        kw = dict(bias=bias, activation="silu")
        max_wall = bench_wall(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw)
        )
        max_host = bench_host(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw)
        )
        nat_wall = bench_wall(lambda: call_native(x, weight, bias, out))
        nat_host = bench_host(lambda: call_native(x, weight, bias, out))
        up_wall = bench_wall(lambda: upstream_fn(x, weight, **kw))
        up_host = bench_host(lambda: upstream_fn(x, weight, **kw))

        print(
            f"{(batch, dim, seqlen, width)!s:>22} | "
            f"{max_wall:8.1f}u | {max_host:8.1f}u | "
            f"{nat_wall:8.1f}u | {nat_host:8.1f}u | "
            f"{up_wall:8.1f}u | {up_host:8.1f}u"
        )


if __name__ == "__main__":
    main()
