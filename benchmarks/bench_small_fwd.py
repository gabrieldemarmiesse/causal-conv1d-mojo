"""Tight iteration loop for the small-shape MPS fwd path.

Shape: (B, D, L, W) = (1, 256, 64, 4) fp16, bias, silu — the regime
where launch overhead dominates and the current Mojo path loses ~10×
to pure PyTorch in plot_bench. Re-run after every kernel/launch tweak
to see the impact in a few seconds.

Usage:
    pixi run -e bench python benchmarks/bench_small_fwd.py
or:
    uv run python benchmarks/bench_small_fwd.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

import causal_conv1d_mojo


B, D, L, W = 1, 256, 64, 4
DTYPE = torch.float16
DEVICE = "mps"
WARMUP = 50
ITERS = 500


def bench(name, fn):
    """Wall-clock μs/iter, sync'd around the whole iter range."""
    for _ in range(WARMUP):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    us = (time.perf_counter_ns() - t0) / 1000.0 / ITERS
    print(f"  {name:36s}  {us:8.2f} μs/iter")


def main():
    torch.manual_seed(0)
    x = torch.randn(B, D, L, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(D, W, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE)
    torch.mps.synchronize()

    # Correctness vs pytorch reference.
    out_mojo = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu"
    )

    def pt_fwd():
        pre = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :L]
        return F.silu(pre)

    out_pt = pt_fwd()
    diff = (out_mojo.float() - out_pt.float()).abs().max().item()
    print(f"correctness (mojo vs pt): max_diff = {diff:.3e}  (fp16 tol ~2e-2)\n")

    print(f"Shape ({B},{D},{L},{W}), fp16+bias+silu, {ITERS} iters")
    print("-" * 56)
    bench(
        "mojo (current)",
        lambda: causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, bias=bias, activation="silu"
        ),
    )
    bench("pytorch (F.conv1d + F.silu)", pt_fwd)

    print()
    print("Cost decomposition (no kernel):")
    print("-" * 56)
    bench("torch.mps.synchronize() alone", lambda: torch.mps.synchronize())
    bench("torch.empty_like(x) only", lambda: torch.empty_like(x))

    from causal_conv1d_mojo._mps import gpu_address  # noqa

    bench(
        "gpu_address(x) (Obj-C msgSend cost)",
        lambda: gpu_address(x),
    )

    # Probe what's inside causal_conv1d_fn. The forward auto-wraps
    # `_CausalConv1dFn.apply(x, weight, bias, ...)` — exercise the
    # path bypasses (no autograd op needed).
    from causal_conv1d_mojo.fwd import native_fwd_mps

    out_buf = torch.empty_like(x)
    bench(
        "native_fwd_mps direct (skip autograd)",
        lambda: native_fwd_mps(x, weight, bias, None, None, out_buf, True),
    )

    # Strip torch.mps.synchronize from native_fwd_mps to see what its
    # call_fwd costs in isolation.
    from causal_conv1d_mojo.fwd._jit import call_fwd
    from causal_conv1d_mojo._dtype import _DTYPE_CODE

    def call_fwd_only():
        call_fwd(
            (
                gpu_address(x),
                gpu_address(weight),
                gpu_address(bias),
                gpu_address(out_buf),
                x.shape[0],
                x.shape[1],
                x.shape[2],
                x.stride(0),
                x.stride(1),
                x.stride(2),
                weight.stride(0),
                weight.stride(1),
                out_buf.stride(0),
                out_buf.stride(1),
                out_buf.stride(2),
                1,
                1,
                _DTYPE_CODE[x.dtype],
                0,
                0,
                0,
                0,
                0,
                weight.shape[1],
                0,
                0,
                0,
                0,
                0,
            )
        )

    bench("call_fwd only (no torch sync, kernel + ctx.sync)", call_fwd_only)

    # If kernel itself is cheap, batching many enqueues per sync should
    # amortise the sync cost.
    def call_fwd_x10():
        for _ in range(10):
            call_fwd_only()

    bench("call_fwd x10 (each call internally syncs)", call_fwd_x10)

    # No-op Mojo kernel — measures launch + sync floor with no work.
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.ExtensionFileLoader("noop_ext", "/tmp/noop_ext.so")
    spec = importlib.util.spec_from_loader("noop_ext", loader)
    noop_ext = importlib.util.module_from_spec(spec)
    loader.exec_module(noop_ext)
    print()
    print("Mojo no-op floor:")
    print("-" * 56)
    bench("just DeviceContext()", lambda: noop_ext.just_devicecontext())
    bench(
        "DeviceContext + compile_function (cached)",
        lambda: noop_ext.launch_noop_no_enqueue(),
    )
    bench("+ enqueue noop (no sync)", lambda: noop_ext.launch_noop_no_sync())
    bench("+ ctx.synchronize() (full noop path)", lambda: noop_ext.launch_noop())

    # Vary the shape to see if time scales with the work or stays
    # constant (indicating launch / driver overhead dominates).
    print()
    print("Scaling probe (mojo current):")
    print("-" * 56)
    for shape in [
        (1, 64, 64, 4),
        (1, 256, 64, 4),
        (1, 256, 256, 4),
        (1, 1024, 256, 4),
        (1, 1024, 1024, 4),
    ]:
        Bs, Ds, Ls, Ws = shape
        xx = torch.randn(Bs, Ds, Ls, dtype=DTYPE, device=DEVICE)
        ww = torch.randn(Ds, Ws, dtype=DTYPE, device=DEVICE)
        bb = torch.randn(Ds, dtype=DTYPE, device=DEVICE)
        torch.mps.synchronize()
        bench(
            f"  shape {shape}",
            lambda xx=xx, ww=ww, bb=bb: causal_conv1d_mojo.causal_conv1d_fn(
                xx, ww, bias=bb, activation="silu"
            ),
        )


if __name__ == "__main__":
    main()
