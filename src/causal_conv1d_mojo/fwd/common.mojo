"""Shared constants + leaf helpers for the causal_conv1d kernels.

Mirrors upstream's `causal_conv1d_common.h`. Imported by the
`fwd`, `bwd`, `cpu`, and `native` (dispatcher) sibling modules.
"""

from std.math import exp, recip
from std.sys import size_of


# Shared by the GPU forward kernel + the GPU launcher (grid math).
comptime kNThreads: Int = 128


# Per-thread element count for the forward kernel: 16 bytes per thread per
# load. This is `8` for fp16/bf16 (16 / 2) and `4` for fp32 (16 / 4). Picking
# this from the dtype enables the widest possible global LDG (LDG.E.128) and
# the matching wide global ST. Mirrors upstream's
#   static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
@always_inline
fn kNEltsFwd[dtype: DType]() -> Int:
    return 16 // size_of[dtype]()


def _silu_f32(x: Float32) -> Float32:
    # silu(x) = x / (1 + exp(-x)). Implementing as `x * recip(1+exp(-x))`
    # so the division lowers to `mul + rcp.approx.ftz.f32` instead of
    # `div.rn.f32` (the IEEE-compliant rounded division), which on H100
    # is ~5× slower than `rcp.approx`. The accuracy loss (~1 ulp on the
    # reciprocal) is well within the dtype's representable range — all
    # 834 fwd tests pass with the same tolerances after the change.
    return x * recip(Float32(1) + exp(-x))
