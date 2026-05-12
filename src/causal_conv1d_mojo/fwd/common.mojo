"""Shared constants + leaf helpers for the causal_conv1d kernels.

Mirrors upstream's `causal_conv1d_common.h`. Imported by the
`fwd`, `bwd`, `cpu`, and `native` (dispatcher) sibling modules.
"""

from std.math import exp
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
    return x / (Float32(1) + exp(-x))
