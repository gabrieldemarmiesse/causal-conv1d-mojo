"""Shared constants + leaf helpers for the fwd subpackage.

Mirrors upstream's `causal_conv1d_common.h`. Imported by the
sibling `kernel.mojo` / `launch.mojo` / `variant.mojo` within
this subpackage.
"""

from std.sys import size_of


# Shared by the GPU forward kernel + the GPU launcher (grid math).
comptime kNThreads: Int = 128


# Per-thread element count for the forward kernel: 16 bytes per thread per
# load. This is `8` for fp16/bf16 (16 / 2) and `4` for fp32 (16 / 4). Picking
# this from the dtype enables the widest possible global LDG (LDG.E.128) and
# the matching wide global ST. Mirrors upstream's
#   static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
@always_inline
def kNEltsFwd[dtype: DType]() -> Int:
    return 16 // size_of[dtype]()
