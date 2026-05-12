"""Shared constants + leaf helpers for the causal_conv1d kernels.

Mirrors upstream's `causal_conv1d_common.h`. Imported by the
`fwd`, `bwd`, `cpu`, and `native` (dispatcher) sibling modules.
"""

from std.math import exp
from std.sys import size_of


# Shared by the GPU forward kernel + the GPU launcher (grid math).
comptime kNThreads: Int = 128
# Forward: kNElts=4 (8 bytes/thread). The fwd grid is (ceildiv(seqlen,
# kNThreads*kNElts), dim, batch); raising kNElts shrinks the grid and
# costs parallelism on small seqlens, even though it would help vector
# load throughput.
comptime kNElts: Int = 4


# Backward: bwd has only one block per (B,D) (it walks the full seqlen
# via an inner chunk loop), so per-thread element count doesn't cost
# parallelism — we want it AS LARGE AS the load width allows. Upstream
# uses 8 for 16-bit dtypes and 4 for fp32; matching that here gives
# 16 bytes/thread = a single LDG.E.128 either way, doubling the per-
# chunk element count for fp16/bf16 vs the old uniform-4 setting.
fn kNEltsBwd_for[dtype: DType]() -> Int:
    return 16 // size_of[dtype]()


def _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))
