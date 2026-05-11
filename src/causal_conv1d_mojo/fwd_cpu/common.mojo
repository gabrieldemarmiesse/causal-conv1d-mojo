"""Shared constants + leaf helpers for the causal_conv1d kernels.

Mirrors upstream's `causal_conv1d_common.h`. Imported by the
`fwd`, `bwd`, `cpu`, and `native` (dispatcher) sibling modules.
"""

from std.math import exp


# Shared by the GPU forward kernel + the GPU launcher (grid math).
comptime kNThreads: Int = 128
# Forward: kNElts=4 (8 bytes/thread). The fwd grid is (ceildiv(seqlen,
# kNThreads*kNElts), dim, batch); raising kNElts shrinks the grid and
# costs parallelism on small seqlens, even though it would help vector
# load throughput.
comptime kNElts: Int = 4
# Backward: bwd has only one block per (B,D) (it walks the full seqlen
# via an inner chunk loop), so per-thread element count doesn't cost
# parallelism. Kept at 4 to match the fwd alignment story (LDG.E.U64 for
# fp16/bf16, LDG.E.128 for fp32 with the alignment=16 promise).
comptime kNEltsBwd: Int = 4


def _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))
