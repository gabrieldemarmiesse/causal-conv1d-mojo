"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block. Matches WARP_SIZE on NVIDIA / one wavefront half
# on AMD CDNA. Single-warp blocks simplify cooperative tile loads
# (no inter-warp barrier needed) and set up cleanly for the
# upcoming MMA refactor: m16n8k16 MMA fragments distribute neatly
# across 32 lanes.
comptime kNThreads: Int = 32


# Number of query positions handled per block. With the current
# "1 lane = 1 query" structure each lane owns one query row's
# running softmax state. When MMA lands in a follow-up commit,
# BM = 16 (= MMA_M) and the warp cooperates on each row's matmul.
comptime kBlockM: Int = 32


# K/V tile size along the seqlen-of-K dim. Cooperatively loaded
# into smem once per outer iteration; reused by every lane's
# inner compute. Equal to kNThreads so the cooperative load is
# "one row per lane".
comptime kBlockN: Int = 32
