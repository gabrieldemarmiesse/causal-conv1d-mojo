"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block — one full warp on NVIDIA. Single-warp blocks
# simplify cooperative tile loads (no inter-warp barrier needed) and
# match the m16n8k16 MMA fragment layout naturally.
comptime kNThreads: Int = 32


# Queries per block. Currently == kNThreads because the algorithm is
# "1 lane = 1 query position"; the MMA refactor lands this at 16 (=
# MMA_M) so Q·Kᵀ fits exactly one MMA-M tile.
comptime kBlockM: Int = 32


# K/V tile size along the seqlen-of-K dim. Cooperatively loaded
# into smem once per outer iteration; reused by every lane's
# inner compute.
comptime kBlockN: Int = 32


# MMA tile dimensions (NVIDIA Ada/Ampere fp16, fp32 accumulator).
# `m16n8k16` is the standard shape; any other choice on this hardware
# would either be emulated or much slower.
comptime kMmaM: Int = 16
comptime kMmaN: Int = 8
comptime kMmaK: Int = 16
