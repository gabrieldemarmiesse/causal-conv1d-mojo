"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block — one full warp on NVIDIA. Single-warp blocks
# simplify cooperative tile loads (no inter-warp barrier needed) and
# match the m16n8k16 MMA fragment layout naturally.
comptime kNThreads: Int = 32


# Queries per block. = MMA_M so Q·Kᵀ uses exactly one MMA-M tile.
# Lanes 0..kBlockM-1 are "owner lanes" for one query row each in the
# per-lane softmax + P·V step. Remaining lanes participate in the
# warp-collective MMA + cooperative smem loads but sit out compute.
comptime kBlockM: Int = 16


# K/V tile size along the seqlen-of-K dim. = 2 × MMA_N so Q·Kᵀ
# produces two (16, 8) MMA-N tiles per outer iter, filling the
# (kBlockM, kBlockN) = (16, 16) score matrix.
comptime kBlockN: Int = 16


# MMA tile dimensions (NVIDIA Ada/Ampere fp16, fp32 accumulator).
# `m16n8k8` is the largest fp16→fp32 shape Modular's stdlib currently
# exposes in `compute/arch/mma_nvidia.mojo` (the m16n8k16 path exists
# but only for bf16). Switching to bf16 would let us use the wider
# k=16 instruction; we stay on fp16 to keep the existing test grid.
comptime kMmaM: Int = 16
comptime kMmaN: Int = 8
comptime kMmaK: Int = 8
