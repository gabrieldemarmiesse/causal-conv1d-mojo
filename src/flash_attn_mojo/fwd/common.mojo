"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block — one full warp on NVIDIA. Single-warp blocks
# simplify cooperative tile loads (no inter-warp barrier needed) and
# match the m16n8k16 MMA fragment layout naturally.
comptime kNThreads: Int = 32


# Queries per block. Matches MMA_M (= 16) so Q·Kᵀ uses exactly one
# MMA-M tile and 16 of the warp's 32 lanes are "owner lanes" for the
# per-lane softmax + P·V step (one lane per query row). The other 16
# lanes still participate in the MMA (which is warp-collective) and
# in cooperative smem loads.
comptime kBlockM: Int = 16


# K/V tile size along the seqlen-of-K dim. 16 = 2 × MMA_N so Q·Kᵀ
# produces exactly 2 MMA-N tiles of (16, 8) per outer iteration.
comptime kBlockN: Int = 16


# MMA tile dimensions (NVIDIA Ada/Ampere fp16, fp32 accumulator).
# `m16n8k8` is the largest fp16→fp32 shape Modular's stdlib currently
# exposes in `compute/arch/mma_nvidia.mojo` (the m16n8k16 path exists
# but only for bf16). Switching to bf16 would let us use the wider
# k=16 instruction; we stay on fp16 to keep the existing test grid.
comptime kMmaM: Int = 16
comptime kMmaN: Int = 8
comptime kMmaK: Int = 8
