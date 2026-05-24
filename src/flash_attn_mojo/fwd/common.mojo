"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block — one full warp on NVIDIA. Single-warp blocks
# simplify cooperative tile loads (no inter-warp barrier needed) and
# match the m16n8k16 MMA fragment layout naturally.
comptime kNThreads: Int = 32


# Queries per block. Currently == kNThreads for the "1 lane = 1 query
# position" structure of the smem-tiled kernel. The MMA refactor (WIP)
# lands this at 16 (= MMA_M).
comptime kBlockM: Int = 32


# K/V tile size along the seqlen-of-K dim. Cooperatively loaded into
# smem once per outer iter; reused by every lane's inner compute.
# Equal to kNThreads so each thread loads exactly one row.
comptime kBlockN: Int = 32


# MMA tile dimensions (NVIDIA Ada/Ampere fp16, fp32 accumulator).
# `m16n8k8` is the largest fp16→fp32 shape Modular's stdlib currently
# exposes in `compute/arch/mma_nvidia.mojo` (the m16n8k16 path exists
# but only for bf16). Switching to bf16 would let us use the wider
# k=16 instruction; we stay on fp16 to keep the existing test grid.
comptime kMmaM: Int = 16
comptime kMmaN: Int = 8
comptime kMmaK: Int = 8
