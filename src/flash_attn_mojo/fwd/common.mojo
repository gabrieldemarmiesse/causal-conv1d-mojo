"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block. Each thread handles one query position; the block
# walks the seqlen dimension via the grid.
comptime kNThreads: Int = 64


# K/V tile size along the seqlen-of-K dim. Chosen equal to `kNThreads`
# so the cooperative load is "1 row per thread, no leftover". Bumping
# kBlockN past 64 starts to push smem over 16 KB for head_dim>=128 and
# costs occupancy.
comptime kBlockN: Int = 64
