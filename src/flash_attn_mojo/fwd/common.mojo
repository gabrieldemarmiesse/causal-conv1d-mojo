"""Shared constants for the fwd subpackage.

Mirrors causal-conv1d-mojo's `fwd/common.mojo` layout. Imported by
the sibling `kernel.mojo` / `launch.mojo` / `variant.mojo`.
"""


# Threads per block. Each thread handles one query position; the block
# walks the seqlen dimension via the grid. 64 is one warp on NVIDIA;
# fits comfortably even for D=128 where per-thread register pressure
# matters most.
comptime kNThreads: Int = 64
