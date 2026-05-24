"""Shared comptime constants for the fwd subpackage.

Tile sizes are picked to match modular's `mha_single_batch` config
for non-MQA fp16 on Ampere/Ada (num_warps_m=4, num_warps_n=1):
    BM = 64   queries per block
    BN = 64   keys per inner KV tile (= depth)
    BK = 32   reduction (head_dim) tile per multistage_mma step
    WM = 16   queries per warp (= MMA_M)
    WN = 64   keys per warp     (= BN, since num_warps_n == 1)
"""


comptime kNThreads: Int = 128  # 4 warps × 32 = num_warps_m * num_warps_n * WARP_SIZE

comptime kBlockM: Int = 64
comptime kBlockN: Int = 64
comptime kBlockK: Int = 32

comptime kWM: Int = 16
comptime kWN: Int = 64
