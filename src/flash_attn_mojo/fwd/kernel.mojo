"""GPU forward kernel for flash-attn — minimal viable implementation.

Mirrors upstream's `flash_fwd_kernel.h` in *intent* (online softmax
attention) but at the simplest possible scope to get the pipeline
working end-to-end:

- Single dtype (fp16). bf16 / fp32 to come.
- Single `head_dim = 64`. The hot, common case for many models;
  generalising over D is a comptime sweep we'll add once correctness
  is locked.
- No causal mask, no dropout, no alibi, no softcap, no sliding
  window, no MQA/GQA (`nheads_q == nheads_kv`).
- Each thread handles **one query position**. No tiling, no smem
  cooperation — the per-thread loop streams K/V from global and runs
  the online softmax entirely in registers. Bandwidth-wasteful (every
  thread re-reads K and V) but algorithmically simple and a clean
  baseline to optimise against.

Grid: `(ceildiv(seqlen_q, kNThreads), nheads, batch)`.
Block: `kNThreads` (one warp on NVIDIA).

Returns just `o`; no softmax-lse is computed (would only be needed
for the backward, which doesn't exist yet).
"""

from std.gpu import block_idx, thread_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.math import exp
from std.utils.index import StaticTuple
from layout import TileTensor, TensorLayout, Coord, Idx

from common import kNThreads


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
def fwd_kernel[
    dtype: DType,
    head_dim: Int,
    QLayoutType: TensorLayout,
    KLayoutType: TensorLayout,
    VLayoutType: TensorLayout,
    OLayoutType: TensorLayout,
](
    seqlen: Int,
    softmax_scale: Float32,
    q: TileTensor[dtype, QLayoutType, ImmutAnyOrigin],
    k: TileTensor[dtype, KLayoutType, ImmutAnyOrigin],
    v: TileTensor[dtype, VLayoutType, ImmutAnyOrigin],
    o: TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin],
) where (
    TileTensor[dtype, QLayoutType, ImmutAnyOrigin].flat_rank == 4
    and TileTensor[dtype, KLayoutType, ImmutAnyOrigin].flat_rank == 4
    and TileTensor[dtype, VLayoutType, ImmutAnyOrigin].flat_rank == 4
    and TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin].flat_rank == 4
):
    comptime accum_t = DType.float32

    # Grid mapping:
    #   blockIdx.x → which Q-tile of length kNThreads
    #   blockIdx.y → which head (out of nheads)
    #   blockIdx.z → which batch element
    var q_pos: Int = block_idx.x * kNThreads + thread_idx.x
    var head: Int = block_idx.y
    var batch: Int = block_idx.z

    if q_pos >= seqlen:
        return

    # ---- Load Q[batch, q_pos, head, :] into accumulator-precision registers
    var q_vec = SIMD[accum_t, head_dim](0)

    comptime for i in range(head_dim):
        q_vec[i] = q[batch, q_pos, head, i].cast[accum_t]()

    # ---- Online softmax state. Starting from -inf max + 0 sum, the first
    # iteration's `correction = exp(running_max - new_max) = exp(-inf - new_max)`
    # would be 0, which correctly zeros out the (zero) prior weighted_v
    # before adding the first contribution. But `exp(-inf - finite) = 0`
    # depends on IEEE behaviour we'd rather not lean on; initialise running
    # values such that the first iteration sets them cleanly.
    var running_max: Scalar[accum_t] = Scalar[accum_t](-1.0e38)
    var running_sum: Scalar[accum_t] = 0
    var weighted_v = SIMD[accum_t, head_dim](0)

    # ---- Iterate over all key positions, online softmax + weighted V accum
    for k_pos in range(seqlen):
        # Load K[batch, k_pos, head, :]
        var k_vec = SIMD[accum_t, head_dim](0)

        comptime for i in range(head_dim):
            k_vec[i] = k[batch, k_pos, head, i].cast[accum_t]()

        # Dot product Q · K, then scale.
        var score: Scalar[accum_t] = 0

        comptime for i in range(head_dim):
            score += q_vec[i] * k_vec[i]
        score = score * softmax_scale

        # Online softmax update:
        #   new_max = max(running_max, score)
        #   correction = exp(running_max - new_max)
        #   p = exp(score - new_max)
        #   weighted_v = correction * weighted_v + p * v_vec
        #   running_sum = correction * running_sum + p
        var new_max = max(running_max, score)
        var correction = exp(running_max - new_max)
        var p = exp(score - new_max)

        # Load V[batch, k_pos, head, :] and fold into weighted_v.
        comptime for i in range(head_dim):
            var v_i = v[batch, k_pos, head, i].cast[accum_t]()
            weighted_v[i] = correction * weighted_v[i] + p * v_i

        running_sum = correction * running_sum + p
        running_max = new_max

    # ---- Normalise and store out[batch, q_pos, head, :]
    var inv_sum = Scalar[accum_t](1) / running_sum

    comptime for i in range(head_dim):
        o[batch, q_pos, head, i] = (weighted_v[i] * inv_sum).cast[dtype]()
