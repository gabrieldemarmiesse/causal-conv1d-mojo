"""GPU forward kernel for flash-attn — smem-tiled K/V.

Improvement on the naive version: instead of every thread streaming
K and V from global for every key position, we cooperatively load
K/V tiles of `kBlockN` rows into shared memory once per outer-loop
iteration, then all threads in the block compute against the tile.
That cuts global K/V traffic by `kNThreads`× and is the single biggest
win between "doesn't tile anything" and "actually flash-attention".

Still simple-ish:
- One thread per query position (no Q tiling across threads).
- No tensor-core matmul yet (per-thread dot products in registers).
- Single dtype (fp16) and single head_dim (64).
- No causal, dropout, alibi, softcap, window, MQA/GQA.

Grid: `(ceildiv(seqlen_q, kNThreads), nheads, batch)`.
Block: `kNThreads` (one warp on NVIDIA).
"""

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.gpu.memory import AddressSpace
from std.math import exp
from std.memory import stack_allocation
from std.utils.index import StaticTuple
from layout import TileTensor, TensorLayout, Coord, Idx

from common import kNThreads, kBlockN


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
    # 16-byte vector width: 8 fp16/bf16 lanes, 4 fp32 lanes. The K/V
    # cooperative loads use this; the per-thread dot product uses
    # individual lanes (compiler is free to vectorize).
    comptime kNElts: Int = 16 // 2  # fp16-only for now

    # Grid mapping (same as before — one block per (q-tile, head, batch)).
    var q_pos: Int = block_idx.x * kNThreads + thread_idx.x
    var head: Int = block_idx.y
    var batch: Int = block_idx.z

    # ---- Shared-memory K/V tiles, contiguous in head_dim.
    #
    # Layout: smem_K[k_local, i] = K at (batch, tile_start + k_local, head, i).
    # Row-major in (k_local, i); inner stride = 1 (i-major), so consecutive
    # threads of a warp reading the same `i` slot hit adjacent banks.
    var smem_K = stack_allocation[
        kBlockN * head_dim, dtype, address_space=AddressSpace.SHARED
    ]()
    var smem_V = stack_allocation[
        kBlockN * head_dim, dtype, address_space=AddressSpace.SHARED
    ]()

    # ---- Load this thread's Q row into fp32 registers (only if it
    # corresponds to a real query position; tail threads still
    # participate in the smem dance below).
    var q_vec = SIMD[accum_t, head_dim](0)

    if q_pos < seqlen:
        comptime for i in range(head_dim):
            q_vec[i] = q[batch, q_pos, head, i].cast[accum_t]()

    # Initialise running softmax state. -1e38 is "as negative as fp32
    # gets without hitting -inf"; using -inf would force NaN out of
    # the first iteration's `exp(running_max - new_max)`.
    var running_max: Scalar[accum_t] = Scalar[accum_t](-1.0e38)
    var running_sum: Scalar[accum_t] = 0
    var weighted_v = SIMD[accum_t, head_dim](0)

    # ---- Outer loop over K/V tiles.
    var n_tiles: Int = (seqlen + kBlockN - 1) // kBlockN

    for tile_idx in range(n_tiles):
        var tile_start: Int = tile_idx * kBlockN
        var k_row: Int = tile_start + thread_idx.x  # this thread's row in K/V

        # ---- Cooperative load: each thread loads one row of K and one
        # row of V into the smem tiles. With `kBlockN == kNThreads`,
        # every thread loads exactly one row. The head_dim=64 row is
        # 8 fp16-vec-of-8 loads of 16 bytes each.
        if k_row < seqlen:
            comptime for vec_off in range(0, head_dim, kNElts):
                var k_vec = k.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(k_row), Idx(head), Idx(vec_off))
                )
                (smem_K + thread_idx.x * head_dim + vec_off).store[
                    alignment=16
                ](k_vec)

                var v_vec = v.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(k_row), Idx(head), Idx(vec_off))
                )
                (smem_V + thread_idx.x * head_dim + vec_off).store[
                    alignment=16
                ](v_vec)
        else:
            # Out-of-range row: zero the smem so the masked compute
            # below produces a -inf score (after the +scale, which is
            # finite) and a 0 weighted-v contribution. The score branch
            # handles the mask explicitly; the V branch will just
            # multiply by p ≈ 0 from the masked score.
            var zero = SIMD[dtype, kNElts](0)

            comptime for vec_off in range(0, head_dim, kNElts):
                (smem_K + thread_idx.x * head_dim + vec_off).store[
                    alignment=16
                ](zero)
                (smem_V + thread_idx.x * head_dim + vec_off).store[
                    alignment=16
                ](zero)

        barrier()

        # ---- Per-thread compute: dot Q · K[k_local] for each k_local
        # in the tile, fold into the online softmax, accumulate
        # weighted V. Only "valid" threads (q_pos < seqlen) do this;
        # tail threads sit out the compute but already did their smem
        # writes above so the other threads see complete tiles.
        if q_pos < seqlen:
            for k_local in range(kBlockN):
                var k_pos: Int = tile_start + k_local
                # Mask scores for keys past the seqlen tail. Without
                # this, a partial tile's last few rows have zeros in
                # smem_K but the score `score = 0` would softmax to a
                # nonzero contribution — wrong.
                if k_pos >= seqlen:
                    continue

                # Dot product Q · K_smem[k_local], vectorised. Each
                # comptime step pulls a `kNElts`-wide SIMD chunk out of
                # smem (one `ld.shared.v4.b32` for fp16/8-lane), casts
                # to fp32, multiplies by the matching slice of q_vec,
                # and reduces. Was previously a scalar smem load per
                # head-dim element — costly even from smem.
                var score: Scalar[accum_t] = 0

                comptime for i in range(0, head_dim, kNElts):
                    var k_chunk = (smem_K + k_local * head_dim + i).load[
                        width=kNElts, alignment=16
                    ]()
                    var k_chunk_f = k_chunk.cast[accum_t]()
                    var q_chunk = q_vec.slice[kNElts, offset=i]()
                    score += (q_chunk * k_chunk_f).reduce_add()
                score = score * softmax_scale

                # Online softmax update (same math as the naive kernel):
                #   new_max = max(running_max, score)
                #   correction = exp(running_max - new_max)
                #   p = exp(score - new_max)
                #   weighted_v = correction * weighted_v + p * V_smem[k_local]
                #   running_sum = correction * running_sum + p
                var new_max = max(running_max, score)
                var correction = exp(running_max - new_max)
                var p = exp(score - new_max)

                # Fold V_smem[k_local] into weighted_v, vectorised
                # the same way: wide smem load + SIMD FMA per chunk.
                comptime for i in range(0, head_dim, kNElts):
                    var v_chunk = (smem_V + k_local * head_dim + i).load[
                        width=kNElts, alignment=16
                    ]()
                    var v_chunk_f = v_chunk.cast[accum_t]()
                    var wv_chunk = weighted_v.slice[kNElts, offset=i]()
                    var new_wv = correction * wv_chunk + p * v_chunk_f
                    weighted_v = weighted_v.insert[offset=i](new_wv)

                running_sum = correction * running_sum + p
                running_max = new_max

        # Block-wide sync before the next tile's smem writes overwrite
        # the K/V we just consumed.
        barrier()

    # ---- Normalise and store
    if q_pos < seqlen:
        var inv_sum = Scalar[accum_t](1) / running_sum

        comptime for i in range(head_dim):
            o[batch, q_pos, head, i] = (weighted_v[i] * inv_sum).cast[dtype]()
