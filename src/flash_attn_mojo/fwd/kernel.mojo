"""GPU forward kernel for flash-attn — MMA-based Q·Kᵀ + scalar softmax/V.

Hybrid step on the way to a fully MMA-based implementation:

- Q·Kᵀ uses `TensorCore` MMA (`m16n8k16`, fp16 inputs, fp32 accumulator).
- The (BM=16, kBlockN=16) score matrix is written to smem via
  `mma_op.store_d`.
- Per-lane online softmax + scalar P·V follow, reading scores from
  smem. P·V will move to MMA in the next commit.

Block: one warp (kNThreads=32). Of these 32 lanes, only lanes 0..15
participate in the softmax + P·V step (one query position each); all
32 lanes cooperate on the MMA and the smem loads.

Grid: `(ceildiv(seqlen_q, kBlockM), nheads, batch)` with kBlockM=16.

Envelope (same as before): fp16, head_dim=64, no causal/dropout/
alibi/softcap/window, no MQA/GQA. Anything outside raises in the
Python wrapper.
"""

from std.gpu import block_idx, thread_idx, barrier, lane_id
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.math import exp
from std.utils import Index
from std.utils.index import StaticTuple
from layout import (
    Layout,
    LayoutTensor,
    TileTensor,
    TensorLayout,
    Coord,
    Idx,
)
from layout.tensor_core import TensorCore

from common import kNThreads, kBlockM, kBlockN, kMmaM, kMmaN, kMmaK


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
    # 16-byte vector width: 8 fp16/bf16 lanes, 4 fp32 lanes.
    comptime kNElts: Int = 16 // 2  # fp16 only for now
    # Number of MMAs along the head_dim (K) axis to compute one Q·Kᵀ output:
    # head_dim / kMmaK. For head_dim=64, kMmaK=16 → 4 inner MMAs.
    comptime kNumKInnerMMAs: Int = head_dim // kMmaK
    # Number of MMA-N tiles to cover kBlockN keys. kBlockN=16, kMmaN=8 → 2.
    comptime kNumNOuterMMAs: Int = kBlockN // kMmaN

    var tid: Int = thread_idx.x
    var lane: Int = Int(lane_id())

    # Block coordinates.
    var q_block_start: Int = block_idx.x * kBlockM
    var head: Int = block_idx.y
    var batch: Int = block_idx.z

    # ---- Shared-memory allocations
    #
    # smem_Q: query tile, loaded once per block, reused for every K/V tile.
    # smem_K, smem_V: key / value tile, refreshed per outer iter.
    # smem_scores: (BM, kBlockN) fp32 — destination of mma_op.store_d, then
    #              source of per-lane softmax + P·V.
    var smem_Q = LayoutTensor[
        dtype,
        Layout.row_major(kBlockM, head_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var smem_K = LayoutTensor[
        dtype,
        Layout.row_major(kBlockN, head_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var smem_V = LayoutTensor[
        dtype,
        Layout.row_major(kBlockN, head_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var smem_scores = LayoutTensor[
        accum_t,
        Layout.row_major(kBlockM, kBlockN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ---- Cooperative Q load
    # 32 lanes loading 16×64 fp16 = 1024 fp16. Per lane = 32 fp16 = 4
    # fp16-vec-of-8. Distribution: lane t handles row (t // 2), columns
    # (t % 2) * 32 .. (t % 2) * 32 + 32. Each pass = 1 vec of 8 fp16, so
    # 4 passes total (covering the 32-element half-row per lane).
    var q_row: Int = tid // 2
    var q_col_half: Int = tid % 2  # 0 or 1 → covers cols 0..31 or 32..63
    var q_pos_load: Int = q_block_start + q_row

    if q_pos_load < seqlen:
        comptime for vec_off_in_half in range(0, head_dim // 2, kNElts):
            var col: Int = q_col_half * (head_dim // 2) + vec_off_in_half
            var q_vec_load = q.load[width=kNElts, alignment=16](
                Coord(Idx(batch), Idx(q_pos_load), Idx(head), Idx(col))
            )
            # Write into smem_Q[q_row, col : col + kNElts]
            var smem_q_dst = smem_Q.ptr + q_row * head_dim + col
            smem_q_dst.store[alignment=16](q_vec_load)
    else:
        # Zero-pad the tail Q rows so the MMA's scores for out-of-range
        # queries are well-defined (we mask them after store_d anyway).
        var zero = SIMD[dtype, kNElts](0)

        comptime for vec_off_in_half in range(0, head_dim // 2, kNElts):
            var col: Int = q_col_half * (head_dim // 2) + vec_off_in_half
            var smem_q_dst = smem_Q.ptr + q_row * head_dim + col
            smem_q_dst.store[alignment=16](zero)

    # ---- MMA op for Q · Kᵀ. transpose_b=True so we can load K in its
    # natural (key, head_dim) layout and let TensorCore handle the
    # transpose for the matmul.
    var mma_op = TensorCore[
        accum_t, dtype, Index(kMmaM, kMmaN, kMmaK), transpose_b=True
    ]()

    # Per-lane online softmax state — only valid for lanes 0..kBlockM-1
    # (which correspond to query rows 0..kBlockM-1). Other lanes' state
    # is unused but cheap to keep symmetric.
    var running_max: Scalar[accum_t] = Scalar[accum_t](-1.0e38)
    var running_sum: Scalar[accum_t] = 0
    var weighted_v = SIMD[accum_t, head_dim](0)

    barrier()  # smem_Q populated

    # ---- Outer loop over K/V tiles along the seqlen-of-K dim.
    var n_tiles: Int = (seqlen + kBlockN - 1) // kBlockN

    for tile_idx in range(n_tiles):
        var tile_start: Int = tile_idx * kBlockN

        # Cooperative load: K, V tiles into smem. Same row-per-half-lane
        # distribution as the Q load above.
        var k_row: Int = tid // 2
        var k_col_half: Int = tid % 2
        var k_global_row: Int = tile_start + k_row

        if k_global_row < seqlen:
            comptime for vec_off_in_half in range(0, head_dim // 2, kNElts):
                var col: Int = k_col_half * (head_dim // 2) + vec_off_in_half
                var k_vec_load = k.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(k_global_row), Idx(head), Idx(col))
                )
                (smem_K.ptr + k_row * head_dim + col).store[alignment=16](
                    k_vec_load
                )

                var v_vec_load = v.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(k_global_row), Idx(head), Idx(col))
                )
                (smem_V.ptr + k_row * head_dim + col).store[alignment=16](
                    v_vec_load
                )
        else:
            var zero = SIMD[dtype, kNElts](0)

            comptime for vec_off_in_half in range(0, head_dim // 2, kNElts):
                var col: Int = k_col_half * (head_dim // 2) + vec_off_in_half
                (smem_K.ptr + k_row * head_dim + col).store[alignment=16](
                    zero
                )
                (smem_V.ptr + k_row * head_dim + col).store[alignment=16](
                    zero
                )

        barrier()

        # ---- Q · Kᵀ via MMA
        # Output: scores (kBlockM, kBlockN) = (16, 16). Two MMA-N tiles
        # of (16, 8) along N, four MMA-K tiles of (16, 16) along K.
        # c_reg accumulator: one per N-outer tile.
        #
        # On the modular `naive_tensor` template, the c_reg is
        # `LayoutTensor[accum_t, Layout.row_major(1, frag_size)]` where
        # frag_size = MMA_M * MMA_N / WARP_SIZE. For m16n8 on 32-lane
        # warp: 16*8/32 = 4 fp32 per lane.
        comptime frag_size: Int = kMmaM * kMmaN // WARP_SIZE

        comptime for n_outer in range(kNumNOuterMMAs):
            var c_reg = (
                LayoutTensor[
                    accum_t,
                    Layout.row_major(1, frag_size),
                    MutAnyOrigin,
                    address_space=AddressSpace.LOCAL,
                ]
                .stack_allocation()
                .fill(0)
            )

            comptime for k_inner in range(kNumKInnerMMAs):
                # Q tile: (kMmaM=16, kMmaK=16) at (0, k_inner) of smem_Q
                # which has shape (kBlockM=16, head_dim=64).
                var q_warp_tile = smem_Q.tile[kMmaM, kMmaK](0, k_inner)
                # K tile: (kMmaN=8, kMmaK=16) at (n_outer, k_inner) of smem_K
                # which has shape (kBlockN=16, head_dim=64). transpose_b
                # means load_b expects the B matrix in its untransposed
                # layout — i.e. it sees an (8, 16) slice from K's natural
                # shape, equivalent to a (16, 8) slice of Kᵀ.
                var k_warp_tile = smem_K.tile[kMmaN, kMmaK](n_outer, k_inner)

                var a_reg = mma_op.load_a(q_warp_tile)
                var b_reg = mma_op.load_b(k_warp_tile)

                var d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)
                c_reg.copy_from(d_reg)

            # Store the (16, 8) score tile into smem_scores at [:, n_outer*8:].
            var scores_dst = smem_scores.tile[kMmaM, kMmaN](0, n_outer)
            mma_op.store_d(scores_dst, c_reg)

        barrier()

        # ---- Per-lane online softmax + scalar P·V
        # Only lanes 0..kBlockM-1 do meaningful work; they each own one
        # query row from this block. Lanes kBlockM..kNThreads-1 sit idle
        # for this section (they still participated in the MMA above).
        if lane < kBlockM:
            var q_pos: Int = q_block_start + lane

            if q_pos < seqlen:
                for k_local in range(kBlockN):
                    var k_pos: Int = tile_start + k_local

                    if k_pos >= seqlen:
                        continue

                    # LayoutTensor element-indexing returns a SIMD of
                    # `element_size`; index [0] to get a scalar.
                    var score: Scalar[accum_t] = (
                        smem_scores[lane, k_local][0].cast[accum_t]()
                    )
                    score = score * softmax_scale

                    var new_max = max(running_max, score)
                    var correction = exp(running_max - new_max)
                    var p = exp(score - new_max)

                    # Fold smem_V[k_local, :] into weighted_v.
                    comptime for i in range(0, head_dim, kNElts):
                        var v_chunk = (smem_V.ptr + k_local * head_dim + i).load[
                            width=kNElts, alignment=16
                        ]()
                        var v_chunk_f = v_chunk.cast[accum_t]()
                        var wv_chunk = weighted_v.slice[kNElts, offset=i]()
                        var new_wv = correction * wv_chunk + p * v_chunk_f
                        weighted_v = weighted_v.insert[offset=i](new_wv)

                    running_sum = correction * running_sum + p
                    running_max = new_max

        barrier()

    # ---- Normalise and store output. Lanes 0..kBlockM-1 each handle one
    # query row's output.
    if lane < kBlockM:
        var q_pos_out: Int = q_block_start + lane

        if q_pos_out < seqlen:
            var inv_sum = Scalar[accum_t](1) / running_sum

            comptime for i in range(head_dim):
                o[batch, q_pos_out, head, i] = (
                    weighted_v[i] * inv_sum
                ).cast[dtype]()
