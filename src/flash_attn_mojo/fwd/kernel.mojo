"""GPU forward kernel for flash-attn — direct `mma.sync.m16n8k8` for Q·Kᵀ.

Replaces the per-thread scalar Q·Kᵀ with `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`
called directly via `std.gpu.compute.mma`. The previous attempt via
`TensorCore.load_a`/`load_b` ran but produced wrong numerics because
the single-arg `load_b(b)` form, with `transpose_b=True`, internally
assumes the warp tile spans `2 * MMA_N` rows — my (MMA_N, MMA_K) slice
was too small and `load_b` was reading past the slice. Going direct
sidesteps that.

Per-lane fragment layout for `m16n8k8.row.col` (PTX ISA reference):
    A (16, 8) fp16:   lane t holds A[t/4, (t%4)*2 + {0,1}] and
                                   A[t/4 + 8, (t%4)*2 + {0,1}]  (4 fp16)
    B  (8, 8) fp16:   lane t holds B[(t%4)*2 + {0,1}, t/4]      (2 fp16)
    C (16, 8) fp32:   same lane layout as A (but fp32, 4 values)

For Q·Kᵀ we set A = Q[:, k_chunk] and B = Kᵀ[k_chunk, :] i.e. B[k, n] =
K[n, k] — so lane t loads:
    a = (Q[t/4, k+(t%4)*2], Q[t/4, k+(t%4)*2+1],
         Q[t/4+8, k+(t%4)*2], Q[t/4+8, k+(t%4)*2+1])
    b = (K[n + t/4, k+(t%4)*2], K[n + t/4, k+(t%4)*2+1])
where `k` is the head-dim chunk offset and `n` is the N-outer offset.

Softmax + P·V stay scalar per-lane for now. Lanes 0..kBlockM-1 are the
"owner lanes" for one query row each; lanes kBlockM..31 idle through
softmax + V multiply but still participate in cooperative smem loads
and the warp-collective MMA. Promoting P·V to MMA is the next commit.

Envelope: fp16, head_dim=64, no causal/dropout/alibi/softcap/window,
no MQA/GQA. Grid: `(ceildiv(seqlen_q, kBlockM), nheads, batch)`.
Block: kNThreads=32 (one warp).
"""

from std.gpu import block_idx, thread_idx, barrier, lane_id
from std.gpu.compute.mma import mma
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.gpu.memory import AddressSpace
from std.math import exp
from std.memory import stack_allocation
from std.utils.index import StaticTuple
from layout import TileTensor, TensorLayout, Coord, Idx

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
    comptime kNElts: Int = 16 // 2  # fp16 vec width = 8 lanes / 16 bytes
    comptime kNumKInner: Int = head_dim // kMmaK  # 64/8 = 8
    comptime kNumNOuter: Int = kBlockN // kMmaN  # 16/8 = 2

    var tid: Int = thread_idx.x
    var lane: Int = Int(lane_id())
    # Lane → (row-pair index, col-pair index) per m16n8k8 fragment layout.
    var lane_row: Int = lane // 4
    var lane_col: Int = (lane % 4) * 2

    var q_block_start: Int = block_idx.x * kBlockM
    var head: Int = block_idx.y
    var batch: Int = block_idx.z

    # ---- Smem buffers
    var smem_Q = stack_allocation[
        kBlockM * head_dim, dtype, address_space=AddressSpace.SHARED
    ]()
    var smem_K = stack_allocation[
        kBlockN * head_dim, dtype, address_space=AddressSpace.SHARED
    ]()
    var smem_V = stack_allocation[
        kBlockN * head_dim, dtype, address_space=AddressSpace.SHARED
    ]()
    var smem_scores = stack_allocation[
        kBlockM * kBlockN, accum_t, address_space=AddressSpace.SHARED
    ]()

    # ---- Cooperative Q load. 32 lanes × (16 rows × 64 cols) = 1024 fp16.
    # Distribution: lane t handles row (t / 2), columns half (t % 2).
    # Each lane does 4 vec-of-8 stores covering its assigned 32-element
    # half-row.
    var q_load_row: Int = tid // 2
    var q_load_half: Int = tid % 2
    var q_pos_load: Int = q_block_start + q_load_row

    if q_pos_load < seqlen:
        comptime for vec_off in range(0, head_dim // 2, kNElts):
            var col: Int = q_load_half * (head_dim // 2) + vec_off
            var qvec = q.load[width=kNElts, alignment=16](
                Coord(Idx(batch), Idx(q_pos_load), Idx(head), Idx(col))
            )
            (smem_Q + q_load_row * head_dim + col).store[alignment=16](qvec)
    else:
        var zero = SIMD[dtype, kNElts](0)

        comptime for vec_off in range(0, head_dim // 2, kNElts):
            var col: Int = q_load_half * (head_dim // 2) + vec_off
            (smem_Q + q_load_row * head_dim + col).store[alignment=16](zero)

    # Per-lane online softmax state (only lanes < kBlockM are "owners").
    var running_max: Scalar[accum_t] = Scalar[accum_t](-1.0e38)
    var running_sum: Scalar[accum_t] = 0
    var weighted_v = SIMD[accum_t, head_dim](0)

    barrier()  # smem_Q ready

    # ---- Outer loop over K/V tiles.
    var n_tiles: Int = (seqlen + kBlockN - 1) // kBlockN

    for tile_idx in range(n_tiles):
        var tile_start: Int = tile_idx * kBlockN
        var kv_load_row: Int = tid // 2
        var kv_load_half: Int = tid % 2
        var kv_global_row: Int = tile_start + kv_load_row

        # Cooperative K, V loads. Same row-per-half-lane distribution.
        if kv_global_row < seqlen:
            comptime for vec_off in range(0, head_dim // 2, kNElts):
                var col: Int = kv_load_half * (head_dim // 2) + vec_off
                var kvec = k.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(kv_global_row), Idx(head), Idx(col))
                )
                (smem_K + kv_load_row * head_dim + col).store[alignment=16](
                    kvec
                )
                var vvec = v.load[width=kNElts, alignment=16](
                    Coord(Idx(batch), Idx(kv_global_row), Idx(head), Idx(col))
                )
                (smem_V + kv_load_row * head_dim + col).store[alignment=16](
                    vvec
                )
        else:
            var zero = SIMD[dtype, kNElts](0)

            comptime for vec_off in range(0, head_dim // 2, kNElts):
                var col: Int = kv_load_half * (head_dim // 2) + vec_off
                (smem_K + kv_load_row * head_dim + col).store[alignment=16](
                    zero
                )
                (smem_V + kv_load_row * head_dim + col).store[alignment=16](
                    zero
                )

        barrier()

        # ---- Q · Kᵀ via direct MMA. For each N-outer tile in 0..kNumNOuter:
        #
        #   c_frag : SIMD[fp32, 4] per lane — accumulator for the (16, 8)
        #            output sub-tile at scores[:, n_outer*8 : n_outer*8 + 8]
        #
        # Inner k loop iterates over head_dim chunks of size kMmaK=8.
        comptime for n_outer in range(kNumNOuter):
            comptime n_off: Int = n_outer * kMmaN  # key base offset in tile
            var c_frag = SIMD[accum_t, 4](0)

            comptime for k_inner in range(kNumKInner):
                comptime k_off: Int = k_inner * kMmaK  # head_dim base offset

                # Per-lane A fragment (4 fp16) from smem_Q.
                #   a0 = Q[lane_row,     k_off + lane_col + 0]
                #   a1 = Q[lane_row,     k_off + lane_col + 1]
                #   a2 = Q[lane_row + 8, k_off + lane_col + 0]
                #   a3 = Q[lane_row + 8, k_off + lane_col + 1]
                var qrow0_base: Int = lane_row * head_dim + k_off + lane_col
                var qrow1_base: Int = (lane_row + 8) * head_dim + k_off + lane_col
                var a_frag = SIMD[dtype, 4](
                    (smem_Q + qrow0_base + 0)[0],
                    (smem_Q + qrow0_base + 1)[0],
                    (smem_Q + qrow1_base + 0)[0],
                    (smem_Q + qrow1_base + 1)[0],
                )

                # Per-lane B fragment (2 fp16) from smem_K.
                # B = Kᵀ ⇒ b[i, j] = K[j, i]. So lane t holds
                # B[(t%4)*2 + i, t/4] = K[t/4, (t%4)*2 + i].
                # Plus the n_off shift on the N (key) axis and k_off shift
                # on the K (head_dim) axis.
                var krow_base: Int = (n_off + lane_row) * head_dim + k_off + lane_col
                var b_frag = SIMD[dtype, 2](
                    (smem_K + krow_base + 0)[0],
                    (smem_K + krow_base + 1)[0],
                )

                # Avoid aliasing d and c — pass a fresh d, then assign.
                var d_frag = SIMD[accum_t, 4](0)
                mma(d_frag, a_frag, b_frag, c_frag)
                c_frag = d_frag

            # Store c_frag back to smem_scores. Each lane writes its 4 fp32
            # values into the (16, 8) sub-tile at column n_off.
            (smem_scores + lane_row * kBlockN + n_off + lane_col + 0)[0] = (
                c_frag[0]
            )
            (smem_scores + lane_row * kBlockN + n_off + lane_col + 1)[0] = (
                c_frag[1]
            )
            (
                smem_scores
                + (lane_row + 8) * kBlockN
                + n_off
                + lane_col
                + 0
            )[0] = c_frag[2]
            (
                smem_scores
                + (lane_row + 8) * kBlockN
                + n_off
                + lane_col
                + 1
            )[0] = c_frag[3]

        barrier()

        # ---- Per-lane online softmax + scalar P·V
        if lane < kBlockM:
            var q_pos: Int = q_block_start + lane

            if q_pos < seqlen:
                for k_local in range(kBlockN):
                    var k_pos: Int = tile_start + k_local

                    if k_pos >= seqlen:
                        continue

                    var score: Scalar[accum_t] = (
                        smem_scores + lane * kBlockN + k_local
                    )[0]
                    score = score * softmax_scale

                    var new_max = max(running_max, score)
                    var correction = exp(running_max - new_max)
                    var p = exp(score - new_max)

                    # Accumulate weighted V — vectorised smem reads.
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

        barrier()

    # ---- Normalise + store output (lanes < kBlockM only).
    if lane < kBlockM:
        var q_pos_out: Int = q_block_start + lane

        if q_pos_out < seqlen:
            var inv_sum = Scalar[accum_t](1) / running_sum

            comptime for i in range(head_dim):
                o[batch, q_pos_out, head, i] = (
                    weighted_v[i] * inv_sum
                ).cast[dtype]()
