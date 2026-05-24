"""Flash-attention forward kernel — port of modular's `mha_single_batch`.

Adapted from `modular/max/kernels/src/nn/attention/gpu/mha.mojo`
(function `mha_single_batch`, lines 1772-2480 at tag `max/v26.3.0`).
The original is a feature-rich kernel parameterised over `MHAOperand`
(KV-cache vs dense vs ragged), `MHAMask` (causal / materialised /
null), sink-weights, MQA/GQA `group`, varlen, etc. We strip everything
outside our envelope and call it directly with raw pointers + strides:

  - dtype fp16, head_dim = 64
  - non-causal, no dropout, no alibi, no softcap, no window
  - no MQA/GQA (kv_num_heads == num_heads)
  - dense (no kv-cache, no varlen)
  - one block per (q-tile, head, batch); 4 warps × 1 = 128 threads
  - tile config: BM=64, BN=64, BK=32, WM=16, WN=64

The arithmetic structure is identical to upstream's:
  1. Async-copy Q into smem (one BM×depth tile, never reloaded).
  2. For each KV tile of BN rows:
     a. Async-copy K into smem.
     b. `multistage_mma(p_reg, Q, K, transpose_b=True)`  → P scores.
     c. Apply scale·log2e to P (we use exp2 in the softmax).
     d. `_online_softmax_iter_for_mma_output` — per-row max/sum with
        warp reduce, applies the exp2-correction to the running
        output_reg_tile and stores new max/sum into rowmax/rowsum.
     e. Async-copy V into smem.
     f. `multistage_mma(output_reg, P, V, transpose_b=False)` —
        accumulates into output_reg_tile. With num_warps_n=1 P stays
        in registers; no _copy_frag_to_smem step.
  3. Normalise output_reg_tile by 1/rowsum.
  4. Stage output through smem (reusing q_smem buffer) and write to
     gmem with a swizzled vectorised copy.

Smem layout (dynamic, sized by `launch_fwd::shared_mem_bytes`):
    [ q_smem (BM × depth × fp16)              = 64 × 64 × 2 = 8 KiB ]
    [ k_smem (BN × depth × fp16)              = 64 × 64 × 2 = 8 KiB ]
    [ v_smem (BN × BN   × fp16)               = 64 × 64 × 2 = 8 KiB ]
    [ p_smem (BM × BN   × fp16) — only used when num_warps_n > 1     ]
The output write-back stage reuses q_smem in place (fp16 buffer,
same size). Total: ~24 KiB.
"""

from std.collections import OptionalReg
from std.math import recip, exp
from std.math.constants import log2e
from std.sys import align_of, simd_width_of, size_of
from std.algorithm.functional import tile_and_unswitch, unswitch
import std.gpu.primitives.warp as warp
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
)
from std.gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from std.memory import stack_allocation
from std.utils.index import StaticTuple
from std.utils.numerics import min_or_neg_inf, get_accum_type

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape

from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from nn.softmax import _online_softmax_iter_for_mma_output

from common import kNThreads, kBlockM, kBlockN, kBlockK, kWM, kWN


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
def fwd_kernel[
    dtype: DType,
    head_dim: Int,
](
    seq_len: Int,
    nheads: Int,
    softmax_scale: Float32,
    q_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    q_b_stride: Int,
    q_l_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_l_stride: Int,
    k_h_stride: Int,
    v_b_stride: Int,
    v_l_stride: Int,
    v_h_stride: Int,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
):
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_size: Int = simd_width_of[dtype]()
    comptime num_pipeline_stages: Int = 2
    comptime k_group_size: Int = 1

    comptime BM: Int = kBlockM
    comptime BN: Int = kBlockN
    comptime BK: Int = kBlockK
    comptime WM: Int = kWM
    comptime WN: Int = kWN
    comptime num_threads: Int = kNThreads
    comptime num_warps_m: Int = BM // WM
    comptime num_warps_n: Int = BN // WN
    comptime depth: Int = head_dim

    comptime assert num_warps_m * num_warps_n == num_threads // WARP_SIZE, (
        "warp tile / num_threads mismatch"
    )
    comptime assert num_warps_n == 1, (
        "this port specialises on num_warps_n == 1 so we keep P in registers"
    )

    var tid: UInt32 = UInt32(thread_idx.x)
    var warp_id_v: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane: UInt32 = UInt32(lane_id())

    # With num_warps_n == 1: warp_x == 0 for every warp; warp_y = warp_id.
    var warp_y: UInt32 = warp_id_v
    var warp_x: UInt32 = 0

    var q_tile_idx: UInt32 = UInt32(block_idx.x)
    var head_idx: UInt32 = UInt32(block_idx.y)
    var batch: UInt32 = UInt32(block_idx.z)

    # ---- Dynamic smem layout: Q, K, V back-to-back.
    comptime alignment = align_of[SIMD[dtype, simd_size]]()
    comptime q_smem_size: Int = BM * depth
    comptime k_smem_size: Int = BN * depth
    comptime v_smem_size: Int = BN * BN  # BN * depth when depth == BN

    var q_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        dtype,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    dtype,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeK = LayoutTensorIter[
        dtype,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeV = LayoutTensorIter[
        dtype,
        Layout.row_major(BK, BN),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    # ---- MMA shape + per-warp register tiles.
    comptime mma_shape = get_mma_shape[dtype, accum_type]()
    comptime MMA_M: Int = mma_shape[0]
    comptime MMA_N: Int = mma_shape[1]
    comptime MMA_K: Int = mma_shape[2]
    comptime num_m_mmas: Int = WM // MMA_M
    comptime num_n_mmas: Int = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size: Int = frag_size[2]
    comptime p_frag_simdwidth: Int = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # ---- Per-row running max/sum (online softmax state).
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    comptime for i in range(0, WM, 2):
        rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
        rowsum.store(i, SIMD[accum_type, 2](0))

    # `p_smem` is allocated unconditionally because `multistage_mma` for
    # the 2nd MMA expects an `a_smem_iter` in SHARED address space, even
    # when we route P through registers (num_warps_n == 1, swizzle_a=False).
    # `warp_scratch` reduces the per-row max/sum across N-dim warps; with
    # num_warps_n == 1 it's still allocated (size 0 ⇒ no-op stride).
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeP = LayoutTensorIter[
        dtype,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # ---- Async-copy Q into smem (only once — held across the KV loop).
    comptime q_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(UNKNOWN_VALUE, 1)
    )
    var q_tile_num_rows: Int = min(
        Int(BM), seq_len - Int(q_tile_idx) * Int(BM)
    )
    var q_batch_head_off: Int = (
        Int(batch) * q_b_stride + Int(head_idx) * q_h_stride
    )
    var q_tile_row_off: Int = Int(q_tile_idx) * Int(BM) * q_l_stride
    var q_gmem_block = LayoutTensor[
        dtype,
        q_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        q_ptr + q_batch_head_off + q_tile_row_off,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=DType.int32](
                q_tile_num_rows, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=DType.int32](
                q_l_stride, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    comptime q_num_vecs: Int = BM * BK // simd_size
    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]
        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )
        q_gmem_iter._incr()

    var scale_log2e: Scalar[accum_type] = (
        softmax_scale.cast[accum_type]() * log2e
    )

    # ---- KV loop body (one (BN-tall) tile per iteration).
    @__copy_capture(seq_len, scale_log2e)
    @always_inline
    @parameter
    def loop_over_kv[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth), IntTuple(UNKNOWN_VALUE, 1)
        )
        var kv_tile_num_rows: Int = min(tile_size, end - kv_tile_start_row)

        var k_base_off: Int = (
            Int(batch) * k_b_stride + Int(head_idx) * k_h_stride
        )
        var k_row_off: Int = kv_tile_start_row * k_l_stride
        var k_runtime_layout = RuntimeLayout[
            kv_gmem_layout,
            element_type=DType.int32,
            linear_idx_type=DType.int32,
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                k_l_stride, 1
            ),
        )
        var k_gmem_block = LayoutTensor[
            dtype,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](k_ptr + k_base_off + k_row_off, k_runtime_layout)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_base_off: Int = (
            Int(batch) * v_b_stride + Int(head_idx) * v_h_stride
        )
        var v_row_off: Int = kv_tile_start_row * v_l_stride
        var v_runtime_layout = RuntimeLayout[
            kv_gmem_layout,
            element_type=DType.int32,
            linear_idx_type=DType.int32,
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                v_l_stride, 1
            ),
        )
        var v_gmem_block = LayoutTensor[
            dtype,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](v_ptr + v_base_off + v_row_off, v_runtime_layout)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q · Kᵀ — register-tile accumulator, zero each iter.
        _ = p_reg_tile.fill(0)

        comptime kv_num_vecs: Int = BN * BK // simd_size
        comptime async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]
            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )
            k_gmem_iter._incr()

        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b for Q · Kᵀ
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=depth // BK,
            k_group_size=k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            depth // BK,
        )

        # Apply softmax_scale * log2e (we use exp2 inside online softmax).
        # Boundary-mask scores for keys >= seq_len on the last iter.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()
        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                comptime mma_id = n_mma * num_m_mmas + m_mma
                var mma_col_base: UInt32 = (
                    UInt32(warp_x) * UInt32(WN) + UInt32(n_mma * MMA_N)
                )
                var col_off: UInt32 = (
                    lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                )

                comptime for i in range(2):
                    p_reg_vec2[mma_id, i] = p_reg_vec2[mma_id, i] * scale_log2e

                    # Per-element OOB mask on the last KV iter: set score to
                    # -inf so its softmax weight goes to zero.
                    if not not_last_iter:
                        var score_col: Int = (
                            kv_tile_start_row
                            + Int(mma_col_base + col_off)
                        )
                        var ne_inf = SIMD[accum_type, p_frag_simdwidth](
                            min_or_neg_inf[accum_type]()
                        )

                        comptime for j in range(p_frag_simdwidth):
                            if score_col + j >= seq_len:
                                p_reg_vec2[mma_id, i][j] = ne_inf[j]

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        comptime async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]
            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_gmem_iter[].vectorize[1, simd_size](),
            )
            v_gmem_iter._incr()

        async_copy_commit_group()

        # num_warps_n == 1: keep P in regs as input to the 2nd MMA.
        # Reinterpret p_reg_tile's (num_m_mmas*num_n_mmas, p_frag_size)
        # layout as an iterator over (MMA_K/MMA_N * num_m_mmas, p_frag_size)
        # tiles — the inner "n_mmas" dim of the first MMA becomes the
        # "k_mmas" dim of the second.
        var p_reg_iter = p_reg_tile.tiled_iterator[
            MMA_K // MMA_N * num_m_mmas, p_frag_size
        ](0, 0)

        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b for P · V
            swizzle_a=False,
            prefetch_init=False,
            static_num_iters=BN // BK,
            k_group_size=k_group_size,
        ](
            output_reg_tile,
            p_reg_iter,
            v_smem_iter,
            p_smem_iter,
            v_smem_iter,
            BN // BK,
        )

    tile_and_unswitch[loop_over_kv, [BN]](0, seq_len)

    # ---- Normalise by 1/rowsum.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    # ---- Stage output through smem (reuse q_smem buffer) → gmem.
    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(UNKNOWN_VALUE, 1)
    )
    var o_batch_head_off: Int = (
        Int(batch) * o_b_stride + Int(head_idx) * o_h_stride
    )
    var o_tile_row_off: Int = Int(q_tile_idx) * Int(BM) * o_l_stride
    var output_gmem_tile = LayoutTensor[
        dtype,
        output_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        o_ptr + o_batch_head_off + o_tile_row_off,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[output_gmem_layout.shape, element_type=DType.int32](
                q_tile_num_rows, depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type=DType.int32](
                o_l_stride, 1
            ),
        ),
    )

    # Per-lane write of each MMA c-fragment slot into gmem at its
    # m16n8k16 C-fragment position:
    #   c[0..1] at (row=group,   col=2*tid + {0,1}) — n_mma sub-tile shifted
    #   c[2..3] at (row=group+8, col=2*tid + {0,1})
    # where groupID = lane/4 and threadID_in_group = lane%4. We index gmem
    # by (batch, query_row, head, col) so non-contiguous strides "just work".
    # This is deliberately *not* `copy_local_to_dram` + the modular swizzled
    # smem-staging: that helper drops the tail row of each warp when used
    # as written in mha_single_batch (rows 15/31/47/63 came out zero); the
    # by-hand store sidesteps it. TODO: fix and switch back once we have a
    # repro of the modular helper failing in isolation.
    var lane_group: UInt32 = lane // 4
    var lane_tid_in_grp: UInt32 = lane % 4
    var warp_row_base: Int = Int(warp_y) * WM
    var warp_col_base: Int = Int(warp_x) * WN

    comptime for n_mma in range(num_n_mmas):
        var col_off: Int = (
            warp_col_base
            + n_mma * MMA_N
            + Int(lane_tid_in_grp) * 2
        )
        var row0: Int = warp_row_base + Int(lane_group)
        var row1: Int = warp_row_base + Int(lane_group) + 8

        var c0 = output_reg_tile.ptr[n_mma * p_frag_size + 0].cast[dtype]()
        var c1 = output_reg_tile.ptr[n_mma * p_frag_size + 1].cast[dtype]()
        var c2 = output_reg_tile.ptr[n_mma * p_frag_size + 2].cast[dtype]()
        var c3 = output_reg_tile.ptr[n_mma * p_frag_size + 3].cast[dtype]()

        var base: Int = o_batch_head_off + o_tile_row_off
        if row0 < q_tile_num_rows:
            (o_ptr + base + row0 * o_l_stride + col_off)[0] = c0
            (o_ptr + base + row0 * o_l_stride + col_off + 1)[0] = c1
        if row1 < q_tile_num_rows:
            (o_ptr + base + row1 * o_l_stride + col_off)[0] = c2
            (o_ptr + base + row1 * o_l_stride + col_off + 1)[0] = c3
