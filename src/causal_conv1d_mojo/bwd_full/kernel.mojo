"""GPU fused backward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_bwd.cu`. The launcher lives in
`causal_conv1d_native.mojo`; this file holds the kernel + the warp /
block reduction helpers it depends on.
"""

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp
from std.memory import stack_allocation
from std.atomic import Atomic, Ordering
from std.sys import llvm_intrinsic
from layout import TileTensor, TensorLayout, Idx, Coord

from common import kNThreads


@always_inline
def _shfl_xor_f32(val: Float32, offset: UInt32) -> Float32:
    """One inlined `shfl.sync.bfly.b32`, fp32.

    Why: the stdlib chain `block.sum -> warp.sum -> shuffle_xor -> _shuffle`
    is `@always_inline` at every level on the Mojo side, but on sm_89 ptxas
    still outlines each shfl into a `__cuda_sm70_shflsync_bfly` helper. SASS
    diff vs. upstream's bwd showed 45 `CALL.REL.NOINC` to that helper. By
    issuing the LLVM intrinsic from a single small leaf and force-inlining
    every wrapper around it, ptxas stops outlining and generates the bare
    SHFL.BFLY (matching upstream).
    """
    return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.f32", Float32](
        Int32(-1), val, offset, Int32(31)
    )


@always_inline
def _warp_sum_f32(val: Float32) -> Float32:
    """5-step butterfly warp reduction, fp32. All lanes hold the warp's sum."""
    var v = val
    v += _shfl_xor_f32(v, UInt32(16))
    v += _shfl_xor_f32(v, UInt32(8))
    v += _shfl_xor_f32(v, UInt32(4))
    v += _shfl_xor_f32(v, UInt32(2))
    v += _shfl_xor_f32(v, UInt32(1))
    return v


@always_inline
def _block_sum_f32[block_size: Int](val: Float32) -> Float32:
    """Block-level fp32 sum specialised for our backward kernel.

    Equivalent to `gpu_block.sum[block_size=block_size, broadcast=False]` for
    fp32 scalars, but everything from the LLVM shuffle intrinsic up is in
    one translation unit and `@always_inline`. The cross-warp reduction is
    gated on `warp == 0` (only one warp's worth of shfl work, like upstream).

    Only thread 0 holds the meaningful result (broadcast=False).
    """
    comptime assert block_size >= 32 and block_size % 32 == 0, (
        "block_size must be a multiple of warp size (32)"
    )
    comptime n_warps: Int = block_size // 32

    # Step 1: per-warp butterfly reduce; all lanes in a warp hold the sum.
    var warp_result = _warp_sum_f32(val)

    comptime if n_warps == 1:
        return warp_result

    var tid: Int = thread_idx.x
    var lane: Int = tid & 31
    var warp: Int = tid >> 5

    # Step 2: lane 0 of each warp writes its warp's sum to smem.
    var smem = stack_allocation[
        n_warps, DType.float32, address_space=AddressSpace.SHARED
    ]()
    if lane == 0:
        smem[warp] = warp_result

    barrier()

    # Step 3: first warp loads the n_warps partials and reduces them.
    var block_val: Float32 = 0
    if warp == 0:
        if lane < n_warps:
            block_val = smem[lane]
        block_val = _warp_sum_f32(block_val)

    return block_val


@always_inline
def _block_sum_f32_vec[
    block_size: Int, n: Int
](vals: SIMD[DType.float32, n]) -> SIMD[DType.float32, n]:
    """Block-level fp32 sum of `n` independent values, **one barrier** total.

    Each thread holds a SIMD[fp32, n] of n independent values; the result
    (held only by lane 0 of warp 0, i.e. tid 0) is the block-wide sum of
    each lane. This replaces `n` independent calls to `_block_sum_f32`,
    each of which would issue its own barrier — we instead pack the n
    per-warp partials into one smem layout and do a single barrier
    before the cross-warp reduce. Saves `n-1` barriers per kernel.
    """
    comptime assert block_size >= 32 and block_size % 32 == 0, (
        "block_size must be a multiple of warp size (32)"
    )
    comptime n_warps: Int = block_size // 32

    # Step 1: per-warp butterfly reduce, every lane.
    var warp_result = SIMD[DType.float32, n](0)

    comptime for j in range(n):
        warp_result[j] = _warp_sum_f32(vals[j])

    comptime if n_warps == 1:
        return warp_result

    var tid: Int = thread_idx.x
    var lane: Int = tid & 31
    var warp: Int = tid >> 5

    # Step 2: lane 0 of each warp writes its `n` warp-sums to smem
    # at layout smem[warp * n + j]. n_warps * n total fp32 slots.
    var smem = stack_allocation[
        n_warps * n, DType.float32, address_space=AddressSpace.SHARED
    ]()
    if lane == 0:

        comptime for j in range(n):
            smem[warp * n + j] = warp_result[j]

    barrier()

    # Step 3: warp 0 reads the n_warps partials per lane j and reduces.
    var block_vals = SIMD[DType.float32, n](0)
    if warp == 0:

        comptime for j in range(n):
            var v: Float32 = 0
            if lane < n_warps:
                v = smem[lane * n + j]
            block_vals[j] = _warp_sum_f32(v)

    return block_vals


def bwd_full_kernel[
    dtype: DType,
    n_elts: Int,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    DoutLayoutType: TensorLayout,
    DxLayoutType: TensorLayout,
](
    seqlen: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout: TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dx: TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dinitial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_batch_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
    dinitial_states_batch_stride: Int,
    dinitial_states_c_stride: Int,
    dinitial_states_l_stride: Int,
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank >= 3
    and TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin].flat_rank >= 3
    and TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin].flat_rank >= 3
):
    """Fused backward: dx + dweight + dbias, one block per (B, D).

    Mirrors upstream's `causal_conv1d_bwd_kernel`:
    - Grid (dim, batch). One block per channel-batch pair so the global
      atomic_add contention is `B` per channel, not `B * num_chunks_l`.
    - Walks chunks in REVERSE (last-to-first) so each chunk's dout serves
      as the *next* chunk's dx halo via smem, and silu' is recomputed
      exactly once per element. The forward-direction version (which we
      had before) had to recompute silu' for the halo positions of the
      next chunk too.
    - x is loaded ONCE per element. tidx>0's "previous kNElts" halo comes
      from the warp neighbour via smem; tidx==0 loads the previous chunk's
      last kNElts directly from global (cheap: one thread, one load/chunk).

    Each iteration:
      [P1] load x_curr, dout_curr from global (per-thread kNElts).
      [P2] x halo from prev-chunk via smem_x[tidx-1] (or global for tidx==0).
      [P3] compute pre = bias + W*x_window;  silu' = sigmoid*(1+pre*(1-sigmoid));
           dpre = dout_curr * silu' (this is the silu/bias-aware "dout" for chains).
      [P4] dout halo from next-chunk via smem_dout — written by previous iter.
           Three-barrier dance (mirrors upstream): tidx>0 writes first, all
           threads read smem_dout[(tidx+1)%kNThreads], thread 0 writes last.
           This lets smem_dout[0] hold the next chunk's thread-0 dpre at
           read-time AND get overwritten with the current chunk's thread-0
           dpre for the *next* iter's read.
      [P5] dx[i] = sum_w weight[w] * dpre_with_halo[i + W-1-w]; store to global.
      [P6] dweight[w] += sum_i x_curr[i] * dpre_with_halo[i + W-1-w];
           dbias    += sum_i dpre[i].

    After the chunk loop: block-reduce dweight,dbias and atomic_add to global.
    """
    comptime accum_t = DType.float32
    # Per-thread element count, set by the dispatcher: 8 for fp16/bf16
    # when seqlen is a multiple of 1024, else 4 (to keep all 128 threads
    # busy on small seqlens). For fp32 it's always 4 (16-byte LDG cap).
    # Forward uses a fixed 4 because its grid scales with seqlen and a
    # wider per-thread tile halves parallelism; here the grid is (D, B)
    # and the chunk loop walks the seqlen, so a wider tile only
    # shortens the loop.
    comptime kNElts: Int = n_elts
    comptime kChunkSize: Int = kNThreads * kNElts

    var tidx: Int = thread_idx.x
    var channel_id: Int = block_idx.x
    var batch_id: Int = block_idx.y

    # Load weights into per-block fp32 registers.
    var weights = SIMD[accum_t, width](0)

    comptime for k in range(width):
        weights[k] = weight[channel_id, k].cast[accum_t]()

    var cur_bias: Scalar[accum_t] = 0

    comptime if has_bias:
        cur_bias = bias_ptr[channel_id].cast[accum_t]()

    # smem_x: each thread's kNElts x values for the next thread's halo.
    # smem_dout: each thread's kNElts dpre values (post-silu') for the
    # *previous chunk's* dx halo (we walk in reverse).
    var smem_x = stack_allocation[
        kChunkSize, dtype, address_space=AddressSpace.SHARED
    ]()
    var smem_dout = stack_allocation[
        kChunkSize, accum_t, address_space=AddressSpace.SHARED
    ]()

    var seq_idx_base: Int = batch_id * seq_idx_b_stride
    var init_base: Int = (
        batch_id * initial_states_batch_stride
        + channel_id * initial_states_c_stride
    )
    var dinit_base: Int = (
        batch_id * dinitial_states_batch_stride
        + channel_id * dinitial_states_c_stride
    )

    # Per-thread accumulators (persist across chunks).
    var local_dweight = SIMD[accum_t, width](0)
    var local_dbias: Scalar[accum_t] = 0

    var n_chunks: Int = ceildiv(seqlen, kChunkSize)

    # Initialise smem_dout slot 0 to zero. On the first iteration (last
    # chunk) thread kNThreads-1 reads smem_dout[0] for its halo; for the
    # last chunk that halo is past the seqlen end and must be zero.
    if tidx == 0:
        smem_dout.store[alignment=16](SIMD[accum_t, kNElts](0))

    barrier()

    # Reverse iteration: chunk n_chunks-1 down to 0.
    for chunk_rev in range(n_chunks):
        var chunk: Int = n_chunks - 1 - chunk_rev
        var chunk_start: Int = chunk * kChunkSize
        var seq_start: Int = chunk_start + tidx * kNElts

        # ---- [P0] seq_idx window for this thread ----
        # Each thread needs seq_idx at positions
        #   [seq_start - (W-1) ..  seq_start + kNElts + (W-1) - 1]
        # = its own kNElts + a `(W-1)` halo on each side. The halos are
        # what Phase 3's silu' (left) and Phase 5/6's dx + dweight
        # (right) need to gate the conv with `seq_idx[s] == seq_idx[t]`.
        # Out-of-range positions get -1 so the gate naturally fails.
        # seq_idx is small (Int32, B*L), no smem dance needed.
        comptime kSeqIdxWindow: Int = 2 * (width - 1) + kNElts
        var seq_idx_window = InlineArray[Int32, kSeqIdxWindow](
            uninitialized=True
        )

        comptime if has_seq_idx:

            comptime for j in range(kSeqIdxWindow):
                var t_j = seq_start + j - (width - 1)
                if 0 <= t_j and t_j < seqlen:
                    seq_idx_window[j] = seq_idx_ptr[
                        seq_idx_base + t_j * seq_idx_l_stride
                    ]
                else:
                    seq_idx_window[j] = -1

        # ---- [P1] load x_curr and dout_curr ----
        # `alignment=16` promises a 16-byte aligned base, letting the
        # compiler emit the widest single-instruction global load: LDG.E.U64
        # for fp16/bf16 (kNElts=4 × 2 B = 8 B/thread) or LDG.E.128 for fp32
        # (kNElts=4 × 4 B = 16 B/thread). Without it the default alignment
        # is align_of[dtype] = 2 or 4, which blocks the merge — even when
        # widths line up — and the compiler falls back to scalar loads.
        # Standard PyTorch row-major tensors satisfy the 16-byte promise:
        # base addresses are large multiples of channel/batch strides
        # which are 16-aligned, and seq_start lands on a kNElts boundary.
        # `aligned_seq` is comptime: when True (the typical case where
        # seqlen % kChunkSize == 0) the per-element bounds-checked
        # fallback isn't compiled at all, halving the kernel size.
        var x_curr = SIMD[accum_t, kNElts](0)
        var dout_curr = SIMD[accum_t, kNElts](0)

        comptime if contig_inner and aligned_seq:
            x_curr = x.load[width=kNElts, alignment=16](
                Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
            ).cast[accum_t]()
            dout_curr = dout.load[width=kNElts, alignment=16](
                Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
            ).cast[accum_t]()
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                x_curr = x.load[width=kNElts, alignment=16](
                    Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
                ).cast[accum_t]()
                dout_curr = dout.load[width=kNElts, alignment=16](
                    Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
                ).cast[accum_t]()
            else:

                comptime for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        x_curr[i] = x[batch_id, channel_id, t].cast[accum_t]()
                        dout_curr[i] = dout[batch_id, channel_id, t].cast[
                            accum_t
                        ]()
        else:

            comptime for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    x_curr[i] = x[batch_id, channel_id, t].cast[accum_t]()
                    dout_curr[i] = dout[batch_id, channel_id, t].cast[
                        accum_t
                    ]()

        # ---- [P2] x halo from previous chunk ----
        # tidx==0: load previous kNElts of x from global (chunk>0 only).
        # tidx>0:  read previous thread's x_curr from smem.
        var x_prev = SIMD[accum_t, kNElts](0)
        if tidx == 0 and chunk > 0:

            comptime for i in range(kNElts):
                var t = chunk_start - kNElts + i
                if t >= 0:
                    x_prev[i] = x[batch_id, channel_id, t].cast[accum_t]()

        # When `has_initial_states`, chunk 0 / tidx 0 reads the trailing
        # `W-1` x_prev positions (which correspond to t in [-(W-1), 0))
        # from initial_states instead of leaving them at zero. This
        # makes Phase 3's silu' recomputation see the same `pre[t]` the
        # forward did for the first (W-1) output positions.
        comptime if has_initial_states:
            if tidx == 0 and chunk == 0:

                comptime for i in range(width - 1):
                    x_prev[kNElts - (width - 1) + i] = initial_states_ptr[
                        init_base + i * initial_states_l_stride
                    ].cast[accum_t]()

        # Publish x_curr to smem so the next thread can pick it up as halo.
        # Single vector store: tidx*kNElts*sizeof(dtype) is 16-byte aligned
        # (kNElts*sizeof(dtype) = 16 for both fp16 kNElts=8 and fp32 kNElts=4),
        # so the compiler emits one st.shared.v4.b32 rather than kNElts scalar
        # stores. Same for the load on the read-back side.
        (smem_x + tidx * kNElts).store[alignment=16](x_curr.cast[dtype]())

        barrier()  # smem_x writes visible

        if tidx > 0:
            x_prev = (
                (smem_x + (tidx - 1) * kNElts)
                .load[width=kNElts, alignment=16]()
                .cast[accum_t]()
            )

        # ---- [P3] derive dpre from dout (and silu' if activation was silu) ----
        # When apply_silu, dpre = dout * silu'(pre); otherwise dpre = dout
        # (the bias-only forward has identity gradient w.r.t. pre).
        # silu'(pre) requires recomputing pre = bias + sum_k weights[k] *
        # x_window[k]; x_window comes from [x_prev || x_curr] using the same
        # offset arithmetic as the forward.
        # With has_seq_idx, the same `seq_idx[src_t] == seq_idx[t]` gate
        # the forward applied is reapplied here on each x lookup, and
        # `seq_idx[t] < 0` (padding token) forces dpre[i] = 0 since the
        # forward forced out=0 there.
        var dpre = SIMD[accum_t, kNElts](0)

        comptime if apply_silu:

            comptime for i in range(kNElts):
                # Cur output position's seq_idx (window index W-1 + i).
                var cur_id: Int32 = 0

                comptime if has_seq_idx:
                    cur_id = seq_idx_window[(width - 1) + i]

                comptime if aligned_seq:
                    var pre: Scalar[accum_t] = cur_bias

                    comptime for k in range(width):
                        comptime offset_w: Int = k - (width - 1)
                        var include: Bool = True

                        comptime if has_seq_idx:
                            # x lookup at window index i + k.
                            include = seq_idx_window[i + k] == cur_id
                        if include:

                            comptime if i + offset_w >= 0:
                                pre += x_curr[i + offset_w] * weights[k]
                            else:
                                pre += (
                                    x_prev[kNElts + i + offset_w] * weights[k]
                                )

                    var sig: Scalar[accum_t] = 1.0 / (1.0 + exp(-pre))
                    var silu_grad: Scalar[accum_t] = sig * (
                        1.0 + pre * (1.0 - sig)
                    )
                    dpre[i] = dout_curr[i] * silu_grad

                    comptime if has_seq_idx:
                        if cur_id < 0:
                            dpre[i] = 0
                else:
                    if seq_start + i < seqlen:
                        var pre: Scalar[accum_t] = cur_bias

                        comptime for k in range(width):
                            comptime offset_w: Int = k - (width - 1)
                            var include: Bool = True

                            comptime if has_seq_idx:
                                include = seq_idx_window[i + k] == cur_id
                            if include:

                                comptime if i + offset_w >= 0:
                                    pre += x_curr[i + offset_w] * weights[k]
                                else:
                                    pre += (
                                        x_prev[kNElts + i + offset_w]
                                        * weights[k]
                                    )

                        var sig: Scalar[accum_t] = 1.0 / (1.0 + exp(-pre))
                        var silu_grad: Scalar[accum_t] = sig * (
                            1.0 + pre * (1.0 - sig)
                        )
                        dpre[i] = dout_curr[i] * silu_grad

                        comptime if has_seq_idx:
                            if cur_id < 0:
                                dpre[i] = 0
        else:
            # No silu — dpre is just dout. `dout_curr` already has out-of-
            # bounds positions zeroed (loaded that way in [P1]). With
            # seq_idx, also zero padding-token positions.
            dpre = dout_curr

            comptime if has_seq_idx:

                comptime for i in range(kNElts):
                    if seq_idx_window[(width - 1) + i] < 0:
                        dpre[i] = 0

        # dbias += sum(dpre) — only when there's a bias to accumulate into.
        # dpre is already 0 at padding positions, so no extra gating
        # needed.
        comptime if has_bias:
            local_dbias += dpre.reduce_add()

        # ---- [P4] dout halo from next chunk (already in smem_dout) ----
        # Three-barrier dance, mirroring upstream:
        #   1. tidx>0 writes its dpre to smem_dout[tidx*kNElts..]
        #      (slot 0 still holds NEXT chunk's thread-0 dpre, which thread
        #       kNThreads-1 will read for its halo).
        #   2. all threads read smem_dout[((tidx+1) % kNThreads)*kNElts..]
        #   3. thread 0 writes its dpre to slot 0 for the next iteration.
        barrier()  # all reads of smem_x done; safe to reuse smem_dout

        # Vector stores/loads on smem_dout: same story as smem_x — one
        # v4.b32 per 16 bytes. For fp16/bf16 with kNElts=8 the dpre is
        # 8 fp32 = 32 bytes, so the compiler emits two v4.b32 (still
        # better than 8 scalars). For fp32 with kNElts=4 it's a single
        # v4.b32.
        if tidx > 0:
            (smem_dout + tidx * kNElts).store[alignment=16](dpre)

        barrier()  # tidx>0 writes visible; slot 0 still holds NEXT chunk's data

        var halo_thread = tidx + 1 if tidx < kNThreads - 1 else 0
        var dout_halo = (smem_dout + halo_thread * kNElts).load[
            width=kNElts, alignment=16
        ]()

        barrier()  # all halo reads done; thread 0 may now stomp slot 0

        if tidx == 0:
            smem_dout.store[alignment=16](dpre)

        # ---- [P5] dx = anti-causal conv on dpre || dout_halo ----
        # dx[i] = sum_w weights[w] * combined[i + (W-1-w)]
        # combined = [dpre (kNElts) || dout_halo (kNElts)]
        # With has_seq_idx: gate each w-term on
        # `seq_idx[seq_start+i] == seq_idx[seq_start+i+(W-1)-w]` —
        # x[seq_start+i] only contributed to position
        # seq_start+i+(W-1)-w in the forward when their ids matched.
        var dx_vals = SIMD[accum_t, kNElts](0)

        comptime for i in range(kNElts):
            var cur_id_dx: Int32 = 0

            comptime if has_seq_idx:
                cur_id_dx = seq_idx_window[(width - 1) + i]

            comptime for k in range(width):
                comptime halo_idx: Int = i + (width - 1) - k
                var include_dx: Bool = True

                comptime if has_seq_idx:
                    include_dx = (
                        seq_idx_window[i + 2 * (width - 1) - k] == cur_id_dx
                    )
                if include_dx:

                    comptime if halo_idx < kNElts:
                        dx_vals[i] += weights[k] * dpre[halo_idx]
                    else:
                        dx_vals[i] += weights[k] * dout_halo[halo_idx - kNElts]

        comptime if contig_inner and aligned_seq:
            dx.store[alignment=16](
                Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start)),
                dx_vals.cast[dtype](),
            )
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                dx.store[alignment=16](
                    Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start)),
                    dx_vals.cast[dtype](),
                )
            else:

                comptime for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        dx[batch_id, channel_id, t] = dx_vals[i].cast[dtype]()
        else:

            comptime for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    dx[batch_id, channel_id, t] = dx_vals[i].cast[dtype]()

        # ---- [P6] dweight[w] += sum_i x_curr[i] * combined[i + (W-1-w)] ----
        # With has_seq_idx: gate each i-term on
        # `seq_idx[seq_start+i] == seq_idx[seq_start+i+(W-1)-k]`. The
        # output position whose dpre we use is at i + (W-1) - k; the x
        # position is i. Same index pattern as Phase 5.
        comptime for k in range(width):
            var acc: Scalar[accum_t] = 0

            comptime for i in range(kNElts):
                comptime halo_idx_dw: Int = i + (width - 1) - k
                var include_dw: Bool = True

                comptime if has_seq_idx:
                    include_dw = (
                        seq_idx_window[i + 2 * (width - 1) - k]
                        == seq_idx_window[(width - 1) + i]
                    )
                if include_dw:

                    comptime if halo_idx_dw < kNElts:
                        acc += x_curr[i] * dpre[halo_idx_dw]
                    else:
                        acc += x_curr[i] * dout_halo[halo_idx_dw - kNElts]
            local_dweight[k] += acc

        # ---- [P7] initial_states-only contributions (chunk 0, tidx 0) ----
        # When the forward read initial_states for `t in [0, W-1)`, two
        # gradient pieces appear:
        #   - dinitial_states[i] = sum_{k<=i} weight[k] * dpre[i - k]
        #     for i in [0, W-1). All needed dpre[0..W-2] are in this
        #     thread's `dpre` register.
        #   - dweight[k] picks up a "boundary" term from the part of the
        #     forward conv that hit initial_states:
        #       dweight[k] += sum_{t<W-1-k} dpre[t] * initial_states[t+k].
        # dinitial_states is written here directly; the dweight terms
        # accumulate into local_dweight and join the block reduce below.
        comptime if has_initial_states:
            if chunk == 0 and tidx == 0:

                comptime for i in range(width - 1):
                    var dinit_v: Scalar[accum_t] = 0

                    comptime for k in range(width):

                        comptime if i - k >= 0:
                            dinit_v += weights[k] * dpre[i - k]
                    dinitial_states_ptr[
                        dinit_base + i * dinitial_states_l_stride
                    ] = dinit_v.cast[dtype]()

                comptime for k in range(width):

                    comptime for t in range(width - 1 - k):
                        var is_v = initial_states_ptr[
                            init_base + (t + k) * initial_states_l_stride
                        ].cast[accum_t]()
                        local_dweight[k] += dpre[t] * is_v

    # === Phase 4: block-reduce dweight, dbias and atomic-add to global ===
    # `scope="device"` + `ordering=MONOTONIC` is the same memory model as
    # CUDA's `atomicAdd(...)` — relaxed, GPU-scope. The default
    # `Atomic.fetch_add(...)` lowers to `ATOMG.E.ADD.F32.STRONG.SYS`, a
    # *system-scope, sequentially-consistent* atomic that drains L2 and
    # synchronises with the CPU. That's enormous: on this kernel it added
    # ~750 ns of fixed per-block overhead and dominated total runtime
    # (3.3 ms vs 1.2 ms with no atomics). Profiler diff:
    #   default ordering : (4,4096,2048) bwd kernel = 14400 us
    #   monotonic+device : (4,4096,2048) bwd kernel =  3700 us
    # Caller does its own torch.cuda.synchronize(); a release/acquire
    # fence here is unnecessary.
    #
    # We use the vectorised block-sum to fuse the (width) dweight
    # reductions plus optional dbias reduction into a single barrier-
    # sharing block-reduce. Naively, `width` independent `_block_sum_f32`
    # calls would each issue its own `barrier()` (the cross-warp sync) —
    # `width = 2..4` => 2..4 extra block-wide stalls per (B,D) block.
    # Packing them lets the smem write/read pair amortise one sync across
    # all reductions.
    comptime if has_bias:
        comptime nred: Int = width + 1
        var packed = SIMD[accum_t, nred](0)

        comptime for k in range(width):
            packed[k] = local_dweight[k]
        packed[width] = local_dbias
        var block_red = _block_sum_f32_vec[block_size=kNThreads, n=nred](
            packed
        )
        if tidx == 0:

            comptime for k in range(width):
                _ = Atomic[DType.float32, scope="device"].fetch_add[
                    ordering=Ordering.RELAXED
                ](
                    dweight_acc_ptr + channel_id * width + k,
                    block_red[k],
                )
            _ = Atomic[DType.float32, scope="device"].fetch_add[
                ordering=Ordering.RELAXED
            ](dbias_acc_ptr + channel_id, block_red[width])
    else:
        var block_red = _block_sum_f32_vec[
            block_size=kNThreads, n=width
        ](local_dweight)
        if tidx == 0:

            comptime for k in range(width):
                _ = Atomic[DType.float32, scope="device"].fetch_add[
                    ordering=Ordering.RELAXED
                ](
                    dweight_acc_ptr + channel_id * width + k,
                    block_red[k],
                )
