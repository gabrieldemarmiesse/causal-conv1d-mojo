"""GPU fused backward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_bwd.cu`. The launcher lives in
`launch.mojo`; this file holds the kernel + the warp / block
reduction helpers it depends on.
"""

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_xor
from std.bit import log2_floor
from std.math import ceildiv, exp, recip
from std.memory import stack_allocation
from std.atomic import Atomic, Ordering
from std.sys import llvm_intrinsic
from std.sys.info import is_nvidia_gpu
from std.utils.index import StaticTuple
from layout import TileTensor, TensorLayout, Idx, Coord

from common import kNThreads


# Atomic scope for the gradient reduce step. CUDA's `atomicAdd(...)`
# is GPU-scope, relaxed — the LLVM syncscope name for that is "device"
# on NVPTX and "agent" on AMDGPU. Picking the wrong one fails to lower
# with "Unsupported non-inclusive atomic synchronization scope" on AMD.
alias kAtomicScope: StaticString = (
    "device" if is_nvidia_gpu() else "agent"
)


@always_inline
def _rcp_approx_f32(x: Float32) -> Float32:
    """Single-instruction `rcp.approx.ftz.f32`, fp32 reciprocal.

    The default `1.0 / x` lowers to `div.rn.f32` (IEEE-accurate, several
    cycles latency on H100). The bwd's silu' computation chains a
    reciprocal onto `1.0 + exp(-pre)` per element — kNElts of them per
    chunk-iter per thread. Swapping the IEEE-accurate divide for the
    `rcp.approx.ftz` PTX intrinsic gets us a single-cycle reciprocal
    at fp16-ish precision (more than enough for the silu sigmoid
    backward, whose result is then multiplied by an fp16/bf16 dout).
    On the fwd kernel this was the largest single perf win (1.25x →
    1.00x ratio). The intrinsic is nvvm-only; on AMD targets we fall
    back to `recip()` (which lowers to `v_rcp_f32` on amdgcn — already
    a fast approximate reciprocal).
    """
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.rcp.approx.ftz.f", Float32](x)
    else:
        return recip(x)


@always_inline
def _shfl_xor_f32(val: Float32, offset: UInt32) -> Float32:
    """One inlined `shfl.sync.bfly.b32`, fp32.

    Why: the stdlib chain `block.sum -> warp.sum -> shuffle_xor -> _shuffle`
    is `@always_inline` at every level on the Mojo side, but on sm_89 ptxas
    still outlines each shfl into a `__cuda_sm70_shflsync_bfly` helper. SASS
    diff vs. upstream's bwd showed 45 `CALL.REL.NOINC` to that helper. By
    issuing the LLVM intrinsic from a single small leaf and force-inlining
    every wrapper around it, ptxas stops outlining and generates the bare
    SHFL.BFLY (matching upstream). The nvvm intrinsic is NVIDIA-only;
    on AMD we use the portable `gpu.warp.shuffle_xor` (lowers to
    `ds_bpermute_b32` on amdgcn).
    """
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.f32", Float32](
            Int32(-1), val, offset, Int32(31)
        )
    else:
        return shuffle_xor(val, offset)


@always_inline
def _warp_sum_f32(val: Float32) -> Float32:
    """log2(WARP_SIZE)-step butterfly warp reduction, fp32. All lanes
    hold the full warp's sum. On NVIDIA WARP_SIZE=32 ⇒ 5 steps; on AMD
    CDNA WARP_SIZE=64 ⇒ 6 steps (the extra step is the cross-half-warp
    xor=32). The old hand-rolled 5 fixed steps left lanes 32..63 of an
    AMD wavefront with a half-warp partial — correct only because we
    then re-bucketed via `lane = tid & 31, warp = tid >> 5`, but that
    burns smem and barriers we don't need with full-warp shuffles."""
    var v = val

    comptime for i in reversed(range(log2_floor(WARP_SIZE))):
        v += _shfl_xor_f32(v, UInt32(1 << i))
    return v


@always_inline
def _warp_sum_f32_vec[n: Int](
    vals: SIMD[DType.float32, n]
) -> SIMD[DType.float32, n]:
    """Vectorised butterfly warp reduction over `n` independent fp32
    values. All lanes return the full warp's sum (component-wise).

    On AMD `_shfl_xor_f32` lowers to `ds_bpermute_b32`, which the
    compiler bookends with `s_waitcnt lgkmcnt(0)` per call. Reducing
    each of the `n` values one-at-a-time would therefore issue
    `n * log2(WARP_SIZE)` serial shuffle + wait pairs. By interleaving
    the `n` reductions at each butterfly step, the `n` independent
    shuffles in a step can be issued back-to-back (one LDS instruction
    queue, no inter-shuffle dep) so the compiler only needs *one*
    waitcnt before the `n` adds. Cuts the final reduce's wait count
    by ~`(n - 1) * log2(WARP_SIZE)` — measurable when n is the full
    `width + 1` packed reduction (5 for width=4, with-bias)."""
    var v = vals

    comptime for i in reversed(range(log2_floor(WARP_SIZE))):
        comptime offset = UInt32(1 << i)
        var shuffled = SIMD[DType.float32, n](0)

        comptime for j in range(n):
            shuffled[j] = _shfl_xor_f32(v[j], offset)
        v += shuffled
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
    comptime assert block_size >= WARP_SIZE and block_size % WARP_SIZE == 0, (
        "block_size must be a multiple of WARP_SIZE"
    )
    comptime n_warps: Int = block_size // WARP_SIZE

    # Step 1: per-warp butterfly reduce; all lanes in a warp hold the sum.
    var warp_result = _warp_sum_f32(val)

    comptime if n_warps == 1:
        return warp_result

    var tid: Int = thread_idx.x
    var lane: Int = tid & (WARP_SIZE - 1)
    var warp: Int = tid // WARP_SIZE

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
    comptime assert block_size >= WARP_SIZE and block_size % WARP_SIZE == 0, (
        "block_size must be a multiple of WARP_SIZE"
    )
    comptime n_warps: Int = block_size // WARP_SIZE

    # Step 1: per-warp butterfly reduce over all n values in lock step
    # (one shuffle-stage per butterfly level, n shuffles per stage).
    var warp_result = _warp_sum_f32_vec[n=n](vals)

    comptime if n_warps == 1:
        return warp_result

    var tid: Int = thread_idx.x
    var lane: Int = tid & (WARP_SIZE - 1)
    var warp: Int = tid // WARP_SIZE

    # Step 2: lane 0 of each warp writes its `n` warp-sums to smem
    # at layout smem[warp * n + j]. n_warps * n total fp32 slots.
    var smem = stack_allocation[
        n_warps * n, DType.float32, address_space=AddressSpace.SHARED
    ]()
    if lane == 0:

        comptime for j in range(n):
            smem[warp * n + j] = warp_result[j]

    barrier()

    # Step 3: warp 0 reads the n_warps partials per lane j and reduces
    # them all in lock-step via the vec warp reduce.
    var block_vals = SIMD[DType.float32, n](0)
    if warp == 0:
        var v = SIMD[DType.float32, n](0)
        if lane < n_warps:

            comptime for j in range(n):
                v[j] = smem[lane * n + j]
        block_vals = _warp_sum_f32_vec[n=n](v)

    return block_vals


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
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
    SLayoutType: TensorLayout,
    ILayoutType: TensorLayout,
    DILayoutType: TensorLayout,
](
    seqlen: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout: TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin],
    seq_idx: TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin],
    initial_states: TileTensor[dtype, ILayoutType, ImmutAnyOrigin],
    dx: TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dinitial_states: TileTensor[
        mut=True, dtype, DILayoutType, MutAnyOrigin
    ],
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin].flat_rank == 3
    and TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, ILayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, DILayoutType, MutAnyOrigin].flat_rank == 3
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
    # Grid layout matches upstream: blockIdx.x = batch, blockIdx.y =
    # dim. See launch.mojo for the rationale.
    var batch_id: Int = block_idx.x
    var channel_id: Int = block_idx.y

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

    # Per-thread accumulators (persist across chunks).
    var local_dweight = SIMD[accum_t, width](0)
    var local_dbias: Scalar[accum_t] = 0

    var n_chunks: Int = ceildiv(seqlen, kChunkSize)

    # Initialise smem_dout slot 0 to zero. On the first iteration
    # (last chunk in time) thread kNThreads-1 reads smem_dout[0] for
    # its halo; for the last chunk that halo is past the seqlen end
    # and must be zero. No barrier needed here — the first chunk's
    # smem_x publish barrier (inside the loop) covers this init write
    # before any thread reads slot 0.
    if tidx == 0:
        smem_dout.store[alignment=16](SIMD[accum_t, kNElts](0))

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
                    seq_idx_window[j] = seq_idx[batch_id, t_j]
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
            # `chunk > 0` ⇒ chunk_start ≥ kChunkSize > kNElts ⇒ all kNElts
            # elements at [chunk_start - kNElts, chunk_start) are in
            # range. `chunk_start` is a multiple of `kChunkSize` (=
            # `kNThreads * kNElts`), so `chunk_start - kNElts` is a
            # multiple of kNElts ⇒ kNElts-aligned addr ⇒ 16-byte aligned
            # for both fp16/bf16 (kNElts=8 × 2 B = 16) and fp32
            # (kNElts=4 × 4 B = 16). Skip the per-element bounds-checked
            # scalar path for `contig_inner` and issue a single LDG.E.128.
            comptime if contig_inner:
                x_prev = x.load[width=kNElts, alignment=16](
                    Coord(
                        Idx(batch_id),
                        Idx(channel_id),
                        Idx(chunk_start - kNElts),
                    )
                ).cast[accum_t]()
            else:

                comptime for i in range(kNElts):
                    var t = chunk_start - kNElts + i
                    x_prev[i] = x[batch_id, channel_id, t].cast[accum_t]()

        # When `has_initial_states`, chunk 0 / tidx 0 reads the trailing
        # `W-1` x_prev positions (which correspond to t in [-(W-1), 0))
        # from initial_states instead of leaving them at zero. This
        # makes Phase 3's silu' recomputation see the same `pre[t]` the
        # forward did for the first (W-1) output positions.
        comptime if has_initial_states:
            if tidx == 0 and chunk == 0:

                comptime for i in range(width - 1):
                    x_prev[kNElts - (width - 1) + i] = initial_states[
                        batch_id, channel_id, i
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

                    var sig: Scalar[accum_t] = _rcp_approx_f32(
                        1.0 + exp(-pre)
                    )
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

                        var sig: Scalar[accum_t] = _rcp_approx_f32(
                        1.0 + exp(-pre)
                    )
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
        # Two-barrier dance:
        #   1. tidx>0 writes its dpre to smem_dout[tidx*kNElts..]
        #      (slot 0 still holds NEXT chunk's thread-0 dpre, which thread
        #       kNThreads-1 will read for its halo).
        #   2. all threads read smem_dout[((tidx+1) % kNThreads)*kNElts..]
        #   3. thread 0 writes its dpre to slot 0 for the next iteration.
        # Note: upstream's bwd has an extra `__syncthreads()` here just
        # before the tidx>0 writes. It's redundant on our control flow
        # because (a) smem_x and smem_dout are *separate* shared buffers
        # so the smem_x reads above don't race with the smem_dout writes
        # below, and (b) every smem_dout slot tidx>0 was last *read* in
        # the *previous* chunk iter and protected by that iter's
        # `barrier()  # all halo reads done` — which any subsequent
        # barrier (including this iter's smem_x barrier above) covers
        # transitively. Dropping it saves one block-wide sync per chunk.
        #
        # Vector stores/loads on smem_dout: same story as smem_x — one
        # v4.b32 per 16 bytes. For fp16/bf16 with kNElts=8 the dpre is
        # 8 fp32 = 32 bytes, so the compiler emits two v4.b32 (still
        # better than 8 scalars). For fp32 with kNElts=4 it's a single
        # v4.b32.
        if tidx > 0:
            (smem_dout + tidx * kNElts).store[alignment=16](dpre)

        # ---- [P5a] dx terms that don't need the halo ----
        # The anti-causal conv `dx[i] = sum_w weights[w] * combined[i +
        # (W-1) - w]` reads from `combined = [dpre || dout_halo]`. Only
        # terms with `halo_idx = i + (W-1) - w >= kNElts` need the halo;
        # the rest are pure dpre-times-weights and can be computed
        # *before* the smem_dout barrier below. That gives the compiler
        # ~width*(width-1)/2 FMAs to interleave with the LDS barrier
        # wait, hiding part of the sync latency.
        var dx_vals = SIMD[accum_t, kNElts](0)

        comptime for i in range(kNElts):
            var cur_id_dx: Int32 = 0

            comptime if has_seq_idx:
                cur_id_dx = seq_idx_window[(width - 1) + i]

            comptime for k in range(width):
                comptime halo_idx: Int = i + (width - 1) - k

                comptime if halo_idx < kNElts:
                    var include_dx: Bool = True

                    comptime if has_seq_idx:
                        include_dx = (
                            seq_idx_window[i + 2 * (width - 1) - k]
                            == cur_id_dx
                        )
                    if include_dx:
                        dx_vals[i] += weights[k] * dpre[halo_idx]

        barrier()  # tidx>0 writes visible; slot 0 still holds NEXT chunk's data

        var halo_thread = tidx + 1 if tidx < kNThreads - 1 else 0
        var dout_halo = (smem_dout + halo_thread * kNElts).load[
            width=kNElts, alignment=16
        ]()

        # The slot-0 stomp + its barrier only matter for the *next* iter
        # (chunk-1 in reverse). On the last iter (chunk==0) nothing reads
        # slot 0 again — drop both. Saves one block-wide sync per kernel
        # call. The remaining iters still need the barrier so tidx=
        # kNThreads-1's halo read of slot 0 (NEXT chunk's data) completes
        # before tidx==0 overwrites slot 0 with the *current* chunk's
        # dpre.
        if chunk > 0:
            barrier()  # all halo reads done; thread 0 may now stomp slot 0
            if tidx == 0:
                smem_dout.store[alignment=16](dpre)

        # ---- [P5b] dx halo terms ----
        # dx[i] += sum_{w: halo_idx>=kNElts} weights[w] *
        #         dout_halo[halo_idx - kNElts]

        comptime for i in range(kNElts):
            var cur_id_dx: Int32 = 0

            comptime if has_seq_idx:
                cur_id_dx = seq_idx_window[(width - 1) + i]

            comptime for k in range(width):
                comptime halo_idx: Int = i + (width - 1) - k

                comptime if halo_idx >= kNElts:
                    var include_dx: Bool = True

                    comptime if has_seq_idx:
                        include_dx = (
                            seq_idx_window[i + 2 * (width - 1) - k]
                            == cur_id_dx
                        )
                    if include_dx:
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
                    dinitial_states[batch_id, channel_id, i] = dinit_v.cast[
                        dtype
                    ]()

                comptime for k in range(width):

                    comptime for t in range(width - 1 - k):
                        var is_v = initial_states[
                            batch_id, channel_id, t + k
                        ].cast[accum_t]()
                        local_dweight[k] += dpre[t] * is_v

    # === Phase 4: block-reduce dweight, dbias and atomic-add to global ===
    # `scope="agent"` + `ordering=MONOTONIC` is the same memory model as
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
                _ = Atomic[DType.float32, scope=kAtomicScope].fetch_add[
                    ordering=Ordering.RELAXED
                ](
                    dweight_acc_ptr + channel_id * width + k,
                    block_red[k],
                )
            _ = Atomic[DType.float32, scope=kAtomicScope].fetch_add[
                ordering=Ordering.RELAXED
            ](dbias_acc_ptr + channel_id, block_red[width])
    else:
        var block_red = _block_sum_f32_vec[
            block_size=kNThreads, n=width
        ](local_dweight)
        if tidx == 0:

            comptime for k in range(width):
                _ = Atomic[DType.float32, scope=kAtomicScope].fetch_add[
                    ordering=Ordering.RELAXED
                ](
                    dweight_acc_ptr + channel_id * width + k,
                    block_red[k],
                )
