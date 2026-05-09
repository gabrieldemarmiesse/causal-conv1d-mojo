"""GPU fused backward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_bwd.cu`. The launcher lives in
`causal_conv1d_native.mojo`; this file holds the kernel + the warp /
block reduction helpers it depends on.
"""

from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
    barrier,
)
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp
from std.memory import stack_allocation
from std.os.atomic import Atomic, Consistency
from std.sys import llvm_intrinsic

from causal_conv1d_common import kNEltsBwd, kNThreads


@always_inline
fn _shfl_xor_f32(val: Float32, offset: UInt32) -> Float32:
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
fn _warp_sum_f32(val: Float32) -> Float32:
    """5-step butterfly warp reduction, fp32. All lanes hold the warp's sum."""
    var v = val
    v += _shfl_xor_f32(v, UInt32(16))
    v += _shfl_xor_f32(v, UInt32(8))
    v += _shfl_xor_f32(v, UInt32(4))
    v += _shfl_xor_f32(v, UInt32(2))
    v += _shfl_xor_f32(v, UInt32(1))
    return v


@always_inline
fn _block_sum_f32[block_size: Int](val: Float32) -> Float32:
    """Block-level fp32 sum specialised for our backward kernel.

    Equivalent to `gpu_block.sum[block_size=block_size, broadcast=False]` for
    fp32 scalars, but everything from the LLVM shuffle intrinsic up is in
    one translation unit and `@always_inline`. The cross-warp reduction is
    gated on `warp == 0` (only one warp's worth of shfl work, like upstream).

    Only thread 0 holds the meaningful result (broadcast=False).
    """
    constrained[
        block_size >= 32 and block_size % 32 == 0,
        "block_size must be a multiple of warp size (32)",
    ]()
    alias n_warps: Int = block_size // 32

    # Step 1: per-warp butterfly reduce; all lanes in a warp hold the sum.
    var warp_result = _warp_sum_f32(val)

    @parameter
    if n_warps == 1:
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


fn bwd_full_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
](
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dx_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    dout_batch_stride: Int,
    dout_c_stride: Int,
    dout_l_stride: Int,
    dx_batch_stride: Int,
    dx_c_stride: Int,
    dx_l_stride: Int,
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
    alias accum_t = DType.float32
    # Local alias: bwd uses kNEltsBwd. Forward uses kNElts=4 because its
    # grid scales with seqlen and we don't want to halve parallelism.
    alias kNElts: Int = kNEltsBwd
    alias kChunkSize: Int = kNThreads * kNElts

    var tidx: Int = thread_idx.x
    var channel_id: Int = block_idx.x
    var batch_id: Int = block_idx.y

    # Load weights into per-block fp32 registers.
    var weights = SIMD[accum_t, width](0)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):

        @parameter
        if contig_inner:
            weights[k] = weight_ptr[weight_base + k].cast[accum_t]()
        else:
            weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[
                accum_t
            ]()

    var cur_bias: Scalar[accum_t] = 0

    @parameter
    if has_bias:
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

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var dout_base = batch_id * dout_batch_stride + channel_id * dout_c_stride
    var dx_base = batch_id * dx_batch_stride + channel_id * dx_c_stride

    # Per-thread accumulators (persist across chunks).
    var local_dweight = SIMD[accum_t, width](0)
    var local_dbias: Scalar[accum_t] = 0

    var n_chunks: Int = ceildiv(seqlen, kChunkSize)

    # Initialise smem_dout slot 0 to zero. On the first iteration (last
    # chunk) thread kNThreads-1 reads smem_dout[0] for its halo; for the
    # last chunk that halo is past the seqlen end and must be zero.
    if tidx == 0:

        @parameter
        for i in range(kNElts):
            smem_dout[i] = 0

    barrier()

    # Reverse iteration: chunk n_chunks-1 down to 0.
    for chunk_rev in range(n_chunks):
        var chunk: Int = n_chunks - 1 - chunk_rev
        var chunk_start: Int = chunk * kChunkSize
        var seq_start: Int = chunk_start + tidx * kNElts

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

        @parameter
        if contig_inner and aligned_seq:
            x_curr = x_ptr.load[width=kNElts, alignment=16](
                x_base + seq_start
            ).cast[accum_t]()
            dout_curr = dout_ptr.load[width=kNElts, alignment=16](
                dout_base + seq_start
            ).cast[accum_t]()
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                x_curr = x_ptr.load[width=kNElts, alignment=16](
                    x_base + seq_start
                ).cast[accum_t]()
                dout_curr = dout_ptr.load[width=kNElts, alignment=16](
                    dout_base + seq_start
                ).cast[accum_t]()
            else:

                @parameter
                for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        x_curr[i] = x_ptr[x_base + t].cast[accum_t]()
                        dout_curr[i] = dout_ptr[dout_base + t].cast[accum_t]()
        else:

            @parameter
            for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    x_curr[i] = x_ptr[x_base + t * x_l_stride].cast[accum_t]()
                    dout_curr[i] = dout_ptr[dout_base + t * dout_l_stride].cast[
                        accum_t
                    ]()

        # ---- [P2] x halo from previous chunk ----
        # tidx==0: load previous kNElts of x from global (chunk>0 only).
        # tidx>0:  read previous thread's x_curr from smem.
        var x_prev = SIMD[accum_t, kNElts](0)
        if tidx == 0 and chunk > 0:

            @parameter
            for i in range(kNElts):
                var t = chunk_start - kNElts + i
                if t >= 0:

                    @parameter
                    if contig_inner:
                        x_prev[i] = x_ptr[x_base + t].cast[accum_t]()
                    else:
                        x_prev[i] = x_ptr[x_base + t * x_l_stride].cast[
                            accum_t
                        ]()

        # Publish x_curr to smem so the next thread can pick it up as halo.
        @parameter
        for i in range(kNElts):
            smem_x[tidx * kNElts + i] = x_curr[i].cast[dtype]()

        barrier()  # smem_x writes visible

        if tidx > 0:

            @parameter
            for i in range(kNElts):
                x_prev[i] = smem_x[(tidx - 1) * kNElts + i].cast[accum_t]()

        # ---- [P3] derive dpre from dout (and silu' if activation was silu) ----
        # When apply_silu, dpre = dout * silu'(pre); otherwise dpre = dout
        # (the bias-only forward has identity gradient w.r.t. pre).
        # silu'(pre) requires recomputing pre = bias + sum_k weights[k] *
        # x_window[k]; x_window comes from [x_prev || x_curr] using the same
        # offset arithmetic as the forward.
        var dpre = SIMD[accum_t, kNElts](0)

        @parameter
        if apply_silu:

            @parameter
            for i in range(kNElts):

                @parameter
                if aligned_seq:
                    var pre: Scalar[accum_t] = cur_bias

                    @parameter
                    for k in range(width):
                        alias offset_w: Int = k - (width - 1)

                        @parameter
                        if i + offset_w >= 0:
                            pre += x_curr[i + offset_w] * weights[k]
                        else:
                            pre += x_prev[kNElts + i + offset_w] * weights[k]

                    var sig: Scalar[accum_t] = 1.0 / (1.0 + exp(-pre))
                    var silu_grad: Scalar[accum_t] = sig * (
                        1.0 + pre * (1.0 - sig)
                    )
                    dpre[i] = dout_curr[i] * silu_grad
                else:
                    if seq_start + i < seqlen:
                        var pre: Scalar[accum_t] = cur_bias

                        @parameter
                        for k in range(width):
                            alias offset_w: Int = k - (width - 1)

                            @parameter
                            if i + offset_w >= 0:
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
        else:
            # No silu — dpre is just dout. `dout_curr` already has out-of-
            # bounds positions zeroed (loaded that way in [P1]).
            dpre = dout_curr

        # dbias += sum(dpre) — only when there's a bias to accumulate into.
        @parameter
        if has_bias:
            local_dbias += dpre.reduce_add()

        # ---- [P4] dout halo from next chunk (already in smem_dout) ----
        # Three-barrier dance, mirroring upstream:
        #   1. tidx>0 writes its dpre to smem_dout[tidx*kNElts..]
        #      (slot 0 still holds NEXT chunk's thread-0 dpre, which thread
        #       kNThreads-1 will read for its halo).
        #   2. all threads read smem_dout[((tidx+1) % kNThreads)*kNElts..]
        #   3. thread 0 writes its dpre to slot 0 for the next iteration.
        barrier()  # all reads of smem_x done; safe to reuse smem_dout

        if tidx > 0:

            @parameter
            for i in range(kNElts):
                smem_dout[tidx * kNElts + i] = dpre[i]

        barrier()  # tidx>0 writes visible; slot 0 still holds NEXT chunk's data

        var halo_thread = tidx + 1 if tidx < kNThreads - 1 else 0
        var dout_halo = SIMD[accum_t, kNElts](0)

        @parameter
        for i in range(kNElts):
            dout_halo[i] = smem_dout[halo_thread * kNElts + i]

        barrier()  # all halo reads done; thread 0 may now stomp slot 0

        if tidx == 0:

            @parameter
            for i in range(kNElts):
                smem_dout[i] = dpre[i]

        # ---- [P5] dx = anti-causal conv on dpre || dout_halo ----
        # dx[i] = sum_w weights[w] * combined[i + (W-1-w)]
        # combined = [dpre (kNElts) || dout_halo (kNElts)]
        var dx_vals = SIMD[accum_t, kNElts](0)

        @parameter
        for i in range(kNElts):

            @parameter
            for k in range(width):
                alias halo_idx: Int = i + (width - 1) - k

                @parameter
                if halo_idx < kNElts:
                    dx_vals[i] += weights[k] * dpre[halo_idx]
                else:
                    dx_vals[i] += weights[k] * dout_halo[halo_idx - kNElts]

        @parameter
        if contig_inner and aligned_seq:
            dx_ptr.store[alignment=16](
                dx_base + seq_start, dx_vals.cast[dtype]()
            )
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                dx_ptr.store[alignment=16](
                    dx_base + seq_start, dx_vals.cast[dtype]()
                )
            else:

                @parameter
                for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        dx_ptr[dx_base + t] = dx_vals[i].cast[dtype]()
        else:

            @parameter
            for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    dx_ptr[dx_base + t * dx_l_stride] = dx_vals[i].cast[dtype]()

        # ---- [P6] dweight[w] += sum_i x_curr[i] * combined[i + (W-1-w)] ----
        @parameter
        for k in range(width):
            var acc: Scalar[accum_t] = 0

            @parameter
            for i in range(kNElts):
                alias halo_idx_dw: Int = i + (width - 1) - k

                @parameter
                if halo_idx_dw < kNElts:
                    acc += x_curr[i] * dpre[halo_idx_dw]
                else:
                    acc += x_curr[i] * dout_halo[halo_idx_dw - kNElts]
            local_dweight[k] += acc

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
    @parameter
    for k in range(width):
        var block_dw_k = _block_sum_f32[block_size=kNThreads](local_dweight[k])
        if tidx == 0:
            _ = Atomic[DType.float32, scope="device"].fetch_add[
                ordering=Consistency.MONOTONIC
            ](
                dweight_acc_ptr + channel_id * width + k,
                block_dw_k,
            )

    @parameter
    if has_bias:
        var block_dbias = _block_sum_f32[block_size=kNThreads](local_dbias)
        if tidx == 0:
            _ = Atomic[DType.float32, scope="device"].fetch_add[
                ordering=Consistency.MONOTONIC
            ](dbias_acc_ptr + channel_id, block_dbias)
