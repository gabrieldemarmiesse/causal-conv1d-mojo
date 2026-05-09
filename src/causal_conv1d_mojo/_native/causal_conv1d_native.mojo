"""Direct Python -> Mojo extension for causal_conv1d (fp16, width=4,
has_bias=True), no MAX framework.

Built as a CPython extension via:
    mojo build causal_conv1d_native.mojo --emit shared-lib -o causal_conv1d_native.so

Then importable as `from causal_conv1d_mojo._native import causal_conv1d_native`.

Four entry points: GPU + CPU × forward + backward, each fp16 / width=4 /
has_bias=True. Activation (silu vs none) and contiguity of inner strides
are passed as runtime ints/bools and dispatched to comptime-specialised
kernels at the launcher.

Folding the stride-1 multiplies into the fast contig path matters:
passing inner strides as runtime args around the kernel, even when
always 1, costs ~2× kernel time on a memory-bound workload because the
compiler can no longer constant-fold the index math.
"""

from std.os import abort
from std.os.atomic import Atomic, Consistency
from std.algorithm import sync_parallelize
from std.math import ceildiv, exp
from std.memory import OpaquePointer, stack_allocation
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import llvm_intrinsic
from std.gpu.host import DeviceContext
from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
    barrier,
)
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor


comptime kNThreads: Int = 128
# Forward: kNElts=4 (8 bytes/thread). The fwd grid is (ceildiv(seqlen,
# kNThreads*kNElts), dim, batch); raising kNElts shrinks the grid and
# costs parallelism on small seqlens, even though it would help vector
# load throughput.
comptime kNElts: Int = 4
# Backward: kNElts=8 (16 bytes/thread → LDG.E.128). Bwd has only one
# block per (B,D) (it walks the full seqlen via an inner chunk loop), so
# raising kNElts costs no parallelism — and it doubles per-thread global
# bandwidth via 128-bit aligned vector loads. With the default
# kNElts=4 the load is only 8-byte-aligned and the compiler emits four
# scalar `LDG.E.U16`s instead of one `LDG.E.128`.
comptime kNEltsBwd: Int = 4


fn _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))


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
    return llvm_intrinsic[
        "llvm.nvvm.shfl.sync.bfly.f32", Float32
    ](Int32(-1), val, offset, Int32(31))


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
        n_warps, DType.float32, address_space = AddressSpace.SHARED
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


fn fwd_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
](
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    """fp16 / silu / has_bias / width=W causal conv1d forward, GPU.

    `contig_inner` is the comptime fast path: when True, the innermost
    axes of x / weight / out have stride=1 and we drop the
    `* x_l_stride` / `* weight_w_stride` / `* out_l_stride` multiplies
    so the compiler can constant-fold the index math (~2× kernel time
    on memory-bound shapes if we don't).
    """
    alias accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):

        @parameter
        if contig_inner:
            weights[k] = weight_ptr[weight_base + k].cast[accum_t]()
        else:
            weights[k] = weight_ptr[
                weight_base + k * weight_w_stride
            ].cast[accum_t]()

    var cur_bias: Scalar[accum_t] = 0

    @parameter
    if has_bias:
        cur_bias = bias_ptr[channel_id].cast[accum_t]()

    var seq_start = chunk_id * kNThreads * kNElts + tidx * kNElts
    if seq_start >= seqlen:
        return

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var out_base = batch_id * out_batch_stride + channel_id * out_c_stride

    @parameter
    for i in range(kNElts):
        var t = seq_start + i
        if t >= seqlen:
            break
        var acc: Scalar[accum_t] = cur_bias

        @parameter
        for k in range(width):
            var src_t = t + k - (width - 1)
            var val: Scalar[accum_t]
            if src_t < 0:
                val = 0
            else:

                @parameter
                if contig_inner:
                    val = x_ptr[x_base + src_t].cast[accum_t]()
                else:
                    val = x_ptr[
                        x_base + src_t * x_l_stride
                    ].cast[accum_t]()
            acc += val * weights[k]

        @parameter
        if apply_silu:
            acc = _silu_f32(Float32(acc))

        @parameter
        if contig_inner:
            output_ptr[out_base + t] = acc.cast[dtype]()
        else:
            output_ptr[out_base + t * out_l_stride] = acc.cast[dtype]()


fn bwd_full_kernel[
    dtype: DType,
    width: Int,
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
    """Fused backward (silu + bias): dx + dweight + dbias, one block per (B,D).

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
    # Local alias: bwd uses kNEltsBwd=8 (16B vector loads). Forward uses
    # kNElts=4 because its grid scales with seqlen and we don't want to
    # halve parallelism.
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
            weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[accum_t]()

    var cur_bias: Scalar[accum_t] = bias_ptr[channel_id].cast[accum_t]()

    # smem_x: each thread's kNElts x values for the next thread's halo.
    # smem_dout: each thread's kNElts dpre values (post-silu') for the
    # *previous chunk's* dx halo (we walk in reverse).
    var smem_x = stack_allocation[
        kChunkSize, dtype, address_space = AddressSpace.SHARED
    ]()
    var smem_dout = stack_allocation[
        kChunkSize, accum_t, address_space = AddressSpace.SHARED
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
        # alignment=16 → 128-bit aligned LDG.E.128 (one load per tensor per
        # thread). The default alignment is align_of[fp16]=2, which blocks
        # the compiler from merging into a 128-bit load even when widths
        # line up. Each thread's seq_start is a multiple of kNElts=8 fp16
        # = 16 bytes, and base addresses are large multiples of the
        # channel/batch strides which are also 16-aligned for standard
        # PyTorch row-major tensors.
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
                    dout_curr[i] = dout_ptr[
                        dout_base + t * dout_l_stride
                    ].cast[accum_t]()

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
                        x_prev[i] = x_ptr[
                            x_base + t * x_l_stride
                        ].cast[accum_t]()

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

        # dbias += sum(dpre)
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
                ordering = Consistency.MONOTONIC
            ](
                dweight_acc_ptr + channel_id * width + k,
                block_dw_k,
            )

    var block_dbias = _block_sum_f32[block_size=kNThreads](local_dbias)
    if tidx == 0:
        _ = Atomic[DType.float32, scope="device"].fetch_add[
            ordering = Consistency.MONOTONIC
        ](dbias_acc_ptr + channel_id, block_dbias)


# ===-----------------------------------------------------------------------=== #
# CPU implementations
# ===-----------------------------------------------------------------------=== #
# Pure-mojo CPU forward + backward, called when the user passes CPU tensors
# instead of CUDA. The point is that the package works on a machine without
# a GPU without forcing users to `pip install causal-conv1d` (which needs a
# C++ toolchain to source-build). These are the slow path; the GPU kernels
# above are the real product.
#
# Pattern follows max/kernels/src/state_space/causal_conv1d.mojo:
# parallelise over (batch, channel) work items via `sync_parallelize`. Each
# worker pre-loads its row of weights into a register, then walks seqlen.


@always_inline
fn _cpu_dpre_at[
    dtype: DType,
    width: Int,
    apply_silu: Bool,
](
    t: Int,
    seqlen: Int,
    bias_v: Float32,
    weights: SIMD[DType.float32, width],
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_base: Int,
    x_l_stride: Int,
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_base: Int,
    dout_l_stride: Int,
) -> Float32:
    """`dpre[t]` for the CPU backward, 0 if `t` is out of [0, seqlen).

    With `apply_silu`, `dpre[t] = silu'(pre[t]) * dout[t]` (the bias-aware
    sigmoid-derivative path). Without it, `dpre[t] = dout[t]` directly —
    bias-only forward has identity gradient w.r.t. pre.
    """
    if t < 0 or t >= seqlen:
        return 0

    @parameter
    if not apply_silu:
        return dout_ptr[dout_base + t * dout_l_stride].cast[DType.float32]()

    var pre: Float32 = bias_v

    @parameter
    for k in range(width):
        var src_t = t + k - (width - 1)
        if src_t >= 0:
            pre += weights[k] * x_ptr[
                x_base + src_t * x_l_stride
            ].cast[DType.float32]()
    var sig: Float32 = 1.0 / (1.0 + exp(-pre))
    var silu_grad: Float32 = sig * (1.0 + pre * (1.0 - sig))
    var dout_v = dout_ptr[
        dout_base + t * dout_l_stride
    ].cast[DType.float32]()
    return dout_v * silu_grad


fn fwd_kernel_cpu[
    dtype: DType,
    width: Int,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    """fp16 / has_bias / width=W causal conv1d forward, CPU path.

    `apply_silu` (comptime): apply silu (= swish) on the output, or skip
    it for the bias-only `activation=None` case.
    """
    alias accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var bias_v: Scalar[accum_t] = bias_ptr[d].cast[accum_t]()

        var weights = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            weights[k] = weight_ptr[
                d * weight_c_stride + k * weight_w_stride
            ].cast[accum_t]()

        var x_base = b * x_batch_stride + d * x_c_stride
        var out_base = b * out_batch_stride + d * out_c_stride

        for t in range(seqlen):
            var pre: Scalar[accum_t] = bias_v

            @parameter
            for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    pre += weights[k] * x_ptr[
                        x_base + src_t * x_l_stride
                    ].cast[accum_t]()

            var out_v: Scalar[accum_t]

            @parameter
            if apply_silu:
                out_v = _silu_f32(pre)
            else:
                out_v = pre
            output_ptr[out_base + t * out_l_stride] = out_v.cast[dtype]()

    sync_parallelize[process_bc](batch * dim)


fn bwd_kernel_cpu[
    dtype: DType,
    width: Int,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
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
    """fp16 / silu / has_bias / width=W causal conv1d backward, CPU path.

    Computes `dx, dweight, dbias` from `x, weight, bias, dout`. Uses a
    sliding window of `width` dpre values to avoid materialising the full
    `dpre` tensor:

        dx[t]       = sum_k weight[W-1-k] * dpre[t + k]
        dweight[w] += sum_t x[t + w - (W-1)] * dpre[t]
        dbias      += sum_t dpre[t]

    where `dpre[t] = silu'(pre[t]) * dout[t]`.

    Parallelised across (batch, channel) workers via `sync_parallelize`.
    Workers may share a `d` (across batches) so the per-channel
    `dweight` / `dbias` accumulators are atomic-added at the end.
    """
    alias accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var bias_v: Scalar[accum_t] = bias_ptr[d].cast[accum_t]()

        var weights = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            weights[k] = weight_ptr[
                d * weight_c_stride + k * weight_w_stride
            ].cast[accum_t]()

        var x_base = b * x_batch_stride + d * x_c_stride
        var dout_base = b * dout_batch_stride + d * dout_c_stride
        var dx_base = b * dx_batch_stride + d * dx_c_stride

        # Sliding window: dpre_win[k] = dpre[t + k]. Prefill with dpre[0..W-1].
        var dpre_win = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            dpre_win[k] = _cpu_dpre_at[dtype, width, apply_silu](
                k,
                seqlen,
                bias_v,
                weights,
                x_ptr,
                x_base,
                x_l_stride,
                dout_ptr,
                dout_base,
                dout_l_stride,
            )

        var local_dweight = SIMD[accum_t, width](0)
        var local_dbias: Scalar[accum_t] = 0

        for t in range(seqlen):
            # dx[t] = sum_k weights[W-1-k] * dpre_win[k]
            var dx_v: Scalar[accum_t] = 0

            @parameter
            for k in range(width):
                dx_v += weights[width - 1 - k] * dpre_win[k]
            dx_ptr[dx_base + t * dx_l_stride] = dx_v.cast[dtype]()

            # dweight[k] += dpre[t] * x[t + k - (W-1)];  dbias += dpre[t]
            var dpre_t: Scalar[accum_t] = dpre_win[0]
            local_dbias += dpre_t

            @parameter
            for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    var x_v = x_ptr[x_base + src_t * x_l_stride].cast[
                        accum_t
                    ]()
                    local_dweight[k] += dpre_t * x_v

            # Slide window left, append dpre[t + W] (or 0 past seqlen).
            @parameter
            for k in range(width - 1):
                dpre_win[k] = dpre_win[k + 1]
            dpre_win[width - 1] = _cpu_dpre_at[dtype, width, apply_silu](
                t + width,
                seqlen,
                bias_v,
                weights,
                x_ptr,
                x_base,
                x_l_stride,
                dout_ptr,
                dout_base,
                dout_l_stride,
            )

        # Atomic-add the (b, d) block's contribution. Multiple parallel
        # workers may target the same `d` across different batches.
        @parameter
        for k in range(width):
            _ = Atomic[DType.float32].fetch_add[
                ordering = Consistency.MONOTONIC
            ](dweight_acc_ptr + d * width + k, local_dweight[k])
        _ = Atomic[DType.float32].fetch_add[
            ordering = Consistency.MONOTONIC
        ](dbias_acc_ptr + d, local_dbias)

    sync_parallelize[process_bc](batch * dim)


def causal_conv1d_fwd_fp16_w4_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Specialized launch: fp16 / width=4 / has_bias=True.

    Python tuple positional args (17, in order):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int)
        3  output_data_ptr  (int)
        4  batch  (int)
        5  dim    (int)
        6  seqlen (int)
        7  x_batch_stride  (int)
        8  x_c_stride      (int)
        9  x_l_stride      (int)
        10 weight_c_stride (int)
        11 weight_w_stride (int)
        12 out_batch_stride  (int)
        13 out_c_stride      (int)
        14 out_l_stride      (int)
        15 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish on the output
        16 cuda_stream_handle (int)  -- torch.cuda.current_stream().cuda_stream
    """

    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])

    var x_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var w_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var b_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=b_addr
    )
    var o_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    var batch_int: Int = Int(py=args[4])
    var dim_int: Int = Int(py=args[5])
    var seqlen_int: Int = Int(py=args[6])

    var x_b_stride: Int = Int(py=args[7])
    var x_c_stride: Int = Int(py=args[8])
    var x_l_stride: Int = Int(py=args[9])
    var w_c_stride: Int = Int(py=args[10])
    var w_w_stride: Int = Int(py=args[11])
    var o_b_stride: Int = Int(py=args[12])
    var o_c_stride: Int = Int(py=args[13])
    var o_l_stride: Int = Int(py=args[14])
    var apply_silu_rt: Bool = Int(py=args[15]) != 0
    var stream_handle_addr: Int = Int(py=args[16])

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    var grid = (
        ceildiv(seqlen_int, kNThreads * kNElts),
        dim_int,
        batch_int,
    )
    var contig_inner_rt: Bool = (
        x_l_stride == 1 and w_w_stride == 1 and o_l_stride == 1
    )

    # Nested @parameter fn captures all runtime args; only the comptime
    # specialisation of `fwd_kernel` differs per leg. Mojo expands one
    # specialised body per (apply_silu, contig_inner) pair, with no
    # runtime branch inside the kernel body.
    @parameter
    fn enqueue_fwd[apply_silu: Bool, contig_inner: Bool]() raises:
        var compiled = ctx.compile_function[
            fwd_kernel[DType.float16, 4, True, apply_silu, contig_inner],
            fwd_kernel[DType.float16, 4, True, apply_silu, contig_inner],
        ]()
        stream.enqueue_function(
            compiled,
            seqlen_int,
            x_ptr,
            w_ptr,
            b_ptr,
            o_ptr,
            x_b_stride,
            x_c_stride,
            x_l_stride,
            w_c_stride,
            w_w_stride,
            o_b_stride,
            o_c_stride,
            o_l_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )

    if apply_silu_rt and contig_inner_rt:
        enqueue_fwd[True, True]()
    elif apply_silu_rt:
        enqueue_fwd[True, False]()
    elif contig_inner_rt:
        enqueue_fwd[False, True]()
    else:
        enqueue_fwd[False, False]()

    return PythonObject(None)


def causal_conv1d_bwd_full_fp16_w4_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Specialized fused backward launch: fp16 / width=4 / has_bias.

    Caller must zero dweight_acc / dbias_acc (fp32 buffers) before this call.

    Python tuple positional args (23, in order):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int)
        3  dout_data_ptr  (int)
        4  dx_data_ptr  (int)
        5  dweight_acc_data_ptr  (int, fp32)
        6  dbias_acc_data_ptr  (int, fp32)
        7  batch  (int)
        8  dim    (int)
        9  seqlen (int)
        10 x_batch_stride  (int)
        11 x_c_stride      (int)
        12 x_l_stride      (int)
        13 weight_c_stride (int)
        14 weight_w_stride (int)
        15 dout_batch_stride  (int)
        16 dout_c_stride      (int)
        17 dout_l_stride      (int)
        18 dx_batch_stride  (int)
        19 dx_c_stride      (int)
        20 dx_l_stride      (int)
        21 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish was applied on fwd
        22 cuda_stream_handle (int)
    """
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var dout_addr: Int = Int(py=args[3])
    var dx_addr: Int = Int(py=args[4])
    var dweight_acc_addr: Int = Int(py=args[5])
    var dbias_acc_addr: Int = Int(py=args[6])

    var x_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var w_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var b_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=b_addr
    )
    var dout_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dout_addr
    )
    var dx_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dx_addr
    )
    var dweight_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dweight_acc_addr
    )
    var dbias_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dbias_acc_addr
    )

    var batch_int: Int = Int(py=args[7])
    var dim_int: Int = Int(py=args[8])
    var seqlen_int: Int = Int(py=args[9])

    var x_b_stride: Int = Int(py=args[10])
    var x_c_stride: Int = Int(py=args[11])
    var x_l_stride: Int = Int(py=args[12])
    var w_c_stride: Int = Int(py=args[13])
    var w_w_stride: Int = Int(py=args[14])
    var dout_b_stride: Int = Int(py=args[15])
    var dout_c_stride: Int = Int(py=args[16])
    var dout_l_stride: Int = Int(py=args[17])
    var dx_b_stride: Int = Int(py=args[18])
    var dx_c_stride: Int = Int(py=args[19])
    var dx_l_stride: Int = Int(py=args[20])
    var apply_silu_rt: Bool = Int(py=args[21]) != 0
    var stream_handle_addr: Int = Int(py=args[22])

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One block per (channel, batch); block walks all chunks of seqlen.
    var grid = (dim_int, batch_int)

    var contig_inner_rt: Bool = (
        x_l_stride == 1
        and w_w_stride == 1
        and dout_l_stride == 1
        and dx_l_stride == 1
    )
    var aligned_seq_rt: Bool = seqlen_int % (kNThreads * kNEltsBwd) == 0

    @parameter
    fn enqueue_bwd[
        apply_silu: Bool,
        contig_inner: Bool,
        aligned_seq: Bool,
    ]() raises:
        var compiled = ctx.compile_function[
            bwd_full_kernel[
                DType.float16, 4, apply_silu, contig_inner, aligned_seq
            ],
            bwd_full_kernel[
                DType.float16, 4, apply_silu, contig_inner, aligned_seq
            ],
        ]()
        stream.enqueue_function(
            compiled,
            seqlen_int,
            x_ptr,
            w_ptr,
            b_ptr,
            dout_ptr,
            dx_ptr,
            dweight_acc_ptr,
            dbias_acc_ptr,
            x_b_stride,
            x_c_stride,
            x_l_stride,
            w_c_stride,
            w_w_stride,
            dout_b_stride,
            dout_c_stride,
            dout_l_stride,
            dx_b_stride,
            dx_c_stride,
            dx_l_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )

    if apply_silu_rt and contig_inner_rt and aligned_seq_rt:
        enqueue_bwd[True, True, True]()
    elif apply_silu_rt and contig_inner_rt:
        enqueue_bwd[True, True, False]()
    elif apply_silu_rt:
        enqueue_bwd[True, False, False]()
    elif contig_inner_rt and aligned_seq_rt:
        enqueue_bwd[False, True, True]()
    elif contig_inner_rt:
        enqueue_bwd[False, True, False]()
    else:
        enqueue_bwd[False, False, False]()

    return PythonObject(None)


def causal_conv1d_fwd_cpu_fp16_w4_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU forward: fp16 / width=4 / has_bias.

    Python tuple positional args (5):
        0  x       — torch.Tensor (B, D, L) fp16
        1  weight  — torch.Tensor (D, W) fp16
        2  bias    — torch.Tensor (D,) fp16
        3  output  — torch.Tensor (B, D, L) fp16 (writeable)
        4  apply_silu (int, 0 or 1) — 1 ⇒ silu/swish on the output

    Pointers / shapes / strides are extracted from the tensors via the
    PythonObject method dispatch (data_ptr/shape/stride). The Python
    wrapper just forwards tensors directly — no `data_ptr()` boilerplate
    on the Python side.
    """
    var x = args[0]
    var w = args[1]
    var b = args[2]
    var o = args[3]
    var apply_silu_rt: Bool = Int(py=args[4]) != 0

    var x_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=Int(py=x.data_ptr())
    )
    var w_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=Int(py=w.data_ptr())
    )
    var b_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=Int(py=b.data_ptr())
    )
    var o_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=Int(py=o.data_ptr())
    )

    var batch_int: Int = Int(py=x.shape[0])
    var dim_int: Int = Int(py=x.shape[1])
    var seqlen_int: Int = Int(py=x.shape[2])

    @parameter
    fn run[apply_silu: Bool]() raises:
        fwd_kernel_cpu[DType.float16, 4, apply_silu](
            batch_int,
            dim_int,
            seqlen_int,
            x_ptr,
            w_ptr,
            b_ptr,
            o_ptr,
            Int(py=x.stride(0)),
            Int(py=x.stride(1)),
            Int(py=x.stride(2)),
            Int(py=w.stride(0)),
            Int(py=w.stride(1)),
            Int(py=o.stride(0)),
            Int(py=o.stride(1)),
            Int(py=o.stride(2)),
        )

    if apply_silu_rt:
        run[True]()
    else:
        run[False]()

    return PythonObject(None)


def causal_conv1d_bwd_full_cpu_fp16_w4_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU backward: fp16 / width=4 / has_bias.

    Caller must zero `dweight_acc` / `dbias_acc` (fp32 buffers) before this
    call. Same arg layout as the GPU launcher minus the `cuda_stream_handle`.

    Python tuple positional args (22, in order):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int)
        3  dout_data_ptr  (int)
        4  dx_data_ptr  (int)
        5  dweight_acc_data_ptr  (int, fp32)
        6  dbias_acc_data_ptr  (int, fp32)
        7  batch  (int)
        8  dim    (int)
        9  seqlen (int)
        10 x_batch_stride  (int)
        11 x_c_stride      (int)
        12 x_l_stride      (int)
        13 weight_c_stride (int)
        14 weight_w_stride (int)
        15 dout_batch_stride  (int)
        16 dout_c_stride      (int)
        17 dout_l_stride      (int)
        18 dx_batch_stride  (int)
        19 dx_c_stride      (int)
        20 dx_l_stride      (int)
        21 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish was applied on fwd
    """
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var dout_addr: Int = Int(py=args[3])
    var dx_addr: Int = Int(py=args[4])
    var dweight_acc_addr: Int = Int(py=args[5])
    var dbias_acc_addr: Int = Int(py=args[6])

    var x_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var w_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var b_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=b_addr
    )
    var dout_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dout_addr
    )
    var dx_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dx_addr
    )
    var dweight_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dweight_acc_addr
    )
    var dbias_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dbias_acc_addr
    )

    var batch_int: Int = Int(py=args[7])
    var dim_int: Int = Int(py=args[8])
    var seqlen_int: Int = Int(py=args[9])
    var x_b_stride: Int = Int(py=args[10])
    var x_c_stride: Int = Int(py=args[11])
    var x_l_stride: Int = Int(py=args[12])
    var w_c_stride: Int = Int(py=args[13])
    var w_w_stride: Int = Int(py=args[14])
    var dout_b_stride: Int = Int(py=args[15])
    var dout_c_stride: Int = Int(py=args[16])
    var dout_l_stride: Int = Int(py=args[17])
    var dx_b_stride: Int = Int(py=args[18])
    var dx_c_stride: Int = Int(py=args[19])
    var dx_l_stride: Int = Int(py=args[20])
    var apply_silu_rt: Bool = Int(py=args[21]) != 0

    @parameter
    def run[apply_silu: Bool]():
        bwd_kernel_cpu[DType.float16, 4, apply_silu](
            batch_int,
            dim_int,
            seqlen_int,
            x_ptr,
            w_ptr,
            b_ptr,
            dout_ptr,
            dx_ptr,
            dweight_acc_ptr,
            dbias_acc_ptr,
            x_b_stride,
            x_c_stride,
            x_l_stride,
            w_c_stride,
            w_w_stride,
            dout_b_stride,
            dout_c_stride,
            dout_l_stride,
            dx_b_stride,
            dx_c_stride,
            dx_l_stride,
        )

    if apply_silu_rt:
        run[True]()
    else:
        run[False]()

    return PythonObject(None)


@export
def PyInit_causal_conv1d_native() -> PythonObject:
    try:
        var m = PythonModuleBuilder("causal_conv1d_native")
        m.def_py_function[causal_conv1d_fwd_fp16_w4_bias](
            "causal_conv1d_fwd_fp16_w4_bias"
        )
        m.def_py_function[causal_conv1d_bwd_full_fp16_w4_bias](
            "causal_conv1d_bwd_full_fp16_w4_bias"
        )
        m.def_py_function[causal_conv1d_fwd_cpu_fp16_w4_bias](
            "causal_conv1d_fwd_cpu_fp16_w4_bias"
        )
        m.def_py_function[causal_conv1d_bwd_full_cpu_fp16_w4_bias](
            "causal_conv1d_bwd_full_cpu_fp16_w4_bias"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
