"""Direct Python -> Mojo extension that launches the causal_conv1d GPU
kernel without going through MAX's CustomOpLibrary.

Built as a CPython extension via:
    mojo build causal_conv1d_native.mojo --emit shared-lib -o causal_conv1d_native.so

Then importable as `from causal_conv1d_mojo._native import causal_conv1d_native`.

The single entry point `causal_conv1d_fwd_fp16_w4_silu_bias` is specialized
for the benchmark workload: fp16 inputs, width=4, has_bias=True,
has_initial_states=False, activation="silu". (Specialization grid is
intentionally tiny for now; the goal is to measure the framework-overhead
delta vs the MAX path, not to ship a general op.)

Two kernel variants are baked in: a fast path that assumes the innermost
strides are 1 (the contiguous case) and a fallback that takes the inner
strides as runtime args. Folding the stride-1 multiplies into the fast
path matters: passing inner strides as runtime args around the kernel,
even when always 1, costs ~2x kernel time on a memory-bound workload
because the compiler can no longer constant-fold the index math.
"""

from std.os import abort
from std.os.atomic import Atomic
from std.math import ceildiv, exp
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext
from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
    barrier,
)
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block as gpu_block
from layout import Layout, LayoutTensor


comptime kNThreads: Int = 128
comptime kNElts: Int = 4


fn _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))


fn fwd_kernel_contig[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    activation: StaticString,
](
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    weight_c_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
):
    """Fast path: x.stride(2)=1, weight.stride(1)=1, output.stride(2)=1."""
    alias accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):
        weights[k] = weight_ptr[weight_base + k].cast[accum_t]()

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
                val = x_ptr[x_base + src_t].cast[accum_t]()
            acc += val * weights[k]

        @parameter
        if activation == "silu":
            acc = _silu_f32(Float32(acc))

        output_ptr[out_base + t] = acc.cast[dtype]()


fn fwd_kernel_strided[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    activation: StaticString,
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
    """Slow path: any inner stride may be non-1."""
    alias accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):
        weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[accum_t]()

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
                val = x_ptr[x_base + src_t * x_l_stride].cast[accum_t]()
            acc += val * weights[k]

        @parameter
        if activation == "silu":
            acc = _silu_f32(Float32(acc))

        output_ptr[out_base + t * out_l_stride] = acc.cast[dtype]()


fn bwd_dx_kernel[
    dtype: DType,
    width: Int,
    contig_inner: Bool,
](
    seqlen: Int,
    dpre_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dx_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dpre_batch_stride: Int,
    dpre_c_stride: Int,
    dpre_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    dx_batch_stride: Int,
    dx_c_stride: Int,
    dx_l_stride: Int,
):
    """Backward of `out = silu(conv1d(x, w) + b)` w.r.t. x, given dpre.

    dpre[b, d, t] = dout[b, d, t] * silu'(pre[b, d, t]) is computed by the
    caller (pytorch). This kernel does the anti-causal conv:

        dx[b, d, t] = sum_k dpre[b, d, t + k] * weight[d, W-1-k]

    Same per-block structure as the forward kernel: one block per
    (batch, channel, chunk_l). Each thread emits kNElts dx positions;
    weights are pre-loaded into per-block fp32 registers in *reversed*
    order so the inner loop stays a straight FMA.
    """
    alias accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    # Load weights in reversed order: weights_rev[k] = weight[d, W-1-k].
    var weights_rev = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):

        @parameter
        if contig_inner:
            weights_rev[k] = weight_ptr[
                weight_base + (width - 1 - k)
            ].cast[accum_t]()
        else:
            weights_rev[k] = weight_ptr[
                weight_base + (width - 1 - k) * weight_w_stride
            ].cast[accum_t]()

    var seq_start = chunk_id * kNThreads * kNElts + tidx * kNElts
    if seq_start >= seqlen:
        return

    var dpre_base = batch_id * dpre_batch_stride + channel_id * dpre_c_stride
    var dx_base = batch_id * dx_batch_stride + channel_id * dx_c_stride

    @parameter
    for i in range(kNElts):
        var t = seq_start + i
        if t >= seqlen:
            break
        var acc: Scalar[accum_t] = 0

        @parameter
        for k in range(width):
            var src_t = t + k  # forward direction
            var val: Scalar[accum_t] = 0
            if src_t < seqlen:

                @parameter
                if contig_inner:
                    val = dpre_ptr[dpre_base + src_t].cast[accum_t]()
                else:
                    val = dpre_ptr[dpre_base + src_t * dpre_l_stride].cast[accum_t]()
            acc += val * weights_rev[k]

        @parameter
        if contig_inner:
            dx_ptr[dx_base + t] = acc.cast[dtype]()
        else:
            dx_ptr[dx_base + t * dx_l_stride] = acc.cast[dtype]()


fn bwd_full_kernel[
    dtype: DType,
    width: Int,
    contig_inner: Bool,
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
    """Fused backward: dx + (dweight, dbias) accumulation, single block
    per (batch, channel).

    Grid: (dim, batch). Each block walks the entire seqlen via an inner
    chunk loop, accumulating per-thread dweight/dbias across chunks in
    fp32 registers. Only ONE block per channel-batch pair contributes
    atomics, so the global atomic_add contention is `B` per channel,
    not `B * num_chunks_l` (which is what the old chunk-grid layout
    cost). This mirrors upstream's causal_conv1d_bwd_kernel grid.

    Per-chunk: recompute pre, fold silu' into dpre, store to smem; the
    first (W-1) threads also compute dpre for the next chunk's first
    (W-1) positions for the dx halo. Then dx is the anti-causal conv on
    smem with reversed weights.
    """
    alias accum_t = DType.float32
    alias kBufSize: Int = kNThreads * kNElts + (width - 1)
    alias smem_layout = Layout.row_major(kBufSize)

    var tidx: Int = thread_idx.x
    var channel_id: Int = block_idx.x
    var batch_id: Int = block_idx.y

    # Load weights, bias into per-block fp32 registers.
    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):

        @parameter
        if contig_inner:
            weights[k] = weight_ptr[weight_base + k].cast[accum_t]()
        else:
            weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[accum_t]()

    var cur_bias: Scalar[accum_t] = bias_ptr[channel_id].cast[accum_t]()

    # Smem buffer for dpre values (in fp32 for precision).
    var dpre_smem = LayoutTensor[
        accum_t,
        smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var dout_base = batch_id * dout_batch_stride + channel_id * dout_c_stride
    var dx_base = batch_id * dx_batch_stride + channel_id * dx_c_stride

    # Per-thread accumulators for dweight (fp32, length W) and dbias.
    # Persist ACROSS the chunk loop -- this is what makes one atomic_add
    # per channel-batch enough.
    var local_dweight = SIMD[accum_t, width](0)
    var local_dbias: Scalar[accum_t] = 0

    var n_chunks: Int = ceildiv(seqlen, kNThreads * kNElts)

    for chunk in range(n_chunks):
        var chunk_start: Int = chunk * kNThreads * kNElts
        var seq_start: Int = chunk_start + tidx * kNElts

        # === Phase 1: dpre + accumulate dweight, dbias for this chunk ===
        @parameter
        for i in range(kNElts):
            var t = seq_start + i
            var dpre_val: Scalar[accum_t] = 0
            if t < seqlen:
                var pre: Scalar[accum_t] = cur_bias
                var x_window = SIMD[accum_t, width](0)

                @parameter
                for k in range(width):
                    var src_t = t + k - (width - 1)
                    if src_t >= 0:

                        @parameter
                        if contig_inner:
                            x_window[k] = x_ptr[x_base + src_t].cast[accum_t]()
                        else:
                            x_window[k] = x_ptr[
                                x_base + src_t * x_l_stride
                            ].cast[accum_t]()
                    pre += x_window[k] * weights[k]

                var sig: Scalar[accum_t] = 1.0 / (1.0 + exp(-pre))
                var silu_grad: Scalar[accum_t] = sig * (1.0 + pre * (1.0 - sig))
                var dout_val: Scalar[accum_t]

                @parameter
                if contig_inner:
                    dout_val = dout_ptr[dout_base + t].cast[accum_t]()
                else:
                    dout_val = dout_ptr[
                        dout_base + t * dout_l_stride
                    ].cast[accum_t]()

                dpre_val = dout_val * silu_grad
                local_dbias += dpre_val

                @parameter
                for k in range(width):
                    local_dweight[k] += dpre_val * x_window[k]

            dpre_smem[tidx * kNElts + i] = dpre_val

        # === Phase 2: halo dpre for next chunk's first (W-1) positions ===
        if tidx < (width - 1):
            var t_halo = chunk_start + kNThreads * kNElts + tidx
            var dpre_halo: Scalar[accum_t] = 0
            if t_halo < seqlen:
                var pre: Scalar[accum_t] = cur_bias

                @parameter
                for k in range(width):
                    var src_t = t_halo + k - (width - 1)
                    var val: Scalar[accum_t] = 0
                    if src_t >= 0:

                        @parameter
                        if contig_inner:
                            val = x_ptr[x_base + src_t].cast[accum_t]()
                        else:
                            val = x_ptr[
                                x_base + src_t * x_l_stride
                            ].cast[accum_t]()
                    pre += val * weights[k]

                var sig: Scalar[accum_t] = 1.0 / (1.0 + exp(-pre))
                var silu_grad: Scalar[accum_t] = sig * (1.0 + pre * (1.0 - sig))
                var dout_val: Scalar[accum_t]

                @parameter
                if contig_inner:
                    dout_val = dout_ptr[dout_base + t_halo].cast[accum_t]()
                else:
                    dout_val = dout_ptr[
                        dout_base + t_halo * dout_l_stride
                    ].cast[accum_t]()

                dpre_halo = dout_val * silu_grad

            dpre_smem[kNThreads * kNElts + tidx] = dpre_halo

        barrier()

        # === Phase 3: dx (anti-causal conv from smem) ===
        @parameter
        for i in range(kNElts):
            var t = seq_start + i
            if t < seqlen:
                var dx_val: Scalar[accum_t] = 0

                @parameter
                for k in range(width):
                    var smem_idx = tidx * kNElts + i + k
                    var dpre_smem_val = dpre_smem[smem_idx][0]
                    dx_val += dpre_smem_val * weights[width - 1 - k]

                @parameter
                if contig_inner:
                    dx_ptr[dx_base + t] = dx_val.cast[dtype]()
                else:
                    dx_ptr[dx_base + t * dx_l_stride] = dx_val.cast[dtype]()

        # Make sure all phase-3 reads are done before next chunk's
        # phase 1 overwrites the smem buffer.
        barrier()

    # === Phase 4: block-reduce dweight, dbias and atomic-add to global ===
    @parameter
    for k in range(width):
        var block_dw_k = gpu_block.sum[
            block_size=kNThreads, broadcast=False
        ](local_dweight[k])
        if tidx == 0:
            _ = Atomic.fetch_add(
                dweight_acc_ptr + channel_id * width + k,
                block_dw_k,
            )

    var block_dbias = gpu_block.sum[
        block_size=kNThreads, broadcast=False
    ](local_dbias)
    if tidx == 0:
        _ = Atomic.fetch_add(dbias_acc_ptr + channel_id, block_dbias)


def causal_conv1d_fwd_fp16_w4_silu_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Specialized launch: fp16 / width=4 / has_bias=True / silu.

    Python tuple positional args (16, in order):
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
        15 cuda_stream_handle (int)  -- torch.cuda.current_stream().cuda_stream
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
    var stream_handle_addr: Int = Int(py=args[15])

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

    if x_l_stride == 1 and w_w_stride == 1 and o_l_stride == 1:
        var compiled = ctx.compile_function[
            fwd_kernel_contig[DType.float16, 4, True, "silu"],
            fwd_kernel_contig[DType.float16, 4, True, "silu"],
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
            w_c_stride,
            o_b_stride,
            o_c_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )
    else:
        var compiled = ctx.compile_function[
            fwd_kernel_strided[DType.float16, 4, True, "silu"],
            fwd_kernel_strided[DType.float16, 4, True, "silu"],
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

    return PythonObject(None)


def causal_conv1d_bwd_dx_fp16_w4(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Specialized backward dx launch: fp16 / width=4.

    Python tuple positional args (15, in order):
        0  dpre_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  dx_data_ptr  (int)
        3  batch  (int)
        4  dim    (int)
        5  seqlen (int)
        6  dpre_batch_stride  (int)
        7  dpre_c_stride      (int)
        8  dpre_l_stride      (int)
        9  weight_c_stride (int)
        10 weight_w_stride (int)
        11 dx_batch_stride  (int)
        12 dx_c_stride      (int)
        13 dx_l_stride      (int)
        14 cuda_stream_handle (int)
    """
    var dpre_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var dx_addr: Int = Int(py=args[2])

    var dpre_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dpre_addr
    )
    var w_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var dx_ptr = UnsafePointer[Scalar[DType.float16], MutAnyOrigin](
        unsafe_from_address=dx_addr
    )

    var batch_int: Int = Int(py=args[3])
    var dim_int: Int = Int(py=args[4])
    var seqlen_int: Int = Int(py=args[5])

    var dpre_b_stride: Int = Int(py=args[6])
    var dpre_c_stride: Int = Int(py=args[7])
    var dpre_l_stride: Int = Int(py=args[8])
    var w_c_stride: Int = Int(py=args[9])
    var w_w_stride: Int = Int(py=args[10])
    var dx_b_stride: Int = Int(py=args[11])
    var dx_c_stride: Int = Int(py=args[12])
    var dx_l_stride: Int = Int(py=args[13])
    var stream_handle_addr: Int = Int(py=args[14])

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

    if dpre_l_stride == 1 and w_w_stride == 1 and dx_l_stride == 1:
        var compiled = ctx.compile_function[
            bwd_dx_kernel[DType.float16, 4, True],
            bwd_dx_kernel[DType.float16, 4, True],
        ]()
        stream.enqueue_function(
            compiled,
            seqlen_int,
            dpre_ptr,
            w_ptr,
            dx_ptr,
            dpre_b_stride,
            dpre_c_stride,
            dpre_l_stride,
            w_c_stride,
            w_w_stride,
            dx_b_stride,
            dx_c_stride,
            dx_l_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )
    else:
        var compiled = ctx.compile_function[
            bwd_dx_kernel[DType.float16, 4, False],
            bwd_dx_kernel[DType.float16, 4, False],
        ]()
        stream.enqueue_function(
            compiled,
            seqlen_int,
            dpre_ptr,
            w_ptr,
            dx_ptr,
            dpre_b_stride,
            dpre_c_stride,
            dpre_l_stride,
            w_c_stride,
            w_w_stride,
            dx_b_stride,
            dx_c_stride,
            dx_l_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )

    return PythonObject(None)


def causal_conv1d_bwd_full_fp16_w4_silu_bias(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Specialized fused backward launch: fp16 / width=4 / has_bias / silu.

    Caller must zero dweight_acc / dbias_acc (fp32 buffers) before this call.

    Python tuple positional args (24, in order):
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
        21 cuda_stream_handle (int)
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
    var stream_handle_addr: Int = Int(py=args[21])

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One block per (channel, batch); block walks all chunks of seqlen.
    var grid = (dim_int, batch_int)

    if (
        x_l_stride == 1
        and w_w_stride == 1
        and dout_l_stride == 1
        and dx_l_stride == 1
    ):
        var compiled = ctx.compile_function[
            bwd_full_kernel[DType.float16, 4, True],
            bwd_full_kernel[DType.float16, 4, True],
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
    else:
        var compiled = ctx.compile_function[
            bwd_full_kernel[DType.float16, 4, False],
            bwd_full_kernel[DType.float16, 4, False],
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

    return PythonObject(None)


@export
def PyInit_causal_conv1d_native() -> PythonObject:
    try:
        var m = PythonModuleBuilder("causal_conv1d_native")
        m.def_py_function[causal_conv1d_fwd_fp16_w4_silu_bias](
            "causal_conv1d_fwd_fp16_w4_silu_bias"
        )
        m.def_py_function[causal_conv1d_bwd_dx_fp16_w4](
            "causal_conv1d_bwd_dx_fp16_w4"
        )
        m.def_py_function[causal_conv1d_bwd_full_fp16_w4_silu_bias](
            "causal_conv1d_bwd_full_fp16_w4_silu_bias"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
