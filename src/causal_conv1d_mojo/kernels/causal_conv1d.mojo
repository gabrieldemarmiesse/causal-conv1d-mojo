"""Fused depthwise causal conv1d, mirroring modular's
max/kernels/src/state_space/causal_conv1d.mojo style for the GPU path:

* Grid (chunk_l, channel, batch); each block owns one (batch, channel) pair
  and one chunk of the sequence dim.
* Each thread emits kNElts consecutive output positions.
* Weights and bias loaded once per block into registers.
* fp32 accumulation when input is fp16/bf16, cast back at store.

CPU path stays on `foreach` -- simple, correct, fast enough.
final_states is a separate small `foreach` (only fired when requested).
"""

import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from std.utils.index import IndexList
from std.math import ceildiv, exp
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
)
from layout import Layout, LayoutTensor


alias kNThreads = 128
alias kNElts = 4


fn _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))


fn causal_conv1d_fwd_gpu_kernel[
    dtype: DType,
    x_layout: Layout,
    w_layout: Layout,
    o_layout: Layout,
    is_layout: Layout,
    b_layout: Layout,
    width: Int,
    has_bias: Bool,
    has_initial_states: Bool,
    activation: StaticString,
](
    seqlen: Int,
    x: LayoutTensor[dtype, x_layout, MutAnyOrigin],
    weight: LayoutTensor[dtype, w_layout, MutAnyOrigin],
    output: LayoutTensor[dtype, o_layout, MutAnyOrigin],
    initial_states: LayoutTensor[dtype, is_layout, MutAnyOrigin],
    bias: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    weight_c_stride: Int,
    is_batch_stride: Int,
    is_c_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
):
    alias accum_t = DType.float32

    var tidx = thread_idx.x
    var batch_id = block_idx.z
    var channel_id = block_idx.y
    var chunk_id = block_idx.x

    # Load weights into per-block registers (fp32).
    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):
        weights[k] = weight.ptr[weight_base + k].cast[accum_t]()

    # Load bias once per block.
    var cur_bias: Scalar[accum_t] = 0

    @parameter
    if has_bias:
        cur_bias = bias.ptr[channel_id].cast[accum_t]()

    # This thread handles kNElts consecutive output positions.
    var seq_start = chunk_id * kNThreads * kNElts + tidx * kNElts
    if seq_start >= seqlen:
        return

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var out_base = batch_id * out_batch_stride + channel_id * out_c_stride
    var is_base = batch_id * is_batch_stride + channel_id * is_c_stride

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

                @parameter
                if has_initial_states:
                    val = initial_states.ptr[is_base + (t + k)].cast[accum_t]()
                else:
                    val = 0
            else:
                val = x.ptr[x_base + src_t].cast[accum_t]()
            acc += val * weights[k]

        @parameter
        if activation == "silu":
            acc = _silu_f32(Float32(acc))

        output.ptr[out_base + t] = acc.cast[dtype]()


@compiler.register("causal_conv1d_fn")
struct CausalConv1dFn[
    width: Int,
    has_bias: Bool,
    has_initial_states: Bool,
    compute_final_states: Bool,
    activation: StaticString,
]:
    """Fused depthwise causal 1D convolution + carry-state.

    out[b, d, t] = silu_opt( sum_k src[b, d, t + k - (W-1)] * weight[d, k] + bias[d] )
    final_states[b, d, s] = src[b, d, L + s] for s in [0, W-1)
    where src = (initial_states or zeros) ++ x along the sequence dim.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[rank=3, ...],
        final_states: OutputTensor[dtype=outp.dtype, rank=3, ...],
        x: InputTensor[dtype=outp.dtype, rank=3, ...],
        weight: InputTensor[dtype=outp.dtype, rank=2, ...],
        bias: InputTensor[dtype=outp.dtype, rank=1, ...],
        initial_states: InputTensor[dtype=outp.dtype, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        alias W: Int = Self.width
        alias dt = outp.dtype

        comptime assert dt.is_floating_point()

        comptime if is_gpu[target]():
            var gpu_ctx: DeviceContext = ctx.get_device_context()

            var X = x.to_layout_tensor()
            var Wt = weight.to_layout_tensor()
            var O = outp.to_layout_tensor()
            var IS = initial_states.to_layout_tensor()
            var Bs = bias.to_layout_tensor()

            var batch: Int = X.dim(0)
            var dim: Int = X.dim(1)
            var seqlen: Int = X.dim(2)

            var x_batch_stride: Int = X.stride(0)
            var x_c_stride: Int = X.stride(1)
            var weight_c_stride: Int = Wt.stride(0)
            var out_batch_stride: Int = O.stride(0)
            var out_c_stride: Int = O.stride(1)

            # initial_states strides only meaningful when present, but we
            # always read them so the kernel doesn't have to branch on
            # whether the pointer is real -- it just won't dereference
            # initial_states when has_initial_states is False.
            var is_batch_stride: Int = IS.stride(0)
            var is_c_stride: Int = IS.stride(1)

            comptime kernel = causal_conv1d_fwd_gpu_kernel[
                dtype = dt,
                x_layout = X.layout,
                w_layout = Wt.layout,
                o_layout = O.layout,
                is_layout = IS.layout,
                b_layout = Bs.layout,
                width = W,
                has_bias = Self.has_bias,
                has_initial_states = Self.has_initial_states,
                activation = Self.activation,
            ]

            gpu_ctx.enqueue_function[kernel, kernel](
                seqlen,
                X,
                Wt,
                O,
                IS,
                Bs,
                x_batch_stride,
                x_c_stride,
                weight_c_stride,
                is_batch_stride,
                is_c_stride,
                out_batch_stride,
                out_c_stride,
                grid_dim=(
                    ceildiv(seqlen, kNThreads * kNElts),
                    dim,
                    batch,
                ),
                block_dim=(kNThreads,),
            )

        else:
            # CPU path: foreach is fine.
            @parameter
            @always_inline
            def compute_out_cpu[
                simd_width: Int
            ](idx: IndexList[3]) -> SIMD[dt, simd_width]:
                var L = x.dim_size(2)
                var b = idx[0]
                var d = idx[1]
                var t = idx[2]

                var acc = SIMD[dt, simd_width](0)
                if t >= (W - 1) and t + simd_width <= L:

                    @parameter
                    for k in range(W):
                        var src_base = t + k - (W - 1)
                        var x_vec = x.load[simd_width](
                            IndexList[3](b, d, src_base)
                        )
                        var w_scalar = weight.load[1](
                            IndexList[2](d, k)
                        )[0]
                        acc = acc + x_vec * w_scalar
                else:

                    @parameter
                    for i in range(simd_width):
                        var ti = t + i
                        var lane: Scalar[dt] = 0

                        @parameter
                        for k in range(W):
                            var src_t = ti + k - (W - 1)
                            var val: Scalar[dt]
                            if src_t < 0:

                                @parameter
                                if Self.has_initial_states:
                                    val = initial_states.load[1](
                                        IndexList[3](b, d, ti + k)
                                    )[0]
                                else:
                                    val = 0
                            elif src_t < L:
                                val = x.load[1](
                                    IndexList[3](b, d, src_t)
                                )[0]
                            else:
                                val = 0
                            lane += val * weight.load[1](
                                IndexList[2](d, k)
                            )[0]
                        acc[i] = lane

                @parameter
                if Self.has_bias:
                    var b_scalar = bias.load[1](IndexList[1](d))[0]
                    acc = acc + SIMD[dt, simd_width](b_scalar)

                @parameter
                if Self.activation == "silu":
                    acc = acc / (SIMD[dt, simd_width](1) + exp(-acc))

                return acc

            foreach[compute_out_cpu, target=target, simd_width=4](outp, ctx)

        # final_states: small, only fired when requested.
        @parameter
        @always_inline
        def compute_final[
            simd_width: Int
        ](idx: IndexList[3]) -> SIMD[dt, simd_width]:
            var L = x.dim_size(2)
            var b = idx[0]
            var d = idx[1]
            var s = idx[2]
            var combined_idx = L + s
            var val: Scalar[dt]
            if combined_idx < (W - 1):

                @parameter
                if Self.has_initial_states:
                    val = initial_states.load[1](
                        IndexList[3](b, d, combined_idx)
                    )[0]
                else:
                    val = 0
            else:
                val = x.load[1](
                    IndexList[3](b, d, combined_idx - (W - 1))
                )[0]
            return SIMD[dt, simd_width](val)

        comptime if Self.compute_final_states:
            foreach[compute_final, target=target, simd_width=1](
                final_states, ctx
            )
