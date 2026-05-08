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
"""

from std.os import abort
from std.math import ceildiv, exp
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext
from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
)


comptime kNThreads: Int = 128
comptime kNElts: Int = 4


fn _silu_f32(x: Float32) -> Float32:
    return x / (Float32(1) + exp(-x))


fn fwd_kernel[
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

    var compiled_func = ctx.compile_function[
        fwd_kernel[DType.float16, 4, True, "silu"],
        fwd_kernel[DType.float16, 4, True, "silu"],
    ]()

    stream.enqueue_function(
        compiled_func,
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
        grid_dim=(
            ceildiv(seqlen_int, kNThreads * kNElts),
            dim_int,
            batch_int,
        ),
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
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
