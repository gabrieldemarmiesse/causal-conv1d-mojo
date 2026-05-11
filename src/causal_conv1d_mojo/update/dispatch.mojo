"""Single-function dispatcher for `causal_conv1d_update` (split out of `causal_conv1d_native.mojo` so each entry
point compiles to its own .so and is imported lazily on
first call from the Python wrapper)."""

from std.gpu.host import DeviceContext
from std.itertools import product
from std.math import ceildiv
from std.memory import OpaquePointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import kNThreadsUpdate, update_kernel

comptime _BOOLS = [False, True]
comptime _WIDTHS = [2, 3, 4]


def causal_conv1d_update(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """GPU single-step update. dtype + width are dispatched at runtime.

    Updates `conv_state` in place (shifts left by `seqlen` and writes the
    new x values at the tail; or, in circular mode, writes at the
    `cache_seqlens[b]` head with wrap) and emits the conv output.

    Python tuple positional args (29):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  conv_state_data_ptr  (int)
        4  output_data_ptr  (int)
        5  batch  (int)
        6  dim    (int)
        7  seqlen (int)
        8  state_len (int) — must be >= width-1
        9  x_batch_stride  (int)
        10 x_c_stride      (int)
        11 x_l_stride      (int)
        12 weight_c_stride (int)
        13 weight_w_stride (int)
        14 state_batch_stride  (int)
        15 state_c_stride      (int)
        16 state_l_stride      (int)
        17 out_batch_stride  (int)
        18 out_c_stride      (int)
        19 out_l_stride      (int)
        20 has_bias (int, 0 or 1)
        21 apply_silu (int, 0 or 1)
        22 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        23 cuda_stream_handle (int)
        24 width (int) — supported: 2, 3, 4
        25 has_state_indices (int, 0 or 1)
        26 state_indices_data_ptr (int, int32) — pass 0 if `has_state_indices=0`
        27 is_circular (int, 0 or 1)
        28 cache_seqlens_data_ptr (int, int32) — pass 0 if `is_circular=0`
    """
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var state_addr: Int = Int(py=args[3])
    var o_addr: Int = Int(py=args[4])

    var batch_int: Int = Int(py=args[5])
    var dim_int: Int = Int(py=args[6])
    var seqlen_int: Int = Int(py=args[7])
    var state_len_int: Int = Int(py=args[8])

    var x_b_stride: Int = Int(py=args[9])
    var x_c_stride: Int = Int(py=args[10])
    var x_l_stride: Int = Int(py=args[11])
    var w_c_stride: Int = Int(py=args[12])
    var w_w_stride: Int = Int(py=args[13])
    var state_b_stride: Int = Int(py=args[14])
    var state_c_stride: Int = Int(py=args[15])
    var state_l_stride: Int = Int(py=args[16])
    var o_b_stride: Int = Int(py=args[17])
    var o_c_stride: Int = Int(py=args[18])
    var o_l_stride: Int = Int(py=args[19])
    var has_bias_rt: Bool = Int(py=args[20]) != 0
    var apply_silu_rt: Bool = Int(py=args[21]) != 0
    var dtype_code: Int = Int(py=args[22])
    var stream_handle_addr: Int = Int(py=args[23])
    var width_rt: Int = Int(py=args[24])
    var has_state_indices_rt: Bool = Int(py=args[25]) != 0
    var state_indices_addr: Int = Int(py=args[26])
    var is_circular_rt: Bool = Int(py=args[27]) != 0
    var cache_seqlens_addr: Int = Int(py=args[28])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One thread per channel; kNThreadsUpdate channels per block.
    var grid = (
        batch_int,
        ceildiv(dim_int, kNThreadsUpdate),
    )

    @parameter
    def run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var state_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=state_addr
        )
        var state_indices_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=state_indices_addr
        )
        var cache_seqlens_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=cache_seqlens_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )

        @parameter
        def enqueue_update[
            width: Int,
            has_bias: Bool,
            apply_silu: Bool,
            has_state_indices: Bool,
            is_circular: Bool,
        ]() raises:
            var x_tt = TileTensor(
                x_ptr,
                Layout(
                    (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                    (Idx(x_b_stride), Idx(x_c_stride), Idx(x_l_stride)),
                ),
            )
            var w_tt = TileTensor(
                w_ptr,
                Layout(
                    (Idx(dim_int), Idx[width]()),
                    (Idx(w_c_stride), Idx(w_w_stride)),
                ),
            )
            # State's batch dim is a no-op when has_state_indices=True
            # (kernel trusts state_indices[b] verbatim, no bounds check),
            # so just pass `batch_int` — it's never read for the indexed
            # case anyway.
            var state_tt = TileTensor(
                state_ptr,
                Layout(
                    (Idx(batch_int), Idx(dim_int), Idx(state_len_int)),
                    (Idx(state_b_stride), Idx(state_c_stride), Idx(state_l_stride)),
                ),
            )
            var o_tt = TileTensor(
                o_ptr,
                Layout(
                    (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                    (Idx(o_b_stride), Idx(o_c_stride), Idx(o_l_stride)),
                ),
            )
            var compiled = ctx.compile_function[
                update_kernel[
                    dtype,
                    width,
                    has_bias,
                    apply_silu,
                    has_state_indices,
                    is_circular,
                    type_of(x_tt).LayoutType,
                    type_of(w_tt).LayoutType,
                    type_of(state_tt).LayoutType,
                    type_of(o_tt).LayoutType,
                ],
                update_kernel[
                    dtype,
                    width,
                    has_bias,
                    apply_silu,
                    has_state_indices,
                    is_circular,
                    type_of(x_tt).LayoutType,
                    type_of(w_tt).LayoutType,
                    type_of(state_tt).LayoutType,
                    type_of(o_tt).LayoutType,
                ],
            ]()
            stream.enqueue_function(
                compiled,
                seqlen_int,
                state_len_int,
                x_tt.as_immut(),
                w_tt.as_immut(),
                b_ptr,
                state_tt,
                state_indices_ptr,
                cache_seqlens_ptr,
                o_tt,
                grid_dim=grid,
                block_dim=(kNThreadsUpdate,),
            )

        @parameter
        def dispatch_w[width: Int]() raises:
            comptime for hb, silu, hi, circ in product(
                _BOOLS, _BOOLS, _BOOLS, _BOOLS
            ):
                if (
                    hb == has_bias_rt
                    and silu == apply_silu_rt
                    and hi == has_state_indices_rt
                    and circ == is_circular_rt
                ):
                    enqueue_update[width, hb, silu, hi, circ]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


@export
def PyInit_dispatch() -> PythonObject:
    try:
        var m = PythonModuleBuilder("dispatch")
        m.def_py_function[causal_conv1d_update]("causal_conv1d_update")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
