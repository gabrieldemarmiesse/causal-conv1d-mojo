"""Single-function dispatcher for `causal_conv1d_update_cpu` (split out of `causal_conv1d_native.mojo` so each entry
point compiles to its own .so and is imported lazily on
first call from the Python wrapper)."""

from std.itertools import product
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import update_kernel_cpu

comptime _BOOLS = [False, True]
comptime _WIDTHS = [2, 3, 4]


def causal_conv1d_update_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU single-step update. dtype + width are dispatched at runtime.

    Same arg layout as the GPU launcher minus the `cuda_stream_handle`
    (28 args instead of 29).
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
    var width_rt: Int = Int(py=args[23])
    var has_state_indices_rt: Bool = Int(py=args[24]) != 0
    var state_indices_addr: Int = Int(py=args[25])
    var is_circular_rt: Bool = Int(py=args[26]) != 0
    var cache_seqlens_addr: Int = Int(py=args[27])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    @parameter
    fn run[dtype: DType]() raises:
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
        fn dispatch[
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
            update_kernel_cpu[
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
            ](
                batch_int,
                dim_int,
                seqlen_int,
                state_len_int,
                x_tt.as_immut(),
                w_tt.as_immut(),
                b_ptr,
                state_tt,
                state_indices_ptr,
                cache_seqlens_ptr,
                o_tt,
            )

        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb, silu, hi, circ in product(
                _BOOLS, _BOOLS, _BOOLS, _BOOLS
            ):
                if (
                    hb == has_bias_rt
                    and silu == apply_silu_rt
                    and hi == has_state_indices_rt
                    and circ == is_circular_rt
                ):
                    dispatch[width, hb, silu, hi, circ]()

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
        m.def_py_function[causal_conv1d_update_cpu]("causal_conv1d_update_cpu")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
