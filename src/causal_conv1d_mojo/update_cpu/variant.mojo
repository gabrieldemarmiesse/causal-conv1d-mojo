"""Static per-config variant entry point for causal_conv1d_update_cpu.

Runtime args tuple (22 positionals) is built in
``update_cpu/__init__.py``; comptime values come from `-D` defines.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int
from layout import TileTensor, Idx
from layout.tile_layout import Layout

from kernel import update_kernel_cpu

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.float32]()
comptime WIDTH: Int = get_defined_int["WIDTH"]()
comptime HAS_BIAS: Bool = get_defined_bool["HAS_BIAS"]()
comptime APPLY_SILU: Bool = get_defined_bool["APPLY_SILU"]()
comptime HAS_STATE_INDICES: Bool = get_defined_bool["HAS_STATE_INDICES"]()
comptime IS_CIRCULAR: Bool = get_defined_bool["IS_CIRCULAR"]()


def causal_conv1d_update_cpu_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
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
    var state_indices_addr: Int = Int(py=args[20])
    var cache_seqlens_addr: Int = Int(py=args[21])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    var x_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var w_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var b_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=b_addr
    )
    var state_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=state_addr
    )
    var state_indices_ptr = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=state_indices_addr
    )
    var cache_seqlens_ptr = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=cache_seqlens_addr
    )
    var o_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

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
            (Idx(dim_int), Idx[WIDTH]()),
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
        DTYPE,
        WIDTH,
        HAS_BIAS,
        APPLY_SILU,
        HAS_STATE_INDICES,
        IS_CIRCULAR,
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
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_update_cpu_variant](
            "causal_conv1d_update_cpu_variant"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
