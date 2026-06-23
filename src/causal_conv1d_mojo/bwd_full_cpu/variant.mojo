"""Static per-config variant entry point for causal_conv1d_bwd_full_cpu.

Runtime args tuple (32 positionals) is built in
``bwd_full_cpu/__init__.py``; comptime values come from `-D` defines.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int
from layout import TileTensor, Idx
from layout.tile_layout import Layout

from kernel import bwd_kernel_cpu

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.float32]()
comptime WIDTH: Int = get_defined_int["WIDTH"]()
comptime HAS_BIAS: Bool = get_defined_bool["HAS_BIAS"]()
comptime HAS_SEQ_IDX: Bool = get_defined_bool["HAS_SEQ_IDX"]()
comptime HAS_INITIAL_STATES: Bool = get_defined_bool["HAS_INITIAL_STATES"]()
comptime APPLY_SILU: Bool = get_defined_bool["APPLY_SILU"]()


def causal_conv1d_bwd_full_cpu_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var dout_addr: Int = Int(py=args[3])
    var dx_addr: Int = Int(py=args[4])
    var dweight_acc_addr: Int = Int(py=args[5])
    var dbias_acc_addr: Int = Int(py=args[6])
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
    var seq_idx_addr: Int = Int(py=args[21])
    var seq_idx_b_stride: Int = Int(py=args[22])
    var seq_idx_l_stride: Int = Int(py=args[23])
    var initial_states_addr: Int = Int(py=args[24])
    var initial_states_b_stride: Int = Int(py=args[25])
    var initial_states_c_stride: Int = Int(py=args[26])
    var initial_states_l_stride: Int = Int(py=args[27])
    var dinitial_states_addr: Int = Int(py=args[28])
    var dinitial_states_b_stride: Int = Int(py=args[29])
    var dinitial_states_c_stride: Int = Int(py=args[30])
    var dinitial_states_l_stride: Int = Int(py=args[31])

    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
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
    var dout_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=dout_addr
    )
    var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=seq_idx_addr
    )
    var initial_states_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=initial_states_addr
    )
    var dx_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=dx_addr
    )
    var dinitial_states_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=dinitial_states_addr
    )
    var dweight_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dweight_acc_addr
    )
    var dbias_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dbias_acc_addr
    )

    var x_tt = TileTensor(
        x_ptr,
        Layout(
            (batch_int, dim_int, seqlen_int),
            (x_b_stride, x_c_stride, x_l_stride),
        ),
    )
    var w_tt = TileTensor(
        w_ptr,
        Layout(
            (dim_int, Idx[WIDTH]),
            (w_c_stride, w_w_stride),
        ),
    )
    var dout_tt = TileTensor(
        dout_ptr,
        Layout(
            (batch_int, dim_int, seqlen_int),
            (dout_b_stride, dout_c_stride, dout_l_stride),
        ),
    )
    var dx_tt = TileTensor(
        dx_ptr,
        Layout(
            (batch_int, dim_int, seqlen_int),
            (dx_b_stride, dx_c_stride, dx_l_stride),
        ),
    )
    var seq_idx_tt = TileTensor(
        seq_idx_ptr,
        Layout(
            (batch_int, seqlen_int),
            (seq_idx_b_stride, seq_idx_l_stride),
        ),
    )
    var initial_states_tt = TileTensor(
        initial_states_ptr,
        Layout(
            (batch_int, dim_int, Idx[WIDTH - 1]),
            (
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
            ),
        ),
    )
    var dinitial_states_tt = TileTensor(
        dinitial_states_ptr,
        Layout(
            (batch_int, dim_int, Idx[WIDTH - 1]),
            (
                dinitial_states_b_stride,
                dinitial_states_c_stride,
                dinitial_states_l_stride,
            ),
        ),
    )
    bwd_kernel_cpu[
        DTYPE,
        WIDTH,
        HAS_BIAS,
        HAS_SEQ_IDX,
        HAS_INITIAL_STATES,
        APPLY_SILU,
        type_of(x_tt).LayoutType,
        type_of(w_tt).LayoutType,
        type_of(dout_tt).LayoutType,
        type_of(dx_tt).LayoutType,
        type_of(seq_idx_tt).LayoutType,
        type_of(initial_states_tt).LayoutType,
        type_of(dinitial_states_tt).LayoutType,
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_tt.as_immut(),
        w_tt.as_immut(),
        b_ptr,
        dout_tt.as_immut(),
        seq_idx_tt.as_immut(),
        initial_states_tt.as_immut(),
        dx_tt,
        dweight_acc_ptr,
        dbias_acc_ptr,
        dinitial_states_tt,
    )
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_bwd_full_cpu_variant](
            "causal_conv1d_bwd_full_cpu_variant"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
