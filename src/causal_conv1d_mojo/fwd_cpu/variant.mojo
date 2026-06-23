"""Static per-config variant entry point for causal_conv1d_fwd_cpu.

Replaces the old AOT comptime-sweep `dispatch.mojo`. All comptime
params (dtype, width, has_bias, …) come from `-D` defines set by
`_jit.py`, so we compile only the variant the caller actually needs.

Runtime args tuple (22 positionals) is built in
``fwd_cpu/__init__.py``; it carries only runtime-varying values —
comptime values are baked into the `.so` via `-D`.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int
from layout import TileTensor, Idx
from layout.tile_layout import Layout

from kernel import fwd_kernel_cpu

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.float32]()
comptime WIDTH: Int = get_defined_int["WIDTH"]()
comptime HAS_BIAS: Bool = get_defined_bool["HAS_BIAS"]()
comptime HAS_SEQ_IDX: Bool = get_defined_bool["HAS_SEQ_IDX"]()
comptime HAS_INITIAL_STATES: Bool = get_defined_bool["HAS_INITIAL_STATES"]()
comptime APPLY_SILU: Bool = get_defined_bool["APPLY_SILU"]()


def causal_conv1d_fwd_cpu_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU forward, one specialised variant.

    Runtime args (22 positionals; built in `fwd_cpu/__init__.py`):
      0  x_data_ptr (int)
      1  weight_data_ptr (int)
      2  bias_data_ptr (int) — 0 if `HAS_BIAS=false`
      3  output_data_ptr (int)
      4  batch  (int)
      5  dim    (int)
      6  seqlen (int)
      7  x_b_stride  (int)
      8  x_c_stride  (int)
      9  x_l_stride  (int)
      10 w_c_stride  (int)
      11 w_w_stride  (int)
      12 o_b_stride  (int)
      13 o_c_stride  (int)
      14 o_l_stride  (int)
      15 seq_idx_data_ptr (int)
      16 seq_idx_b_stride (int)
      17 seq_idx_l_stride (int)
      18 initial_states_data_ptr (int)
      19 initial_states_b_stride (int)
      20 initial_states_c_stride (int)
      21 initial_states_l_stride (int)
    """
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var batch_int: Int = Int(py=args[4])
    var dim_int: Int = Int(py=args[5])
    var seqlen_int: Int = Int(py=args[6])
    var x_b_stride: UInt32 = UInt32(py=args[7])
    var x_c_stride: UInt32 = UInt32(py=args[8])
    var x_l_stride: UInt32 = UInt32(py=args[9])
    var w_c_stride: UInt32 = UInt32(py=args[10])
    var w_w_stride: UInt32 = UInt32(py=args[11])
    var o_b_stride: UInt32 = UInt32(py=args[12])
    var o_c_stride: UInt32 = UInt32(py=args[13])
    var o_l_stride: UInt32 = UInt32(py=args[14])
    var seq_idx_addr: Int = Int(py=args[15])
    var seq_idx_b_stride: UInt32 = UInt32(py=args[16])
    var seq_idx_l_stride: UInt32 = UInt32(py=args[17])
    var initial_states_addr: Int = Int(py=args[18])
    var initial_states_b_stride: UInt32 = UInt32(py=args[19])
    var initial_states_c_stride: UInt32 = UInt32(py=args[20])
    var initial_states_l_stride: UInt32 = UInt32(py=args[21])

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
    var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
        unsafe_from_address=seq_idx_addr
    )
    var initial_states_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=initial_states_addr
    )
    var o_ptr = UnsafePointer[Scalar[DTYPE], MutAnyOrigin](
        unsafe_from_address=o_addr
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
    var o_tt = TileTensor(
        o_ptr,
        Layout(
            (batch_int, dim_int, seqlen_int),
            (o_b_stride, o_c_stride, o_l_stride),
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
    fwd_kernel_cpu[
        DTYPE,
        WIDTH,
        HAS_BIAS,
        HAS_SEQ_IDX,
        HAS_INITIAL_STATES,
        APPLY_SILU,
        type_of(x_tt).LayoutType,
        type_of(w_tt).LayoutType,
        type_of(o_tt).LayoutType,
        type_of(seq_idx_tt).LayoutType,
        type_of(initial_states_tt).LayoutType,
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_tt.as_immut(),
        w_tt.as_immut(),
        b_ptr,
        seq_idx_tt.as_immut(),
        initial_states_tt.as_immut(),
        o_tt,
    )
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_fwd_cpu_variant](
            "causal_conv1d_fwd_cpu_variant"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
