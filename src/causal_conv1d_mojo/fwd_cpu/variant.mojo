"""Static per-config variant entry point for causal_conv1d_fwd_cpu.

Replaces the old AOT comptime-sweep `dispatch.mojo`. All comptime
params (dtype, width, has_bias, …) come from `-D` defines set by
`_jit.py`, so we compile only the variant the caller actually needs.
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

    Args tuple (28 positionals — same shape as the old AOT dispatcher
    for backwards compat; the comptime fields are ignored at runtime):
      0  x_data_ptr (int)
      1  weight_data_ptr (int)
      2  bias_data_ptr (int) — pass 0 if `HAS_BIAS=false`
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
      15 has_bias (int)               — comptime, unused
      16 apply_silu (int)             — comptime, unused
      17 dtype_code (int)             — comptime, unused
      18 has_seq_idx (int)            — comptime, unused
      19 seq_idx_data_ptr (int)
      20 seq_idx_b_stride (int)
      21 seq_idx_l_stride (int)
      22 width (int)                  — comptime, unused
      23 has_initial_states (int)     — comptime, unused
      24 initial_states_data_ptr (int)
      25 initial_states_b_stride (int)
      26 initial_states_c_stride (int)
      27 initial_states_l_stride (int)
    """
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
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
    var seq_idx_addr: Int = Int(py=args[19])
    var seq_idx_b_stride: Int = Int(py=args[20])
    var seq_idx_l_stride: Int = Int(py=args[21])
    var initial_states_addr: Int = Int(py=args[24])
    var initial_states_b_stride: Int = Int(py=args[25])
    var initial_states_c_stride: Int = Int(py=args[26])
    var initial_states_l_stride: Int = Int(py=args[27])

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
    var o_tt = TileTensor(
        o_ptr,
        Layout(
            (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
            (Idx(o_b_stride), Idx(o_c_stride), Idx(o_l_stride)),
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
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_tt.as_immut(),
        w_tt.as_immut(),
        b_ptr,
        seq_idx_ptr,
        initial_states_ptr,
        o_tt,
        seq_idx_b_stride,
        seq_idx_l_stride,
        initial_states_b_stride,
        initial_states_c_stride,
        initial_states_l_stride,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_fwd_cpu_variant](
            "causal_conv1d_fwd_cpu_variant"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
