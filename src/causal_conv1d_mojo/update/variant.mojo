"""Static variant entry point for causal_conv1d_update.

Comptime params come from `-D` defines set by `_jit.py`.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
)

from launch import launch_update
from _ctx import acquire_ctx_handle

comptime DTYPE = get_defined_dtype["DTYPE", DType.float32]()
comptime WIDTH = get_defined_int["WIDTH"]()
comptime HAS_BIAS = get_defined_bool["HAS_BIAS"]()
comptime APPLY_SILU = get_defined_bool["APPLY_SILU"]()
comptime HAS_STATE_INDICES = get_defined_bool["HAS_STATE_INDICES"]()
comptime IS_CIRCULAR = get_defined_bool["IS_CIRCULAR"]()
comptime USE_EXTERNAL_STREAM = get_defined_bool["USE_EXTERNAL_STREAM"]()
# When non-empty, the path `compile_function` dumps this variant's PTX to
# (set via the `DUMP_ASSEMBLY_INTO` env var, threaded through `_jit_common`
# as a `-D` define). Empty => no dump. `%` expands to the module name.
comptime DUMP_ASSEMBLY_INTO = get_defined_string["DUMP_ASSEMBLY_INTO", ""]()


def causal_conv1d_update_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext.

    Called once per variant from the Python side; the returned address
    is reused for every subsequent `causal_conv1d_update_variant` call.
    """
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def causal_conv1d_update_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var w_addr = Int(py=args[1])
    var b_addr = Int(py=args[2])
    var state_addr = Int(py=args[3])
    var o_addr = Int(py=args[4])
    var batch_int = Int(py=args[5])
    var dim_int = Int(py=args[6])
    var seqlen_int = Int(py=args[7])
    var state_len_int = Int(py=args[8])
    var x_b_stride = Int32(py=args[9])
    var x_c_stride = Int32(py=args[10])
    var x_l_stride = Int32(py=args[11])
    var w_c_stride = Int32(py=args[12])
    var w_w_stride = Int32(py=args[13])
    var state_b_stride = Int32(py=args[14])
    var state_c_stride = Int32(py=args[15])
    var state_l_stride = Int32(py=args[16])
    var o_b_stride = Int32(py=args[17])
    var o_c_stride = Int32(py=args[18])
    var o_l_stride = Int32(py=args[19])
    var stream_handle_addr = Int(py=args[23])
    var state_indices_addr = Int(py=args[26])
    var cache_seqlens_addr = Int(py=args[28])
    # args[29] is `use_external_stream` (already a comptime define);
    # ctx_handle is appended as args[30] by `call_update`.
    var ctx_handle_addr = Int(py=args[30])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    launch_update[
        DTYPE,
        WIDTH,
        HAS_BIAS,
        APPLY_SILU,
        HAS_STATE_INDICES,
        IS_CIRCULAR,
        USE_EXTERNAL_STREAM,
        DUMP_ASSEMBLY_INTO,
    ](
        batch_int,
        dim_int,
        seqlen_int,
        state_len_int,
        x_addr,
        w_addr,
        b_addr,
        state_addr,
        o_addr,
        state_indices_addr,
        cache_seqlens_addr,
        x_b_stride,
        x_c_stride,
        x_l_stride,
        w_c_stride,
        w_w_stride,
        state_b_stride,
        state_c_stride,
        state_l_stride,
        o_b_stride,
        o_c_stride,
        o_l_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_update_variant](
            "causal_conv1d_update_variant"
        )
        m.def_py_function[causal_conv1d_update_acquire_ctx](
            "causal_conv1d_update_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
