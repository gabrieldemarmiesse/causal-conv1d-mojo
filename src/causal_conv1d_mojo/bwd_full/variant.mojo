"""Static variant entry point for causal_conv1d_bwd_full.

Mirrors `fwd/variant.mojo`: comptime params come from `-D` defines
set by `_jit.py`. See that file for the full set of dimensions.
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

from launch import launch_bwd_full
from _ctx import acquire_ctx_handle

comptime DTYPE = get_defined_dtype["DTYPE", DType.float32]()
comptime N_ELTS = get_defined_int["N_ELTS"]()
comptime WIDTH = get_defined_int["WIDTH"]()
comptime HAS_BIAS = get_defined_bool["HAS_BIAS"]()
comptime HAS_SEQ_IDX = get_defined_bool["HAS_SEQ_IDX"]()
comptime HAS_INITIAL_STATES = get_defined_bool["HAS_INITIAL_STATES"]()
comptime APPLY_SILU = get_defined_bool["APPLY_SILU"]()
comptime CONTIG_INNER = get_defined_bool["CONTIG_INNER"]()
comptime ALIGNED_SEQ = get_defined_bool["ALIGNED_SEQ"]()
comptime USE_EXTERNAL_STREAM = get_defined_bool["USE_EXTERNAL_STREAM"]()
# When non-empty, the path `compile_function` dumps this variant's PTX to
# (set via the `DUMP_ASSEMBLY_INTO` env var, threaded through `_jit_common`
# as a `-D` define). Empty => no dump. `%` expands to the module name.
comptime DUMP_ASSEMBLY_INTO = get_defined_string["DUMP_ASSEMBLY_INTO", ""]()


def causal_conv1d_bwd_full_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext.

    Called once per variant from the Python side; the returned address
    is reused for every subsequent `causal_conv1d_bwd_full_variant`
    call to avoid `hipStreamCreate`/`Destroy` per launch.
    """
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def causal_conv1d_bwd_full_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var w_addr = Int(py=args[1])
    var b_addr = Int(py=args[2])
    var dout_addr = Int(py=args[3])
    var dx_addr = Int(py=args[4])
    var dweight_acc_addr = Int(py=args[5])
    var dbias_acc_addr = Int(py=args[6])
    var batch_int = Int(py=args[7])
    var dim_int = Int(py=args[8])
    var seqlen_int = Int(py=args[9])
    var x_b_stride = UInt32(py=args[10])
    var x_c_stride = UInt32(py=args[11])
    var x_l_stride = UInt32(py=args[12])
    var w_c_stride = UInt32(py=args[13])
    var w_w_stride = UInt32(py=args[14])
    var dout_b_stride = UInt32(py=args[15])
    var dout_c_stride = UInt32(py=args[16])
    var dout_l_stride = UInt32(py=args[17])
    var dx_b_stride = UInt32(py=args[18])
    var dx_c_stride = UInt32(py=args[19])
    var dx_l_stride = UInt32(py=args[20])
    var stream_handle_addr = Int(py=args[24])
    var seq_idx_addr = Int(py=args[27])
    var seq_idx_b_stride = UInt32(py=args[28])
    var seq_idx_l_stride = UInt32(py=args[29])
    var initial_states_addr = Int(py=args[31])
    var initial_states_b_stride = UInt32(py=args[32])
    var initial_states_c_stride = UInt32(py=args[33])
    var initial_states_l_stride = UInt32(py=args[34])
    var dinitial_states_addr = Int(py=args[35])
    var dinitial_states_b_stride = UInt32(py=args[36])
    var dinitial_states_c_stride = UInt32(py=args[37])
    var dinitial_states_l_stride = UInt32(py=args[38])
    # args[39] is `use_external_stream` (already a comptime define);
    # ctx_handle is appended as args[40] by `call_bwd_full`.
    var ctx_handle_addr = Int(py=args[40])

    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
        return PythonObject(None)

    launch_bwd_full[
        DTYPE,
        N_ELTS,
        WIDTH,
        HAS_BIAS,
        HAS_SEQ_IDX,
        HAS_INITIAL_STATES,
        APPLY_SILU,
        CONTIG_INNER,
        ALIGNED_SEQ,
        USE_EXTERNAL_STREAM,
        DUMP_ASSEMBLY_INTO,
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_addr,
        w_addr,
        b_addr,
        dout_addr,
        dx_addr,
        dweight_acc_addr,
        dbias_acc_addr,
        seq_idx_addr,
        initial_states_addr,
        dinitial_states_addr,
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
        seq_idx_b_stride,
        seq_idx_l_stride,
        initial_states_b_stride,
        initial_states_c_stride,
        initial_states_l_stride,
        dinitial_states_b_stride,
        dinitial_states_c_stride,
        dinitial_states_l_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_bwd_full_variant](
            "causal_conv1d_bwd_full_variant"
        )
        m.def_py_function[causal_conv1d_bwd_full_acquire_ctx](
            "causal_conv1d_bwd_full_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
