"""Static variant entry point for causal_conv1d_fwd.

All comptime params are read from `-D` defines passed by Python
(`_jit.py`), so a single source file covers every config. Compared
to the previous f-string codegen, this keeps the Mojo source out of
Python literals and lets `mojo build` handle the parameter wiring.

The Python wrapper:
- Computes the config tuple (dtype, width, has_bias, …).
- Picks a stable, human-readable ``mod_name`` from the config (used
  only for the cache directory + the loaded module name).
- Calls `mojo build variant.mojo -D DTYPE=float16 -D WIDTH=4 …` and
  caches the resulting `.so` content-addressed.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from launch import launch_fwd
from _ctx import acquire_ctx_handle

comptime DTYPE = get_defined_dtype["DTYPE", DType.float32]()
comptime WIDTH = get_defined_int["WIDTH"]()
comptime HAS_BIAS = get_defined_bool["HAS_BIAS"]()
comptime HAS_SEQ_IDX = get_defined_bool["HAS_SEQ_IDX"]()
comptime HAS_INITIAL_STATES = get_defined_bool["HAS_INITIAL_STATES"]()
comptime APPLY_SILU = get_defined_bool["APPLY_SILU"]()
comptime CONTIG_INNER = get_defined_bool["CONTIG_INNER"]()
comptime ALIGNED_SEQ = get_defined_bool["ALIGNED_SEQ"]()
comptime VEC_ALIGNED = get_defined_bool["VEC_ALIGNED"]()
comptime USE_EXTERNAL_STREAM = get_defined_bool["USE_EXTERNAL_STREAM"]()


def causal_conv1d_fwd_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext.

    Called once per variant from the Python side; the returned address
    is reused for every subsequent `causal_conv1d_fwd_variant` call.
    """
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def causal_conv1d_fwd_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var w_addr = Int(py=args[1])
    var b_addr = Int(py=args[2])
    var o_addr = Int(py=args[3])
    var batch_int = Int(py=args[4])
    var dim_int = Int(py=args[5])
    var seqlen_int = Int(py=args[6])
    var x_b_stride = UInt32(py=args[7])
    var x_c_stride = UInt32(py=args[8])
    var x_l_stride = UInt32(py=args[9])
    var w_c_stride = UInt32(py=args[10])
    var w_w_stride = UInt32(py=args[11])
    var o_b_stride = UInt32(py=args[12])
    var o_c_stride = UInt32(py=args[13])
    var o_l_stride = UInt32(py=args[14])
    var stream_handle_addr = Int(py=args[18])
    var seq_idx_addr = Int(py=args[20])
    var seq_idx_b_stride = UInt32(py=args[21])
    var seq_idx_l_stride = UInt32(py=args[22])
    var initial_states_addr = Int(py=args[25])
    var initial_states_b_stride = UInt32(py=args[26])
    var initial_states_c_stride = UInt32(py=args[27])
    var initial_states_l_stride = UInt32(py=args[28])
    # args[29] is `use_external_stream` (already a comptime define).
    # ctx_handle is appended as the 30th positional by `call_fwd`.
    var ctx_handle_addr = Int(py=args[30])

    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
        return PythonObject(None)

    launch_fwd[
        DTYPE,
        WIDTH,
        HAS_BIAS,
        HAS_SEQ_IDX,
        HAS_INITIAL_STATES,
        APPLY_SILU,
        CONTIG_INNER,
        ALIGNED_SEQ,
        VEC_ALIGNED,
        USE_EXTERNAL_STREAM,
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_addr,
        w_addr,
        b_addr,
        o_addr,
        seq_idx_addr,
        initial_states_addr,
        x_b_stride,
        x_c_stride,
        x_l_stride,
        w_c_stride,
        w_w_stride,
        o_b_stride,
        o_c_stride,
        o_l_stride,
        seq_idx_b_stride,
        seq_idx_l_stride,
        initial_states_b_stride,
        initial_states_c_stride,
        initial_states_l_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_fwd_variant]("causal_conv1d_fwd_variant")
        m.def_py_function[causal_conv1d_fwd_acquire_ctx]("causal_conv1d_fwd_acquire_ctx")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
