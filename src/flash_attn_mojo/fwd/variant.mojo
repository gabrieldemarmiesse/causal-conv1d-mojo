"""Static variant entry point for flash_attn_fwd.

All comptime params are read from `-D` defines passed by Python
(`_jit.py`), so a single source file covers every config. Mirrors
causal-conv1d-mojo's `fwd/variant.mojo`.

Runtime args tuple (22 positionals) is built in `fwd/__init__.py`.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from launch import launch_fwd
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.float16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()


def flash_attn_fwd_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def flash_attn_fwd_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var batch_int: Int = Int(py=args[4])
    var seqlen_int: Int = Int(py=args[5])
    var nheads_int: Int = Int(py=args[6])
    var softmax_scale: Float32 = Float32(py=args[7])
    var q_b_stride: Int = Int(py=args[8])
    var q_l_stride: Int = Int(py=args[9])
    var q_h_stride: Int = Int(py=args[10])
    var k_b_stride: Int = Int(py=args[11])
    var k_l_stride: Int = Int(py=args[12])
    var k_h_stride: Int = Int(py=args[13])
    var v_b_stride: Int = Int(py=args[14])
    var v_l_stride: Int = Int(py=args[15])
    var v_h_stride: Int = Int(py=args[16])
    var o_b_stride: Int = Int(py=args[17])
    var o_l_stride: Int = Int(py=args[18])
    var o_h_stride: Int = Int(py=args[19])
    var stream_handle_addr: Int = Int(py=args[20])
    # args[21..23] (dtype, head_dim, use_external_stream) are all
    # comptime defines — read at module level via get_defined_*, so
    # they're skipped here. ctx_handle is appended by `call_fwd` as
    # the 25th positional (index 24).
    var ctx_handle_addr: Int = Int(py=args[24])

    if batch_int == 0 or seqlen_int == 0 or nheads_int == 0:
        return PythonObject(None)

    launch_fwd[
        DTYPE,
        HEAD_DIM,
        USE_EXTERNAL_STREAM,
    ](
        batch_int,
        seqlen_int,
        nheads_int,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        o_addr,
        q_b_stride,
        q_l_stride,
        q_h_stride,
        k_b_stride,
        k_l_stride,
        k_h_stride,
        v_b_stride,
        v_l_stride,
        v_h_stride,
        o_b_stride,
        o_l_stride,
        o_h_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[flash_attn_fwd_variant]("flash_attn_fwd_variant")
        m.def_py_function[flash_attn_fwd_acquire_ctx]("flash_attn_fwd_acquire_ctx")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
