"""Direct Python -> Mojo extension for flash_attn_mojo, no MAX framework.

Built as a CPython extension via `mojo build --emit shared-lib`, same
pattern as `causal_conv1d_native.mojo`. `mojo.importer` triggers the
build on first import and caches the resulting `.so` under
`__mojocache__/`.

This file is the dispatcher (mirrors upstream's `flash_api.cpp`):
parses Python tuple args, builds a comptime dispatch tree on
`(dtype, headdim, causal)`, forwards to the kernel implementations.

Current entry points:
- `flash_attn_fwd_cpu` — naive CPU forward (MHA only, fp16 only,
  headdim=64 only; causal optional).

GPU forward, backward, and other features land in subsequent phases.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from flash_fwd_cpu import fwd_kernel_cpu


# Must match the dispatch in the Python wrapper.
# 0=fp16, 1=bf16, 2=fp32. (Phase 1.1 only accepts 0.)
comptime _DTYPES = [DType.float16, DType.bfloat16, DType.float32]


def flash_attn_fwd_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU forward for `flash_attn_func`.

    Python tuple positional args:
        0  q_data_ptr  (int)
        1  k_data_ptr  (int)
        2  v_data_ptr  (int)
        3  out_data_ptr  (int)
        4  batch  (int)
        5  seqlen_q  (int)
        6  seqlen_k  (int)
        7  nheads  (int)
        8  q_batch_stride  (int)
        9  q_seq_stride    (int)
        10 q_head_stride   (int)
        11 q_dim_stride    (int)
        12 k_batch_stride  (int)
        13 k_seq_stride    (int)
        14 k_head_stride   (int)
        15 k_dim_stride    (int)
        16 v_batch_stride  (int)
        17 v_seq_stride    (int)
        18 v_head_stride   (int)
        19 v_dim_stride    (int)
        20 out_batch_stride (int)
        21 out_seq_stride   (int)
        22 out_head_stride  (int)
        23 out_dim_stride   (int)
        24 softmax_scale (float)
        25 dtype_code  (int) — 0=fp16, 1=bf16, 2=fp32
        26 headdim     (int) — supported: 64
        27 causal      (int) — 0 = no mask, 1 = causal (bottom-right)
    """
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])

    var batch_int: Int = Int(py=args[4])
    var seqlen_q_int: Int = Int(py=args[5])
    var seqlen_k_int: Int = Int(py=args[6])
    var nheads_int: Int = Int(py=args[7])

    var q_b_stride: Int = Int(py=args[8])
    var q_s_stride: Int = Int(py=args[9])
    var q_h_stride: Int = Int(py=args[10])
    var q_d_stride: Int = Int(py=args[11])
    var k_b_stride: Int = Int(py=args[12])
    var k_s_stride: Int = Int(py=args[13])
    var k_h_stride: Int = Int(py=args[14])
    var k_d_stride: Int = Int(py=args[15])
    var v_b_stride: Int = Int(py=args[16])
    var v_s_stride: Int = Int(py=args[17])
    var v_h_stride: Int = Int(py=args[18])
    var v_d_stride: Int = Int(py=args[19])
    var o_b_stride: Int = Int(py=args[20])
    var o_s_stride: Int = Int(py=args[21])
    var o_h_stride: Int = Int(py=args[22])
    var o_d_stride: Int = Int(py=args[23])

    # Python passes softmax_scale as a Python float; convert via the
    # standard cast.
    var softmax_scale: Float32 = Float32(py=args[24])
    var dtype_code: Int = Int(py=args[25])
    var headdim_rt: Int = Int(py=args[26])
    var causal_rt: Int = Int(py=args[27])

    if batch_int == 0 or seqlen_q_int == 0 or nheads_int == 0:
        return PythonObject(None)

    @parameter
    fn run[dtype: DType, headdim: Int, causal: Bool]() raises:
        var q_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=q_addr
        )
        var k_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=k_addr
        )
        var v_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=v_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )
        fwd_kernel_cpu[dtype, headdim, causal](
            batch_int,
            seqlen_q_int,
            seqlen_k_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            q_b_stride,
            q_s_stride,
            q_h_stride,
            q_d_stride,
            k_b_stride,
            k_s_stride,
            k_h_stride,
            k_d_stride,
            v_b_stride,
            v_s_stride,
            v_h_stride,
            v_d_stride,
            o_b_stride,
            o_s_stride,
            o_h_stride,
            o_d_stride,
        )

    # Currently only fp16 + headdim=64. Other (dtype, headdim) combos
    # raise and the Python wrapper catches them earlier — this is a
    # defence-in-depth check.
    if dtype_code == 0 and headdim_rt == 64:
        if causal_rt != 0:
            run[DType.float16, 64, True]()
        else:
            run[DType.float16, 64, False]()
    else:
        raise Error("currently only supports dtype=fp16 and headdim=64")

    return PythonObject(None)


@export
def PyInit_flash_attn_native() -> PythonObject:
    try:
        var m = PythonModuleBuilder("flash_attn_native")
        m.def_py_function[flash_attn_fwd_cpu]("flash_attn_fwd_cpu")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
