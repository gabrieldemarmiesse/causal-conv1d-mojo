"""Direct Python -> Mojo extension for flash_attn_mojo, no MAX framework.

Built as a CPython extension via `mojo build --emit shared-lib`, same
pattern as `causal_conv1d_native.mojo`. `mojo.importer` triggers the
build on first import and caches the resulting `.so` under
`__mojocache__/`.

This file is the dispatcher (mirrors upstream's `flash_api.cpp`):
parses Python tuple args, builds a comptime dispatch tree on
`(dtype, headdim, causal)`, forwards to the kernel implementations.

Current entry points:
- `flash_attn_fwd_cpu` — naive CPU forward (MHA / MQA / GQA, fp16,
  headdim ∈ {64, 96, 128}; causal optional).
- `flash_attn_bwd_cpu` — matching CPU backward.

GPU forward / backward and other features land in subsequent phases.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from flash_fwd_cpu import fwd_kernel_cpu
from flash_bwd_cpu import bwd_kernel_cpu


# Must match the dispatch in the Python wrapper. Order is fixed
# so the integer code from Python indexes directly into this list.
# 0=fp16, 1=bf16, 2=fp32.
alias _DTYPES = [DType.float16, DType.bfloat16, DType.float32]
alias _HEADDIMS = [64, 96, 128]


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
        4  lse_data_ptr (int) — fp32 (B, H_q, S_q), contiguous
        5  batch  (int)
        6  seqlen_q  (int)
        7  seqlen_k  (int)
        8  nheads_q  (int)
        9  nheads_kv (int) — must divide nheads_q (MQA/GQA)
        10 q_batch_stride  (int)
        11 q_seq_stride    (int)
        12 q_head_stride   (int)
        13 q_dim_stride    (int)
        14 k_batch_stride  (int)
        15 k_seq_stride    (int)
        16 k_head_stride   (int)
        17 k_dim_stride    (int)
        18 v_batch_stride  (int)
        19 v_seq_stride    (int)
        20 v_head_stride   (int)
        21 v_dim_stride    (int)
        22 out_batch_stride (int)
        23 out_seq_stride   (int)
        24 out_head_stride  (int)
        25 out_dim_stride   (int)
        26 softmax_scale (float)
        27 dtype_code  (int) — 0=fp16, 1=bf16, 2=fp32
        28 headdim     (int) — supported: 64, 96, 128
        29 causal      (int) — 0 = no mask, 1 = causal (bottom-right)
    """
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var lse_addr: Int = Int(py=args[4])

    var batch_int: Int = Int(py=args[5])
    var seqlen_q_int: Int = Int(py=args[6])
    var seqlen_k_int: Int = Int(py=args[7])
    var nheads_q_int: Int = Int(py=args[8])
    var nheads_kv_int: Int = Int(py=args[9])

    var q_b_stride: Int = Int(py=args[10])
    var q_s_stride: Int = Int(py=args[11])
    var q_h_stride: Int = Int(py=args[12])
    var q_d_stride: Int = Int(py=args[13])
    var k_b_stride: Int = Int(py=args[14])
    var k_s_stride: Int = Int(py=args[15])
    var k_h_stride: Int = Int(py=args[16])
    var k_d_stride: Int = Int(py=args[17])
    var v_b_stride: Int = Int(py=args[18])
    var v_s_stride: Int = Int(py=args[19])
    var v_h_stride: Int = Int(py=args[20])
    var v_d_stride: Int = Int(py=args[21])
    var o_b_stride: Int = Int(py=args[22])
    var o_s_stride: Int = Int(py=args[23])
    var o_h_stride: Int = Int(py=args[24])
    var o_d_stride: Int = Int(py=args[25])

    # Python passes softmax_scale as a Python float; convert via the
    # standard cast.
    var softmax_scale: Float32 = Float32(py=args[26])
    var dtype_code: Int = Int(py=args[27])
    var headdim_rt: Int = Int(py=args[28])
    var causal_rt: Int = Int(py=args[29])

    if batch_int == 0 or seqlen_q_int == 0 or nheads_q_int == 0:
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
        var lse_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=lse_addr
        )
        fwd_kernel_cpu[dtype, headdim, causal](
            batch_int,
            seqlen_q_int,
            seqlen_k_int,
            nheads_q_int,
            nheads_kv_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            lse_ptr,
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

    # Comptime expansion: 3 dtypes × 3 headdims × 2 causal = 18 leaves.
    # The Python wrapper validates these earlier — this is defence in depth.
    alias N_DTYPES = len(_DTYPES)
    alias N_HEADDIMS = len(_HEADDIMS)
    if dtype_code < 0 or dtype_code >= N_DTYPES:
        raise Error("invalid dtype_code")
    var dispatched: Bool = False

    @parameter
    for dt_idx in range(N_DTYPES):
        alias dt = _DTYPES[dt_idx]

        @parameter
        for hd_idx in range(N_HEADDIMS):
            alias hd = _HEADDIMS[hd_idx]
            if dtype_code == dt_idx and headdim_rt == hd:
                if causal_rt != 0:
                    run[dt, hd, True]()
                else:
                    run[dt, hd, False]()
                dispatched = True

    if not dispatched:
        raise Error(
            "unsupported (dtype, headdim) — dtype ∈ {fp16, bf16, fp32},"
            " headdim ∈ {64, 96, 128}"
        )

    return PythonObject(None)


def flash_attn_bwd_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU backward for `flash_attn_func`.

    Python tuple positional args:
        0  q_data_ptr  (int)
        1  k_data_ptr  (int)
        2  v_data_ptr  (int)
        3  out_data_ptr (int)
        4  dout_data_ptr (int)
        5  lse_data_ptr (int) — fp32 (B, H_q, S_q), contiguous
        6  dq_data_ptr (int)
        7  dk_data_ptr (int)
        8  dv_data_ptr (int)
        9  batch  (int)
        10 seqlen_q  (int)
        11 seqlen_k  (int)
        12 nheads_q  (int)
        13 nheads_kv (int)
        14 q_b/s/h/d  strides     (4 ints, args 14..17)
        18 k_b/s/h/d  strides     (args 18..21)
        22 v_b/s/h/d  strides     (args 22..25)
        26 out_b/s/h/d  strides   (args 26..29)
        30 dout_b/s/h/d strides   (args 30..33)
        34 dq_b/s/h/d strides     (args 34..37)
        38 dk_b/s/h/d strides     (args 38..41)
        42 dv_b/s/h/d strides     (args 42..45)
        46 softmax_scale (float)
        47 dtype_code (int)
        48 headdim    (int)
        49 causal     (int)
    """
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var do_addr: Int = Int(py=args[4])
    var lse_addr: Int = Int(py=args[5])
    var dq_addr: Int = Int(py=args[6])
    var dk_addr: Int = Int(py=args[7])
    var dv_addr: Int = Int(py=args[8])

    var batch_int: Int = Int(py=args[9])
    var seqlen_q_int: Int = Int(py=args[10])
    var seqlen_k_int: Int = Int(py=args[11])
    var nheads_q_int: Int = Int(py=args[12])
    var nheads_kv_int: Int = Int(py=args[13])

    var q_b_stride: Int = Int(py=args[14])
    var q_s_stride: Int = Int(py=args[15])
    var q_h_stride: Int = Int(py=args[16])
    var q_d_stride: Int = Int(py=args[17])
    var k_b_stride: Int = Int(py=args[18])
    var k_s_stride: Int = Int(py=args[19])
    var k_h_stride: Int = Int(py=args[20])
    var k_d_stride: Int = Int(py=args[21])
    var v_b_stride: Int = Int(py=args[22])
    var v_s_stride: Int = Int(py=args[23])
    var v_h_stride: Int = Int(py=args[24])
    var v_d_stride: Int = Int(py=args[25])
    var o_b_stride: Int = Int(py=args[26])
    var o_s_stride: Int = Int(py=args[27])
    var o_h_stride: Int = Int(py=args[28])
    var o_d_stride: Int = Int(py=args[29])
    var do_b_stride: Int = Int(py=args[30])
    var do_s_stride: Int = Int(py=args[31])
    var do_h_stride: Int = Int(py=args[32])
    var do_d_stride: Int = Int(py=args[33])
    var dq_b_stride: Int = Int(py=args[34])
    var dq_s_stride: Int = Int(py=args[35])
    var dq_h_stride: Int = Int(py=args[36])
    var dq_d_stride: Int = Int(py=args[37])
    var dk_b_stride: Int = Int(py=args[38])
    var dk_s_stride: Int = Int(py=args[39])
    var dk_h_stride: Int = Int(py=args[40])
    var dk_d_stride: Int = Int(py=args[41])
    var dv_b_stride: Int = Int(py=args[42])
    var dv_s_stride: Int = Int(py=args[43])
    var dv_h_stride: Int = Int(py=args[44])
    var dv_d_stride: Int = Int(py=args[45])

    var softmax_scale: Float32 = Float32(py=args[46])
    var dtype_code: Int = Int(py=args[47])
    var headdim_rt: Int = Int(py=args[48])
    var causal_rt: Int = Int(py=args[49])

    if batch_int == 0 or seqlen_q_int == 0 or nheads_q_int == 0:
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
        var do_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=do_addr
        )
        var lse_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=lse_addr
        )
        var dq_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dq_addr
        )
        var dk_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dk_addr
        )
        var dv_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dv_addr
        )
        bwd_kernel_cpu[dtype, headdim, causal](
            batch_int,
            seqlen_q_int,
            seqlen_k_int,
            nheads_q_int,
            nheads_kv_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            do_ptr,
            lse_ptr,
            dq_ptr,
            dk_ptr,
            dv_ptr,
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
            do_b_stride,
            do_s_stride,
            do_h_stride,
            do_d_stride,
            dq_b_stride,
            dq_s_stride,
            dq_h_stride,
            dq_d_stride,
            dk_b_stride,
            dk_s_stride,
            dk_h_stride,
            dk_d_stride,
            dv_b_stride,
            dv_s_stride,
            dv_h_stride,
            dv_d_stride,
        )

    alias N_DTYPES = len(_DTYPES)
    alias N_HEADDIMS = len(_HEADDIMS)
    if dtype_code < 0 or dtype_code >= N_DTYPES:
        raise Error("invalid dtype_code")
    var dispatched: Bool = False

    @parameter
    for dt_idx in range(N_DTYPES):
        alias dt = _DTYPES[dt_idx]

        @parameter
        for hd_idx in range(N_HEADDIMS):
            alias hd = _HEADDIMS[hd_idx]
            if dtype_code == dt_idx and headdim_rt == hd:
                if causal_rt != 0:
                    run[dt, hd, True]()
                else:
                    run[dt, hd, False]()
                dispatched = True

    if not dispatched:
        raise Error(
            "unsupported (dtype, headdim) — dtype ∈ {fp16, bf16, fp32},"
            " headdim ∈ {64, 96, 128}"
        )

    return PythonObject(None)


@export
def PyInit_flash_attn_native() -> PythonObject:
    try:
        var m = PythonModuleBuilder("flash_attn_native")
        m.def_py_function[flash_attn_fwd_cpu]("flash_attn_fwd_cpu")
        m.def_py_function[flash_attn_bwd_cpu]("flash_attn_bwd_cpu")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
