"""Direct Python -> Mojo extension for causal_conv1d, no MAX framework.

Built as a CPython extension via:
    mojo build causal_conv1d_native.mojo --emit shared-lib -o causal_conv1d_native.so

Then importable as `from causal_conv1d_mojo._native import causal_conv1d_native`.

This file is the dispatcher (mirrors upstream's `causal_conv1d.cpp`):
it parses the Python tuple args, builds the dtype × width × flags
comptime dispatch tree, and forwards to the kernel implementations
in the sibling `_fwd.mojo` / `_bwd.mojo` / `_cpu.mojo` files.

Folding the stride-1 multiplies into the fast `contig_inner` path
matters: passing inner strides as runtime args around the kernel,
even when always 1, costs ~2× kernel time on a memory-bound workload
because the compiler can no longer constant-fold the index math.
"""

from std.gpu.host import DeviceContext
from std.itertools import product
from std.math import ceildiv
from std.memory import OpaquePointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from causal_conv1d_bwd import bwd_full_kernel
from causal_conv1d_bwd_cpu import bwd_kernel_cpu
from causal_conv1d_common import kNElts, kNEltsBwd, kNThreads
from causal_conv1d_fwd import fwd_kernel
from causal_conv1d_fwd_cpu import fwd_kernel_cpu
from causal_conv1d_update import kNThreadsUpdate, update_kernel
from causal_conv1d_update_cpu import update_kernel_cpu


# Bool list, used everywhere in the dispatch trees.
comptime _BOOLS = [False, True]
# Width allowlist. Used by each launcher's `comptime for w in _WIDTHS`
# sweep to lift the runtime `width` to a comptime parameter; an unmatched
# width falls through and the launcher raises.
comptime _WIDTHS = [2, 3, 4]


def causal_conv1d_fwd(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """GPU forward. dtype + width are dispatched at runtime.

    Python tuple positional args (28):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  output_data_ptr  (int)
        4  batch  (int)
        5  dim    (int)
        6  seqlen (int)
        7  x_batch_stride  (int)
        8  x_c_stride      (int)
        9  x_l_stride      (int)
        10 weight_c_stride (int)
        11 weight_w_stride (int)
        12 out_batch_stride  (int)
        13 out_c_stride      (int)
        14 out_l_stride      (int)
        15 has_bias (int, 0 or 1) — 1 ⇒ load `bias_ptr[d]` per channel
        16 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish on the output
        17 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        18 cuda_stream_handle (int)  -- torch.cuda.current_stream().cuda_stream
        19 has_seq_idx (int, 0 or 1) — 1 ⇒ mask reads on seq_idx
        20 seq_idx_data_ptr (int, int32) — pass 0 if `has_seq_idx=0`
        21 seq_idx_batch_stride (int)
        22 seq_idx_l_stride (int)
        23 width (int) — supported: 2, 3, 4
        24 has_initial_states (int, 0 or 1) — 1 ⇒ read pre-`t=0` history
        25 initial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        26 initial_states_batch_stride (int)
        27 initial_states_c_stride (int)
        28 initial_states_l_stride (int)
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
    var has_bias_rt: Bool = Int(py=args[15]) != 0
    var apply_silu_rt: Bool = Int(py=args[16]) != 0
    var dtype_code: Int = Int(py=args[17])
    var stream_handle_addr: Int = Int(py=args[18])
    var has_seq_idx_rt: Bool = Int(py=args[19]) != 0
    var seq_idx_addr: Int = Int(py=args[20])
    var seq_idx_b_stride: Int = Int(py=args[21])
    var seq_idx_l_stride: Int = Int(py=args[22])
    var width_rt: Int = Int(py=args[23])
    var has_initial_states_rt: Bool = Int(py=args[24]) != 0
    var initial_states_addr: Int = Int(py=args[25])
    var initial_states_b_stride: Int = Int(py=args[26])
    var initial_states_c_stride: Int = Int(py=args[27])
    var initial_states_l_stride: Int = Int(py=args[28])

    # Zero-sized tensor: nothing to compute. `enqueue_function` rejects
    # any grid_dim == 0, so early-out before touching DeviceContext.
    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
        return PythonObject(None)

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    var grid = (
        ceildiv(seqlen_int, kNThreads * kNElts),
        dim_int,
        batch_int,
    )
    var contig_inner_rt: Bool = (
        x_l_stride == 1 and w_w_stride == 1 and o_l_stride == 1
    )

    # `run[dtype]` materialises the dtype-typed pointers and the comptime
    # dispatch tree below it. Every reachable (dtype, width, ...) leaf is
    # fully compiled at `.so` build time -- host machine code for the CPU
    # kernels, and ptxas-emitted SASS cubins embedded in `.rodata` for
    # the GPU kernels. There is no Mojo-side JIT: at runtime,
    # `compile_function[]()` just hands the prebuilt cubin to the CUDA
    # driver to load (cached per-context after first call).
    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=seq_idx_addr
        )
        var initial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=initial_states_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )

        # Kernel variants per (dtype, width): has_bias × has_seq_idx ×
        # has_initial_states × apply_silu × contig_inner. seq_idx and
        # initial_states are mutually exclusive at the public API; we only
        # emit cubins for the 3 reachable (seq_idx, init) combinations.
        @parameter
        fn enqueue_fwd[
            width: Int,
            has_bias: Bool,
            has_seq_idx: Bool,
            has_initial_states: Bool,
            apply_silu: Bool,
            contig_inner: Bool,
        ]() raises:
            var compiled = ctx.compile_function[
                fwd_kernel[
                    dtype,
                    width,
                    has_bias,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    contig_inner,
                ],
                fwd_kernel[
                    dtype,
                    width,
                    has_bias,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    contig_inner,
                ],
            ]()
            stream.enqueue_function(
                compiled,
                seqlen_int,
                x_ptr,
                w_ptr,
                b_ptr,
                seq_idx_ptr,
                initial_states_ptr,
                o_ptr,
                x_b_stride,
                x_c_stride,
                x_l_stride,
                w_c_stride,
                w_w_stride,
                seq_idx_b_stride,
                seq_idx_l_stride,
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
                o_b_stride,
                o_c_stride,
                o_l_stride,
                grid_dim=grid,
                block_dim=(kNThreads,),
            )

        # 4-way comptime sweep across (has_seq_idx, has_initial_states,
        # apply_silu, contig_inner), nested under the has_bias loop. The
        # `comptime if` filter drops the (seq_idx & init) combination that
        # the public API rules out as mutually exclusive — so we don't
        # waste a cubin on a code path that will never be hit at runtime.
        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb in _BOOLS:
                comptime for hs, hi, silu, contig in product(
                    _BOOLS, _BOOLS, _BOOLS, _BOOLS
                ):
                    comptime if not (hs and hi):
                        if (
                            hb == has_bias_rt
                            and hs == has_seq_idx_rt
                            and hi == has_initial_states_rt
                            and silu == apply_silu_rt
                            and contig == contig_inner_rt
                        ):
                            enqueue_fwd[width, hb, hs, hi, silu, contig]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


def causal_conv1d_bwd_full(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """GPU fused backward. dtype + width are dispatched at runtime.

    Caller must zero `dweight_acc` (and `dbias_acc` if `has_bias=1`)
    before this call. `dweight_acc` / `dbias_acc` are always fp32
    accumulators regardless of the input dtype (precision-preserving).

    Python tuple positional args (29):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  dout_data_ptr  (int)
        4  dx_data_ptr  (int)
        5  dweight_acc_data_ptr  (int, fp32)
        6  dbias_acc_data_ptr  (int, fp32) — pass 0 if `has_bias=0`
        7  batch  (int)
        8  dim    (int)
        9  seqlen (int)
        10 x_batch_stride  (int)
        11 x_c_stride      (int)
        12 x_l_stride      (int)
        13 weight_c_stride (int)
        14 weight_w_stride (int)
        15 dout_batch_stride  (int)
        16 dout_c_stride      (int)
        17 dout_l_stride      (int)
        18 dx_batch_stride  (int)
        19 dx_c_stride      (int)
        20 dx_l_stride      (int)
        21 has_bias (int, 0 or 1)
        22 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish was applied on fwd
        23 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        24 cuda_stream_handle (int)
        25 width (int) — supported: 2, 3, 4
        26 has_seq_idx (int, 0 or 1)
        27 seq_idx_data_ptr (int, int32) — pass 0 if `has_seq_idx=0`
        28 seq_idx_batch_stride (int)
        29 seq_idx_l_stride (int)
        30 has_initial_states (int, 0 or 1)
        31 initial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        32 initial_states_batch_stride (int)
        33 initial_states_c_stride (int)
        34 initial_states_l_stride (int)
        35 dinitial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        36 dinitial_states_batch_stride (int)
        37 dinitial_states_c_stride (int)
        38 dinitial_states_l_stride (int)
    """
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
    var has_bias_rt: Bool = Int(py=args[21]) != 0
    var apply_silu_rt: Bool = Int(py=args[22]) != 0
    var dtype_code: Int = Int(py=args[23])
    var stream_handle_addr: Int = Int(py=args[24])
    var width_rt: Int = Int(py=args[25])
    var has_seq_idx_rt: Bool = Int(py=args[26]) != 0
    var seq_idx_addr: Int = Int(py=args[27])
    var seq_idx_b_stride: Int = Int(py=args[28])
    var seq_idx_l_stride: Int = Int(py=args[29])
    var has_initial_states_rt: Bool = Int(py=args[30]) != 0
    var initial_states_addr: Int = Int(py=args[31])
    var initial_states_b_stride: Int = Int(py=args[32])
    var initial_states_c_stride: Int = Int(py=args[33])
    var initial_states_l_stride: Int = Int(py=args[34])
    var dinitial_states_addr: Int = Int(py=args[35])
    var dinitial_states_b_stride: Int = Int(py=args[36])
    var dinitial_states_c_stride: Int = Int(py=args[37])
    var dinitial_states_l_stride: Int = Int(py=args[38])

    # Zero-sized tensor: nothing to compute and no atomic updates to
    # dweight_acc / dbias_acc needed (the autograd `backward` already
    # zero-initialised them). Early-out before `enqueue_function`, which
    # rejects any grid_dim == 0.
    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
        return PythonObject(None)

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One block per (channel, batch); block walks all chunks of seqlen.
    var grid = (dim_int, batch_int)

    var contig_inner_rt: Bool = (
        x_l_stride == 1
        and w_w_stride == 1
        and dout_l_stride == 1
        and dx_l_stride == 1
    )
    var aligned_seq_rt: Bool = seqlen_int % (kNThreads * kNEltsBwd) == 0

    # `run[dtype]` materialises dtype-typed pointers and the comptime
    # has_bias/apply_silu/contig/aligned dispatch tree below it. The
    # dweight_acc / dbias_acc accumulators stay fp32 regardless of dtype.
    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var dout_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dout_addr
        )
        var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=seq_idx_addr
        )
        var initial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=initial_states_addr
        )
        var dx_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dx_addr
        )
        var dweight_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=dweight_acc_addr
        )
        var dbias_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=dbias_acc_addr
        )
        var dinitial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dinitial_states_addr
        )

        @parameter
        fn enqueue_bwd[
            width: Int,
            has_bias: Bool,
            has_seq_idx: Bool,
            has_initial_states: Bool,
            apply_silu: Bool,
            contig_inner: Bool,
            aligned_seq: Bool,
        ]() raises:
            var compiled = ctx.compile_function[
                bwd_full_kernel[
                    dtype,
                    width,
                    has_bias,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    contig_inner,
                    aligned_seq,
                ],
                bwd_full_kernel[
                    dtype,
                    width,
                    has_bias,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    contig_inner,
                    aligned_seq,
                ],
            ]()
            stream.enqueue_function(
                compiled,
                seqlen_int,
                x_ptr,
                w_ptr,
                b_ptr,
                dout_ptr,
                seq_idx_ptr,
                initial_states_ptr,
                dx_ptr,
                dweight_acc_ptr,
                dbias_acc_ptr,
                dinitial_states_ptr,
                x_b_stride,
                x_c_stride,
                x_l_stride,
                w_c_stride,
                w_w_stride,
                dout_b_stride,
                dout_c_stride,
                dout_l_stride,
                seq_idx_b_stride,
                seq_idx_l_stride,
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
                dx_b_stride,
                dx_c_stride,
                dx_l_stride,
                dinitial_states_b_stride,
                dinitial_states_c_stride,
                dinitial_states_l_stride,
                grid_dim=grid,
                block_dim=(kNThreads,),
            )

        # 6-flag comptime sweep across (has_bias, has_seq_idx,
        # has_initial_states, apply_silu, contig_inner, aligned_seq).
        # std.itertools.product caps at 4 iterables, so has_seq_idx and
        # has_initial_states are the outer comptime loops and the
        # remaining 4 form the inner product. `aligned_seq=True` only
        # makes sense with `contig_inner=True`. Note: seq_idx and
        # initial_states are mutually exclusive at the public API, but
        # we still emit the (hs=T, hi=T) combination — keeps the sweep
        # symmetric and the `comptime if` filter only catches the
        # aligned/contig invariant.
        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hs in _BOOLS:
                comptime for hi in _BOOLS:
                    comptime for hb, silu, contig, aligned in product(
                        _BOOLS, _BOOLS, _BOOLS, _BOOLS
                    ):
                        comptime if not (aligned and not contig):
                            if (
                                hb == has_bias_rt
                                and hs == has_seq_idx_rt
                                and hi == has_initial_states_rt
                                and silu == apply_silu_rt
                                and contig == contig_inner_rt
                                and aligned == aligned_seq_rt
                            ):
                                enqueue_bwd[
                                    width, hb, hs, hi, silu, contig, aligned
                                ]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


def causal_conv1d_fwd_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU forward. dtype + width are dispatched at runtime.

    Python tuple positional args (28):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  output_data_ptr  (int)
        4  batch  (int)
        5  dim    (int)
        6  seqlen (int)
        7  x_batch_stride  (int)
        8  x_c_stride      (int)
        9  x_l_stride      (int)
        10 weight_c_stride (int)
        11 weight_w_stride (int)
        12 out_batch_stride  (int)
        13 out_c_stride      (int)
        14 out_l_stride      (int)
        15 has_bias (int, 0 or 1)
        16 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish on the output
        17 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        18 has_seq_idx (int, 0 or 1)
        19 seq_idx_data_ptr (int, int32) — pass 0 if `has_seq_idx=0`
        20 seq_idx_batch_stride (int)
        21 seq_idx_l_stride (int)
        22 width (int) — supported: 2, 3, 4
        23 has_initial_states (int, 0 or 1)
        24 initial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        25 initial_states_batch_stride (int)
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
    var has_bias_rt: Bool = Int(py=args[15]) != 0
    var apply_silu_rt: Bool = Int(py=args[16]) != 0
    var dtype_code: Int = Int(py=args[17])
    var has_seq_idx_rt: Bool = Int(py=args[18]) != 0
    var seq_idx_addr: Int = Int(py=args[19])
    var seq_idx_b_stride: Int = Int(py=args[20])
    var seq_idx_l_stride: Int = Int(py=args[21])
    var width_rt: Int = Int(py=args[22])
    var has_initial_states_rt: Bool = Int(py=args[23]) != 0
    var initial_states_addr: Int = Int(py=args[24])
    var initial_states_b_stride: Int = Int(py=args[25])
    var initial_states_c_stride: Int = Int(py=args[26])
    var initial_states_l_stride: Int = Int(py=args[27])

    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=seq_idx_addr
        )
        var initial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=initial_states_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )

        @parameter
        fn dispatch[
            width: Int,
            has_bias: Bool,
            has_seq_idx: Bool,
            has_initial_states: Bool,
            apply_silu: Bool,
        ]() raises:
            fwd_kernel_cpu[
                dtype,
                width,
                has_bias,
                has_seq_idx,
                has_initial_states,
                apply_silu,
            ](
                batch_int,
                dim_int,
                seqlen_int,
                x_ptr,
                w_ptr,
                b_ptr,
                seq_idx_ptr,
                initial_states_ptr,
                o_ptr,
                x_b_stride,
                x_c_stride,
                x_l_stride,
                w_c_stride,
                w_w_stride,
                seq_idx_b_stride,
                seq_idx_l_stride,
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
                o_b_stride,
                o_c_stride,
                o_l_stride,
            )

        # Comptime sweep across (has_bias, has_seq_idx, has_initial_states,
        # apply_silu). seq_idx & init are mutually exclusive at the public
        # API; the `comptime if` drops that combo so we don't waste a CPU
        # specialisation on it.
        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb in _BOOLS:
                comptime for hs, hi, silu in product(_BOOLS, _BOOLS, _BOOLS):
                    comptime if not (hs and hi):
                        if (
                            hb == has_bias_rt
                            and hs == has_seq_idx_rt
                            and hi == has_initial_states_rt
                            and silu == apply_silu_rt
                        ):
                            dispatch[width, hb, hs, hi, silu]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


def causal_conv1d_bwd_full_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU fused backward. dtype + width are dispatched at runtime.

    Caller must zero `dweight_acc` (and `dbias_acc` if `has_bias=1`)
    before this call. Same arg layout as the GPU launcher minus the
    `cuda_stream_handle`.

    Python tuple positional args (29):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  dout_data_ptr  (int)
        4  dx_data_ptr  (int)
        5  dweight_acc_data_ptr  (int, fp32)
        6  dbias_acc_data_ptr  (int, fp32) — pass 0 if `has_bias=0`
        7  batch  (int)
        8  dim    (int)
        9  seqlen (int)
        10 x_batch_stride  (int)
        11 x_c_stride      (int)
        12 x_l_stride      (int)
        13 weight_c_stride (int)
        14 weight_w_stride (int)
        15 dout_batch_stride  (int)
        16 dout_c_stride      (int)
        17 dout_l_stride      (int)
        18 dx_batch_stride  (int)
        19 dx_c_stride      (int)
        20 dx_l_stride      (int)
        21 has_bias (int, 0 or 1)
        22 apply_silu (int, 0 or 1) — 1 ⇒ silu/swish was applied on fwd
        23 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        24 width (int) — supported: 2, 3, 4
        25 has_seq_idx (int, 0 or 1)
        26 seq_idx_data_ptr (int, int32) — pass 0 if `has_seq_idx=0`
        27 seq_idx_batch_stride (int)
        28 seq_idx_l_stride (int)
        29 has_initial_states (int, 0 or 1)
        30 initial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        31 initial_states_batch_stride (int)
        32 initial_states_c_stride (int)
        33 initial_states_l_stride (int)
        34 dinitial_states_data_ptr (int) — pass 0 if `has_initial_states=0`
        35 dinitial_states_batch_stride (int)
        36 dinitial_states_c_stride (int)
        37 dinitial_states_l_stride (int)
    """
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
    var has_bias_rt: Bool = Int(py=args[21]) != 0
    var apply_silu_rt: Bool = Int(py=args[22]) != 0
    var dtype_code: Int = Int(py=args[23])
    var width_rt: Int = Int(py=args[24])
    var has_seq_idx_rt: Bool = Int(py=args[25]) != 0
    var seq_idx_addr: Int = Int(py=args[26])
    var seq_idx_b_stride: Int = Int(py=args[27])
    var seq_idx_l_stride: Int = Int(py=args[28])
    var has_initial_states_rt: Bool = Int(py=args[29]) != 0
    var initial_states_addr: Int = Int(py=args[30])
    var initial_states_b_stride: Int = Int(py=args[31])
    var initial_states_c_stride: Int = Int(py=args[32])
    var initial_states_l_stride: Int = Int(py=args[33])
    var dinitial_states_addr: Int = Int(py=args[34])
    var dinitial_states_b_stride: Int = Int(py=args[35])
    var dinitial_states_c_stride: Int = Int(py=args[36])
    var dinitial_states_l_stride: Int = Int(py=args[37])

    var dweight_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dweight_acc_addr
    )
    var dbias_acc_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dbias_acc_addr
    )

    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var dout_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dout_addr
        )
        var seq_idx_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=seq_idx_addr
        )
        var initial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=initial_states_addr
        )
        var dx_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dx_addr
        )
        var dinitial_states_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=dinitial_states_addr
        )

        @parameter
        fn dispatch[
            width: Int,
            has_bias: Bool,
            has_seq_idx: Bool,
            has_initial_states: Bool,
            apply_silu: Bool,
        ]() raises:
            bwd_kernel_cpu[
                dtype,
                width,
                has_bias,
                has_seq_idx,
                has_initial_states,
                apply_silu,
            ](
                batch_int,
                dim_int,
                seqlen_int,
                x_ptr,
                w_ptr,
                b_ptr,
                dout_ptr,
                seq_idx_ptr,
                initial_states_ptr,
                dx_ptr,
                dweight_acc_ptr,
                dbias_acc_ptr,
                dinitial_states_ptr,
                x_b_stride,
                x_c_stride,
                x_l_stride,
                w_c_stride,
                w_w_stride,
                dout_b_stride,
                dout_c_stride,
                dout_l_stride,
                seq_idx_b_stride,
                seq_idx_l_stride,
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
                dx_b_stride,
                dx_c_stride,
                dx_l_stride,
                dinitial_states_b_stride,
                dinitial_states_c_stride,
                dinitial_states_l_stride,
            )

        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb, hs, hi, silu in product(
                _BOOLS, _BOOLS, _BOOLS, _BOOLS
            ):
                if (
                    hb == has_bias_rt
                    and hs == has_seq_idx_rt
                    and hi == has_initial_states_rt
                    and silu == apply_silu_rt
                ):
                    dispatch[width, hb, hs, hi, silu]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


def causal_conv1d_update(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """GPU single-step update. dtype + width are dispatched at runtime.

    Updates `conv_state` in place (shifts left by `seqlen` and writes the
    new x values at the tail; or, in circular mode, writes at the
    `cache_seqlens[b]` head with wrap) and emits the conv output.

    Python tuple positional args (29):
        0  x_data_ptr  (int)
        1  weight_data_ptr  (int)
        2  bias_data_ptr  (int) — pass 0 if `has_bias=0`
        3  conv_state_data_ptr  (int)
        4  output_data_ptr  (int)
        5  batch  (int)
        6  dim    (int)
        7  seqlen (int)
        8  state_len (int) — must be >= width-1
        9  x_batch_stride  (int)
        10 x_c_stride      (int)
        11 x_l_stride      (int)
        12 weight_c_stride (int)
        13 weight_w_stride (int)
        14 state_batch_stride  (int)
        15 state_c_stride      (int)
        16 state_l_stride      (int)
        17 out_batch_stride  (int)
        18 out_c_stride      (int)
        19 out_l_stride      (int)
        20 has_bias (int, 0 or 1)
        21 apply_silu (int, 0 or 1)
        22 dtype_code (int) — 0=fp16, 1=bf16, 2=fp32
        23 cuda_stream_handle (int)
        24 width (int) — supported: 2, 3, 4
        25 has_state_indices (int, 0 or 1)
        26 state_indices_data_ptr (int, int32) — pass 0 if `has_state_indices=0`
        27 is_circular (int, 0 or 1)
        28 cache_seqlens_data_ptr (int, int32) — pass 0 if `is_circular=0`
    """
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
    var has_bias_rt: Bool = Int(py=args[20]) != 0
    var apply_silu_rt: Bool = Int(py=args[21]) != 0
    var dtype_code: Int = Int(py=args[22])
    var stream_handle_addr: Int = Int(py=args[23])
    var width_rt: Int = Int(py=args[24])
    var has_state_indices_rt: Bool = Int(py=args[25]) != 0
    var state_indices_addr: Int = Int(py=args[26])
    var is_circular_rt: Bool = Int(py=args[27]) != 0
    var cache_seqlens_addr: Int = Int(py=args[28])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    var ctx = DeviceContext()
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One thread per channel; kNThreadsUpdate channels per block.
    var grid = (
        batch_int,
        ceildiv(dim_int, kNThreadsUpdate),
    )

    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var state_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=state_addr
        )
        var state_indices_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=state_indices_addr
        )
        var cache_seqlens_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=cache_seqlens_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )

        @parameter
        fn enqueue_update[
            width: Int,
            has_bias: Bool,
            apply_silu: Bool,
            has_state_indices: Bool,
            is_circular: Bool,
        ]() raises:
            var compiled = ctx.compile_function[
                update_kernel[
                    dtype,
                    width,
                    has_bias,
                    apply_silu,
                    has_state_indices,
                    is_circular,
                ],
                update_kernel[
                    dtype,
                    width,
                    has_bias,
                    apply_silu,
                    has_state_indices,
                    is_circular,
                ],
            ]()
            stream.enqueue_function(
                compiled,
                batch_int,
                dim_int,
                seqlen_int,
                state_len_int,
                x_ptr,
                w_ptr,
                b_ptr,
                state_ptr,
                state_indices_ptr,
                cache_seqlens_ptr,
                o_ptr,
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
                grid_dim=grid,
                block_dim=(kNThreadsUpdate,),
            )

        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb, silu, hi, circ in product(
                _BOOLS, _BOOLS, _BOOLS, _BOOLS
            ):
                if (
                    hb == has_bias_rt
                    and silu == apply_silu_rt
                    and hi == has_state_indices_rt
                    and circ == is_circular_rt
                ):
                    enqueue_update[width, hb, silu, hi, circ]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


def causal_conv1d_update_cpu(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """CPU single-step update. dtype + width are dispatched at runtime.

    Same arg layout as the GPU launcher minus the `cuda_stream_handle`
    (28 args instead of 29).
    """
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
    var has_bias_rt: Bool = Int(py=args[20]) != 0
    var apply_silu_rt: Bool = Int(py=args[21]) != 0
    var dtype_code: Int = Int(py=args[22])
    var width_rt: Int = Int(py=args[23])
    var has_state_indices_rt: Bool = Int(py=args[24]) != 0
    var state_indices_addr: Int = Int(py=args[25])
    var is_circular_rt: Bool = Int(py=args[26]) != 0
    var cache_seqlens_addr: Int = Int(py=args[27])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    @parameter
    fn run[dtype: DType]() raises:
        var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=x_addr
        )
        var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=w_addr
        )
        var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=b_addr
        )
        var state_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=state_addr
        )
        var state_indices_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=state_indices_addr
        )
        var cache_seqlens_ptr = UnsafePointer[Int32, MutAnyOrigin](
            unsafe_from_address=cache_seqlens_addr
        )
        var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )

        @parameter
        fn dispatch[
            width: Int,
            has_bias: Bool,
            apply_silu: Bool,
            has_state_indices: Bool,
            is_circular: Bool,
        ]() raises:
            update_kernel_cpu[
                dtype,
                width,
                has_bias,
                apply_silu,
                has_state_indices,
                is_circular,
            ](
                batch_int,
                dim_int,
                seqlen_int,
                state_len_int,
                x_ptr,
                w_ptr,
                b_ptr,
                state_ptr,
                state_indices_ptr,
                cache_seqlens_ptr,
                o_ptr,
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
            )

        @parameter
        fn dispatch_w[width: Int]() raises:
            comptime for hb, silu, hi, circ in product(
                _BOOLS, _BOOLS, _BOOLS, _BOOLS
            ):
                if (
                    hb == has_bias_rt
                    and silu == apply_silu_rt
                    and hi == has_state_indices_rt
                    and circ == is_circular_rt
                ):
                    dispatch[width, hb, silu, hi, circ]()

        comptime for w in _WIDTHS:
            if width_rt == w:
                dispatch_w[w]()
                return
        raise Error("unsupported width (only 2, 3, 4 are supported)")

    if dtype_code == 0:
        run[DType.float16]()
    elif dtype_code == 1:
        run[DType.bfloat16]()
    else:
        run[DType.float32]()

    return PythonObject(None)


@export
def PyInit_causal_conv1d_native() -> PythonObject:
    try:
        var m = PythonModuleBuilder("causal_conv1d_native")
        m.def_py_function[causal_conv1d_fwd]("causal_conv1d_fwd")
        m.def_py_function[causal_conv1d_bwd_full]("causal_conv1d_bwd_full")
        m.def_py_function[causal_conv1d_fwd_cpu]("causal_conv1d_fwd_cpu")
        m.def_py_function[causal_conv1d_bwd_full_cpu](
            "causal_conv1d_bwd_full_cpu"
        )
        m.def_py_function[causal_conv1d_update]("causal_conv1d_update")
        m.def_py_function[causal_conv1d_update_cpu]("causal_conv1d_update_cpu")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
