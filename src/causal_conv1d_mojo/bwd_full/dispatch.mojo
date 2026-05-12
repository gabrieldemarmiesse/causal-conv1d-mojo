"""Single-function dispatcher for `causal_conv1d_bwd_full` (split out of `causal_conv1d_native.mojo` so each entry
point compiles to its own .so and is imported lazily on
first call from the Python wrapper)."""

from std.gpu.host import DeviceContext
from std.itertools import product
from std.math import ceildiv
from std.memory import OpaquePointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import bwd_full_kernel
from common import kNEltsBwd, kNThreads

comptime _BOOLS = [False, True]
comptime _WIDTHS = [2, 3, 4]


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
    def run[dtype: DType]() raises:
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
        def enqueue_bwd[
            width: Int,
            has_bias: Bool,
            has_seq_idx: Bool,
            has_initial_states: Bool,
            apply_silu: Bool,
            contig_inner: Bool,
            aligned_seq: Bool,
        ]() raises:
            # Identical to fwd's pattern: when `contig_inner` is True, bake
            # `Idx[1]()` into the inner stride slot of every Layout so the
            # innermost-stride multiply folds out at comptime. Width slot
            # of the weight Layout is also comptime (`Idx[width]()`). The
            # compile+enqueue block is identical in both arms and lives in
            # `launch` below.
            @parameter
            def launch[
                XLT: TensorLayout,
                WLT: TensorLayout,
                DoutLT: TensorLayout,
                DxLT: TensorLayout,
            ](
                x_tt: TileTensor[dtype, XLT, ImmutAnyOrigin],
                w_tt: TileTensor[dtype, WLT, ImmutAnyOrigin],
                dout_tt: TileTensor[dtype, DoutLT, ImmutAnyOrigin],
                dx_tt: TileTensor[mut=True, dtype, DxLT, MutAnyOrigin],
            ) raises:
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
                        XLT,
                        WLT,
                        DoutLT,
                        DxLT,
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
                        XLT,
                        WLT,
                        DoutLT,
                        DxLT,
                    ],
                    dump_asm=StaticString("./ptx/bwd_full_%.ptx"),
                ]()
                stream.enqueue_function(
                    compiled,
                    seqlen_int,
                    x_tt,
                    w_tt,
                    b_ptr,
                    dout_tt,
                    seq_idx_ptr,
                    initial_states_ptr,
                    dx_tt,
                    dweight_acc_ptr,
                    dbias_acc_ptr,
                    dinitial_states_ptr,
                    seq_idx_b_stride,
                    seq_idx_l_stride,
                    initial_states_b_stride,
                    initial_states_c_stride,
                    initial_states_l_stride,
                    dinitial_states_b_stride,
                    dinitial_states_c_stride,
                    dinitial_states_l_stride,
                    grid_dim=grid,
                    block_dim=(kNThreads,),
                )

            comptime if contig_inner:
                var x_tt = TileTensor(
                    x_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(x_b_stride), Idx(x_c_stride), Idx[1]()),
                    ),
                )
                var w_tt = TileTensor(
                    w_ptr,
                    Layout(
                        (Idx(dim_int), Idx[width]()),
                        (Idx(w_c_stride), Idx[1]()),
                    ),
                )
                var dout_tt = TileTensor(
                    dout_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(dout_b_stride), Idx(dout_c_stride), Idx[1]()),
                    ),
                )
                var dx_tt = TileTensor(
                    dx_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(dx_b_stride), Idx(dx_c_stride), Idx[1]()),
                    ),
                )
                launch(x_tt.as_immut(), w_tt.as_immut(), dout_tt.as_immut(), dx_tt)
            else:
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
                        (Idx(dim_int), Idx[width]()),
                        (Idx(w_c_stride), Idx(w_w_stride)),
                    ),
                )
                var dout_tt = TileTensor(
                    dout_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(dout_b_stride), Idx(dout_c_stride), Idx(dout_l_stride)),
                    ),
                )
                var dx_tt = TileTensor(
                    dx_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(dx_b_stride), Idx(dx_c_stride), Idx(dx_l_stride)),
                    ),
                )
                launch(x_tt.as_immut(), w_tt.as_immut(), dout_tt.as_immut(), dx_tt)

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
        def dispatch_w[width: Int]() raises:
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


@export
def PyInit_dispatch() -> PythonObject:
    try:
        var m = PythonModuleBuilder("dispatch")
        m.def_py_function[causal_conv1d_bwd_full]("causal_conv1d_bwd_full")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
