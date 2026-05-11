"""Single-function dispatcher for `causal_conv1d_fwd` (split out of `causal_conv1d_native.mojo` so each entry
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

from kernel import fwd_kernel
from common import kNElts, kNThreads

comptime _BOOLS = [False, True]
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
            # The `contig_inner` fast path bakes `Idx[1]()` into the inner
            # stride slot of each Layout, so the multiply on the innermost
            # stride folds out at comptime — same effect as the old
            # `@parameter if contig_inner` branch in the kernel body. The
            # Layout *types* differ per branch (comptime stride slot vs
            # runtime), so the TileTensor construction has to live inside
            # the comptime if. The compile+enqueue is identical in both
            # arms — hoisted into `launch` below.
            @parameter
            fn launch[
                XLT: TensorLayout, WLT: TensorLayout, OLT: TensorLayout
            ](
                x_tt: TileTensor[dtype, XLT, ImmutAnyOrigin],
                w_tt: TileTensor[dtype, WLT, ImmutAnyOrigin],
                o_tt: TileTensor[mut=True, dtype, OLT, MutAnyOrigin],
            ) raises:
                var compiled = ctx.compile_function[
                    fwd_kernel[
                        dtype,
                        width,
                        has_bias,
                        has_seq_idx,
                        has_initial_states,
                        apply_silu,
                        XLT,
                        WLT,
                        OLT,
                    ],
                    fwd_kernel[
                        dtype,
                        width,
                        has_bias,
                        has_seq_idx,
                        has_initial_states,
                        apply_silu,
                        XLT,
                        WLT,
                        OLT,
                    ],
                ]()
                stream.enqueue_function(
                    compiled,
                    seqlen_int,
                    x_tt,
                    w_tt,
                    b_ptr,
                    seq_idx_ptr,
                    initial_states_ptr,
                    o_tt,
                    seq_idx_b_stride,
                    seq_idx_l_stride,
                    initial_states_b_stride,
                    initial_states_c_stride,
                    initial_states_l_stride,
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
                var o_tt = TileTensor(
                    o_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(o_b_stride), Idx(o_c_stride), Idx[1]()),
                    ),
                )
                launch(x_tt.as_immut(), w_tt.as_immut(), o_tt)
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
                var o_tt = TileTensor(
                    o_ptr,
                    Layout(
                        (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                        (Idx(o_b_stride), Idx(o_c_stride), Idx(o_l_stride)),
                    ),
                )
                launch(x_tt.as_immut(), w_tt.as_immut(), o_tt)

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


@export
def PyInit_dispatch() -> PythonObject:
    try:
        var m = PythonModuleBuilder("dispatch")
        m.def_py_function[causal_conv1d_fwd]("causal_conv1d_fwd")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
