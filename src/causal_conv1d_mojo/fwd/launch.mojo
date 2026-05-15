"""Single-variant launch helper for the GPU fwd kernel.

The JIT-generated `variant_<hash>.mojo` files (see `_jit.py`) all call
into this with their comptime params hard-coded. Centralising the
launch logic here keeps each JIT variant template small (~30 lines
instead of the full ~150-line launcher), so codegen + compile per
new variant stays cheap.

Caller responsibilities:
- Bail out on any zero-sized dim (`batch == 0 || dim == 0 || seqlen == 0`)
  *before* calling — DeviceContext + enqueue_function reject grid_dim==0.
- Pass the comptime params that select the right kernel specialisation.

On AMD specifically, `var ctx = DeviceContext()` per call ends up
calling `hipStreamCreate` + matching `hipStreamDestroy` each time
(visible in torch.profiler traces — these CPU calls bleed into
`self_device_time_total` for the surrounding kernel and inflate the
measured per-call time). The Python wrapper caches a "context handle"
the first time a variant is loaded and passes it in as
`ctx_handle_addr`; we wrap that via the non-owning DeviceContext
constructor so no fresh hipStream is created per call. Matches the
pattern from `update/` and `bwd_full/`.
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.memory import OpaquePointer
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import fwd_kernel
from common import kNThreads


def acquire_ctx_handle() raises -> Int:
    """Create a DeviceContext, retain its handle, and leak the wrapper.

    The Python side calls this once per variant and caches the returned
    integer. The handle stays alive for the duration of the process —
    the matching release happens at process exit (or never).

    Returns the address of the underlying C++ DeviceContext as an Int.
    """
    var ctx = DeviceContext()
    # Retain to bump the refcount so `ctx.__del__` (when this function
    # returns) doesn't free the underlying resource. The caller is now
    # responsible for the extra refcount.
    ctx._retain()
    var raw_ptr = ctx._handle.value()
    return Int(raw_ptr)


def launch_fwd[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
    vec_aligned: Bool,
](
    batch_int: Int,
    dim_int: Int,
    seqlen_int: Int,
    x_addr: Int,
    w_addr: Int,
    b_addr: Int,
    o_addr: Int,
    seq_idx_addr: Int,
    initial_states_addr: Int,
    x_b_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    w_c_stride: Int,
    w_w_stride: Int,
    o_b_stride: Int,
    o_c_stride: Int,
    o_l_stride: Int,
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids the hipStreamCreate/Destroy that would happen with the
    # default `DeviceContext()` constructor each call.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )
    var stream = ctx.create_external_stream(stream_opaque)

    # One block per (channel, batch); block walks all chunks of seqlen.
    var grid = (dim_int, batch_int)

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

    # The `contig_inner` fast path bakes `Idx[1]()` into the inner stride
    # slot of each Layout, so the multiply on the innermost stride folds
    # out at comptime. The Layout *types* differ per branch (comptime
    # stride slot vs runtime), so the TileTensor construction has to
    # live inside the comptime if. The compile+enqueue is identical in
    # both arms — hoisted into `launch` below.
    @parameter
    def launch[
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
                contig_inner,
                aligned_seq,
                vec_aligned,
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
                contig_inner,
                aligned_seq,
                vec_aligned,
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

    # 32-bit strides: the kernel's address arithmetic
    # (`batch*b_stride + chan*c_stride + seq*l_stride`) drops from i64
    # multiplies (`s_mul_hi_u32 + s_mul_i32 + s_add`) to a single i32
    # `s_mul_i32`. Saves ~7 SGPRs of address setup per block, which
    # the small-shape regime is sensitive to. Mirrors modular's own
    # `causal_conv1d.mojo` (all stride args are `UInt32`) and upstream
    # Tri Dao (uses int for ConvParamsBase stride fields). 32-bit
    # is fine — strides this large would need a single buffer >16GB.
    comptime if contig_inner:
        var x_tt = TileTensor(
            x_ptr,
            Layout(
                (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                (
                    Idx(UInt32(x_b_stride)),
                    Idx(UInt32(x_c_stride)),
                    Idx[1](),
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (Idx(dim_int), Idx[width]()),
                (Idx(UInt32(w_c_stride)), Idx[1]()),
            ),
        )
        var o_tt = TileTensor(
            o_ptr,
            Layout(
                (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                (
                    Idx(UInt32(o_b_stride)),
                    Idx(UInt32(o_c_stride)),
                    Idx[1](),
                ),
            ),
        )
        launch(x_tt.as_immut(), w_tt.as_immut(), o_tt)
    else:
        var x_tt = TileTensor(
            x_ptr,
            Layout(
                (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                (
                    Idx(UInt32(x_b_stride)),
                    Idx(UInt32(x_c_stride)),
                    Idx(UInt32(x_l_stride)),
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (Idx(dim_int), Idx[width]()),
                (Idx(UInt32(w_c_stride)), Idx(UInt32(w_w_stride))),
            ),
        )
        var o_tt = TileTensor(
            o_ptr,
            Layout(
                (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
                (
                    Idx(UInt32(o_b_stride)),
                    Idx(UInt32(o_c_stride)),
                    Idx(UInt32(o_l_stride)),
                ),
            ),
        )
        launch(x_tt.as_immut(), w_tt.as_immut(), o_tt)
