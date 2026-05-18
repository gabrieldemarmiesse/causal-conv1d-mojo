"""Single-variant launch helper for the GPU fwd kernel.

The JIT-generated `variant_<hash>.mojo` files (see `_jit.py`) all call
into this with their comptime params hard-coded. Centralising the
launch logic here keeps each JIT variant template small (~30 lines
instead of the full ~150-line launcher), so codegen + compile per
new variant stays cheap.

`stream_handle_addr == 0` is the "no external stream" sentinel —
used both by pure-Mojo callers (tests) and by the Mac/Metal route
(Metal has no CUDA-style streams; `DeviceStream` raises "Metal
stream not implemented" on Apple). We then enqueue on `ctx`
directly. Otherwise we wrap the caller-supplied CUDA stream.

Caller responsibilities:
- Bail out on any zero-sized dim (`batch == 0 || dim == 0 || seqlen == 0`)
  *before* calling — DeviceContext + enqueue_function reject grid_dim==0.
- Pass the comptime params that select the right kernel specialisation.
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.memory import OpaquePointer
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import fwd_kernel
from common import kNThreads


# When `stream_handle_addr == 0` (Mac/Metal — Metal has no CUDA-style
# streams and `DeviceStream` raises "Metal stream not implemented" on
# the Apple backend), enqueue on `ctx` directly. Otherwise wrap the
# caller-supplied CUDA stream.
fn _has_external_stream(stream_handle_addr: Int) -> Bool:
    return stream_handle_addr != 0


def launch_fwd[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
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
) raises:
    var ctx = DeviceContext()
    var has_stream = _has_external_stream(stream_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

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
                XLT,
                WLT,
                OLT,
            ],
        ]()
        if has_stream:
            var stream = ctx.create_external_stream(stream_opaque)
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
        else:
            ctx.enqueue_function(
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
            # No external stream: caller is the Mac/MPS path (or a
            # pure-Mojo test). On MPS we run on Mojo's own command
            # queue, separate from torch's MPS queue. Block until our
            # writes are committed so torch's subsequent reads observe
            # them — torch doesn't track resources written through a
            # raw `gpuAddress`, so it can't insert the hazard itself.
            # Measured cost: ~10-30% on large shapes, ~0% on small
            # ones (launch overhead dominates regardless).
            ctx.synchronize()

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


