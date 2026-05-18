"""Single-variant launch helper for the GPU bwd_full kernel.

The JIT-generated `variant_<hash>.mojo` files (see `_jit.py`) all call
into this with their comptime params hard-coded. Centralising the
launch logic here keeps each JIT variant template small.

Caller responsibilities:
- Bail out on any zero-sized dim (`batch == 0 || dim == 0 || seqlen == 0`)
  *before* calling.
- Zero `dweight_acc` (and `dbias_acc` if `has_bias=True`) before the call
  — the kernel atomic-adds into them. (Autograd's `backward()` already
  does this for the grad tensors it allocates.)
- Pass the comptime params that select the right kernel specialisation.
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.memory import OpaquePointer
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import bwd_full_kernel
from common import kNThreads


# When `stream_handle_addr == 0` (Mac/Metal — Metal has no CUDA-style
# streams and `DeviceStream` raises "Metal stream not implemented" on
# the Apple backend), enqueue on `ctx` directly. Otherwise wrap the
# caller-supplied CUDA stream.
fn _has_external_stream(stream_handle_addr: Int) -> Bool:
    return stream_handle_addr != 0


def launch_bwd_full[
    dtype: DType,
    n_elts: Int,
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
    dout_addr: Int,
    dx_addr: Int,
    dweight_acc_addr: Int,
    dbias_acc_addr: Int,
    seq_idx_addr: Int,
    initial_states_addr: Int,
    dinitial_states_addr: Int,
    x_b_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    w_c_stride: Int,
    w_w_stride: Int,
    dout_b_stride: Int,
    dout_c_stride: Int,
    dout_l_stride: Int,
    dx_b_stride: Int,
    dx_c_stride: Int,
    dx_l_stride: Int,
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
    dinitial_states_b_stride: Int,
    dinitial_states_c_stride: Int,
    dinitial_states_l_stride: Int,
    stream_handle_addr: Int,
) raises:
    var ctx = DeviceContext()
    var has_stream = _has_external_stream(stream_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

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
                n_elts,
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
                n_elts,
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
        ]()
        if has_stream:
            var stream = ctx.create_external_stream(stream_opaque)
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
        else:
            ctx.enqueue_function(
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
            # See comment in fwd/launch.mojo's equivalent branch — block
            # on Mojo's command queue so torch sees our writes (we wrote
            # through a raw `gpuAddress`, which torch can't hazard-track).
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


