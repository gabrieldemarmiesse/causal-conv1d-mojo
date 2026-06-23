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

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.memory import OpaquePointer
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import bwd_full_kernel
from common import kNThreads


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
    use_external_stream: Bool,
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
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids the hipStreamCreate/Destroy that would otherwise happen
    # on every launch with the default `DeviceContext()` constructor.
    # See `acquire_ctx_handle` above.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    # `use_external_stream` is a comptime gate — see the matching
    # block in `fwd/launch.mojo` for why this can't be a runtime if
    # without regressing NVIDIA wall-clock by ~30 μs/call.
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    # Grid layout (batch, dim) matches upstream Tri Dao. On AMD with a
    # single batch (the small-shape regime we care about), this means
    # gridDim.x = 1 and gridDim.y = dim — and AMD's CU dispatcher
    # scans gridDim.x first when filling the GPU, so adjacent dim_id
    # blocks land on adjacent CUs (better channel-stride locality on
    # the dweight atomics). With (dim, batch) the order is inverted
    # and the dweight atomics scatter more across L2 partitions.
    var grid = (batch_int, dim_int)

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

    # seq_idx (B, L), initial_states (B, D, W-1), dinitial_states
    # (B, D, W-1) become TileTensors so the kernel can do scalar
    # `tensor[b, c, i]` indexing instead of base + i * stride by hand.
    # All strides dynamic — keeps the variant layout type uniform so
    # the JIT cache doesn't fragment per-shape. When the corresponding
    # `has_*` comptime flag is False, the kernel never indexes into
    # the tensor, so a null pointer + zero strides is fine.
    var seq_idx_tt = TileTensor(
        seq_idx_ptr,
        Layout(
            (batch_int, seqlen_int),
            (UInt32(seq_idx_b_stride), UInt32(seq_idx_l_stride)),
        ),
    )
    var initial_states_tt = TileTensor(
        initial_states_ptr,
        Layout(
            (batch_int, dim_int, Idx[width - 1]),
            (
                UInt32(initial_states_b_stride),
                UInt32(initial_states_c_stride),
                UInt32(initial_states_l_stride),
            ),
        ),
    )
    var dinitial_states_tt = TileTensor(
        dinitial_states_ptr,
        Layout(
            (batch_int, dim_int, Idx[width - 1]),
            (
                UInt32(dinitial_states_b_stride),
                UInt32(dinitial_states_c_stride),
                UInt32(dinitial_states_l_stride),
            ),
        ),
    )

    @parameter
    def launch[
        XLT: TensorLayout,
        WLT: TensorLayout,
        DoutLT: TensorLayout,
        DxLT: TensorLayout,
        SLT: TensorLayout,
        ILT: TensorLayout,
        DILT: TensorLayout,
    ](
        x_tt: TileTensor[dtype, XLT, ImmutAnyOrigin],
        w_tt: TileTensor[dtype, WLT, ImmutAnyOrigin],
        dout_tt: TileTensor[dtype, DoutLT, ImmutAnyOrigin],
        dx_tt: TileTensor[mut=True, dtype, DxLT, MutAnyOrigin],
        s_tt: TileTensor[DType.int32, SLT, ImmutAnyOrigin],
        i_tt: TileTensor[dtype, ILT, ImmutAnyOrigin],
        di_tt: TileTensor[mut=True, dtype, DILT, MutAnyOrigin],
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
                SLT,
                ILT,
                DILT,
            ]
        ]()
        comptime if use_external_stream:
            var stream = ctx.create_external_stream(stream_opaque)
            stream.enqueue_function(
                compiled,
                seqlen_int,
                x_tt,
                w_tt,
                b_ptr,
                dout_tt,
                s_tt,
                i_tt,
                dx_tt,
                dweight_acc_ptr,
                dbias_acc_ptr,
                di_tt,
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
                s_tt,
                i_tt,
                dx_tt,
                dweight_acc_ptr,
                dbias_acc_ptr,
                di_tt,
                grid_dim=grid,
                block_dim=(kNThreads,),
            )
            ctx.synchronize()

    # Pass strides as `UInt32` to keep the kernel-side address math at
    # 32 bits. The default `Int` (64-bit) routes every stride-times-
    # index multiply through `s_mul_i32` + `s_mul_hi_u32` (a 64×32 →
    # 64 multiply pair on AMD); with `UInt32` the high-half multiplies
    # are dead-code-eliminated and we get a single `s_mul_i32`. Saves
    # ~7 SGPRs of address setup per block, which the small-shape
    # regime is sensitive to. Mirrors the fwd launcher. 32-bit is fine
    # — strides this large would need a single buffer >16 GB.
    comptime if contig_inner:
        var x_tt = TileTensor(
            x_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(x_b_stride),
                    UInt32(x_c_stride),
                    Idx[1],
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (dim_int, Idx[width]),
                (UInt32(w_c_stride), Idx[1]),
            ),
        )
        var dout_tt = TileTensor(
            dout_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(dout_b_stride),
                    UInt32(dout_c_stride),
                    Idx[1],
                ),
            ),
        )
        var dx_tt = TileTensor(
            dx_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(dx_b_stride),
                    UInt32(dx_c_stride),
                    Idx[1],
                ),
            ),
        )
        launch(
            x_tt.as_immut(),
            w_tt.as_immut(),
            dout_tt.as_immut(),
            dx_tt,
            seq_idx_tt.as_immut(),
            initial_states_tt.as_immut(),
            dinitial_states_tt,
        )
    else:
        var x_tt = TileTensor(
            x_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(x_b_stride),
                    UInt32(x_c_stride),
                    UInt32(x_l_stride),
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (dim_int, Idx[width]),
                (UInt32(w_c_stride), UInt32(w_w_stride)),
            ),
        )
        var dout_tt = TileTensor(
            dout_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(dout_b_stride),
                    UInt32(dout_c_stride),
                    UInt32(dout_l_stride),
                ),
            ),
        )
        var dx_tt = TileTensor(
            dx_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    UInt32(dx_b_stride),
                    UInt32(dx_c_stride),
                    UInt32(dx_l_stride),
                ),
            ),
        )
        launch(
            x_tt.as_immut(),
            w_tt.as_immut(),
            dout_tt.as_immut(),
            dx_tt,
            seq_idx_tt.as_immut(),
            initial_states_tt.as_immut(),
            dinitial_states_tt,
        )
