"""Single-variant launch helper for the GPU update kernel.

The JIT-generated `variant_<hash>.mojo` files (see `_jit.py`) all call
into this with their comptime params hard-coded.

Caller responsibilities:
- Bail out on `batch == 0 || dim == 0` before calling.
- Pass the comptime params that select the right kernel specialisation.
"""

from std.gpu.host import DeviceContext, DeviceStream
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from layout import TileTensor, Idx
from layout.tile_layout import Layout

from kernel import kNThreadsUpdate, update_kernel


# When `stream_handle_addr == 0` (Mac/Metal — Metal has no CUDA-style
# streams and `DeviceStream` raises "Metal stream not implemented" on
# the Apple backend), enqueue on `ctx` directly. Otherwise wrap the
# caller-supplied CUDA stream.
fn _has_external_stream(stream_handle_addr: Int) -> Bool:
    return stream_handle_addr != 0


def acquire_ctx_handle() raises -> Int:
    """See fwd/launch.mojo. Caches one DeviceContext per variant —
    skips the ~340 µs newCommandQueue cost on every call."""
    var ctx = DeviceContext()
    ctx._retain()
    return Int(ctx._handle.value())


def launch_update[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    has_state_indices: Bool,
    is_circular: Bool,
](
    batch_int: Int,
    dim_int: Int,
    seqlen_int: Int,
    state_len_int: Int,
    x_addr: Int,
    w_addr: Int,
    b_addr: Int,
    state_addr: Int,
    o_addr: Int,
    state_indices_addr: Int,
    cache_seqlens_addr: Int,
    x_b_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    w_c_stride: Int,
    w_w_stride: Int,
    state_b_stride: Int,
    state_c_stride: Int,
    state_l_stride: Int,
    o_b_stride: Int,
    o_c_stride: Int,
    o_l_stride: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var has_stream = _has_external_stream(stream_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    var grid = (batch_int, ceildiv(dim_int, kNThreadsUpdate))

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
    var state_tt = TileTensor(
        state_ptr,
        Layout(
            (Idx(batch_int), Idx(dim_int), Idx(state_len_int)),
            (Idx(state_b_stride), Idx(state_c_stride), Idx(state_l_stride)),
        ),
    )
    var o_tt = TileTensor(
        o_ptr,
        Layout(
            (Idx(batch_int), Idx(dim_int), Idx(seqlen_int)),
            (Idx(o_b_stride), Idx(o_c_stride), Idx(o_l_stride)),
        ),
    )
    var compiled = ctx.compile_function[
        update_kernel[
            dtype,
            width,
            has_bias,
            apply_silu,
            has_state_indices,
            is_circular,
            type_of(x_tt).LayoutType,
            type_of(w_tt).LayoutType,
            type_of(state_tt).LayoutType,
            type_of(o_tt).LayoutType,
        ],
        update_kernel[
            dtype,
            width,
            has_bias,
            apply_silu,
            has_state_indices,
            is_circular,
            type_of(x_tt).LayoutType,
            type_of(w_tt).LayoutType,
            type_of(state_tt).LayoutType,
            type_of(o_tt).LayoutType,
        ],
    ]()
    if has_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            state_len_int,
            x_tt.as_immut(),
            w_tt.as_immut(),
            b_ptr,
            state_tt,
            state_indices_ptr,
            cache_seqlens_ptr,
            o_tt,
            grid_dim=grid,
            block_dim=(kNThreadsUpdate,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            state_len_int,
            x_tt.as_immut(),
            w_tt.as_immut(),
            b_ptr,
            state_tt,
            state_indices_ptr,
            cache_seqlens_ptr,
            o_tt,
            grid_dim=grid,
            block_dim=(kNThreadsUpdate,),
        )
        # See comment in fwd/launch.mojo's equivalent branch — block
        # on Mojo's command queue so torch sees our writes (we wrote
        # through a raw `gpuAddress`, which torch can't hazard-track).
        ctx.synchronize()
