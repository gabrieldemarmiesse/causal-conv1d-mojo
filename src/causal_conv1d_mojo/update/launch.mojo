"""Single-variant launch helper for the GPU update kernel.

The JIT-generated `variant_<hash>.mojo` files (see `_jit.py`) all call
into this with their comptime params hard-coded.

Caller responsibilities:
- Bail out on `batch == 0 || dim == 0` before calling.
- Pass the comptime params that select the right kernel specialisation.

Implementation note: launch passes raw pointers + Int32 strides (not
TileTensor). See the long-form explanation at the top of
`update/kernel.mojo` for the PTX-level reasoning. Short version: at
decode shapes the kernel is 2-8μs total per call, and TileTensor's
prologue (offsetted `ld.param.b32` loads from a packed kernarg layout
struct, plus i64 address math by default) costs ~0.15-0.30μs that we
don't pay with direct `.u32` stride kernargs. `fwd/` and `bwd_full/`
hide the same cost behind much longer kernel runtimes.

On AMD specifically, `var ctx = DeviceContext()` per call ends up
calling `hipStreamCreate` + matching `hipStreamDestroy` each time
(visible in torch.profiler traces — these CPU calls are the bulk of
per-call overhead). The Python wrapper caches a "context handle" the
first time a variant is loaded and passes it in as
`ctx_handle_addr`; we wrap that via the non-owning DeviceContext
constructor so no fresh hipStream is created per call.
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer

from kernel import kNThreadsUpdate, update_kernel


def launch_update[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    has_state_indices: Bool,
    is_circular: Bool,
    use_external_stream: Bool,
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
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids the hipStreamCreate/Destroy that would happen with the
    # default `DeviceContext()` constructor each call.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    # `use_external_stream` is comptime — see fwd/launch.mojo for why
    # this can't be a runtime branch without regressing wall-clock.
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
    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            Int32(dim_int),
            Int32(seqlen_int),
            Int32(state_len_int),
            x_ptr,
            w_ptr,
            b_ptr,
            state_ptr,
            state_indices_ptr,
            cache_seqlens_ptr,
            o_ptr,
            Int32(x_b_stride),
            Int32(x_c_stride),
            Int32(x_l_stride),
            Int32(w_c_stride),
            Int32(state_b_stride),
            Int32(state_c_stride),
            Int32(state_l_stride),
            Int32(o_b_stride),
            Int32(o_c_stride),
            Int32(o_l_stride),
            grid_dim=grid,
            block_dim=(kNThreadsUpdate,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            Int32(dim_int),
            Int32(seqlen_int),
            Int32(state_len_int),
            x_ptr,
            w_ptr,
            b_ptr,
            state_ptr,
            state_indices_ptr,
            cache_seqlens_ptr,
            o_ptr,
            Int32(x_b_stride),
            Int32(x_c_stride),
            Int32(x_l_stride),
            Int32(w_c_stride),
            Int32(state_b_stride),
            Int32(state_c_stride),
            Int32(state_l_stride),
            Int32(o_b_stride),
            Int32(o_c_stride),
            Int32(o_l_stride),
            grid_dim=grid,
            block_dim=(kNThreadsUpdate,),
        )
        ctx.synchronize()
