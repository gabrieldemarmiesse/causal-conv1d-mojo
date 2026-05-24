"""Single-variant launch helper for the GPU fwd kernel.

The JIT-generated `variant.mojo` (see `_jit.py`) calls into this
with comptime params hard-coded — keeps each JIT variant compile
cheap because all the boilerplate (DeviceContext setup, TileTensor
construction, compile_function + enqueue_function) lives here.

Mirrors causal-conv1d-mojo's `fwd/launch.mojo` pattern. See its
docstrings for the AMD `hipStreamCreate`-per-call rationale and the
`use_external_stream` comptime gate — same considerations apply.
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from layout import TileTensor, Idx, TensorLayout
from layout.tile_layout import Layout

from kernel import fwd_kernel
from common import kNThreads, kBlockM


def launch_fwd[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    o_addr: Int,
    q_b_stride: Int,
    q_l_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_l_stride: Int,
    k_h_stride: Int,
    v_b_stride: Int,
    v_l_stride: Int,
    v_h_stride: Int,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids hipStreamCreate/Destroy every call. See causal-conv1d-mojo's
    # `_ctx.mojo` + matching launch.mojo for the long-form rationale.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    # Grid: one block per (q-tile, head, batch). Each block handles
    # kBlockM query positions and has kNThreads threads.
    var grid = (ceildiv(seqlen_int, kBlockM), nheads_int, batch_int)

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

    # Layout for (B, L, H, D). `head_dim` is comptime → bake it as a
    # static dim. The inner stride is always 1 for the contiguous-in-D
    # layout torch hands us; baking it as `Idx[1]()` lets the comptime
    # multiply fold out. All other strides are dynamic UInt32 — keeps
    # address math at 32 bits (see causal-conv1d-mojo's launchers for
    # the `mul.lo.s64` vs `mul.wide.s32` story).
    var q_tt = TileTensor(
        q_ptr,
        Layout(
            (
                Idx(batch_int),
                Idx(seqlen_int),
                Idx(nheads_int),
                Idx[head_dim](),
            ),
            (
                Idx(UInt32(q_b_stride)),
                Idx(UInt32(q_l_stride)),
                Idx(UInt32(q_h_stride)),
                Idx[1](),
            ),
        ),
    )
    var k_tt = TileTensor(
        k_ptr,
        Layout(
            (
                Idx(batch_int),
                Idx(seqlen_int),
                Idx(nheads_int),
                Idx[head_dim](),
            ),
            (
                Idx(UInt32(k_b_stride)),
                Idx(UInt32(k_l_stride)),
                Idx(UInt32(k_h_stride)),
                Idx[1](),
            ),
        ),
    )
    var v_tt = TileTensor(
        v_ptr,
        Layout(
            (
                Idx(batch_int),
                Idx(seqlen_int),
                Idx(nheads_int),
                Idx[head_dim](),
            ),
            (
                Idx(UInt32(v_b_stride)),
                Idx(UInt32(v_l_stride)),
                Idx(UInt32(v_h_stride)),
                Idx[1](),
            ),
        ),
    )
    var o_tt = TileTensor(
        o_ptr,
        Layout(
            (
                Idx(batch_int),
                Idx(seqlen_int),
                Idx(nheads_int),
                Idx[head_dim](),
            ),
            (
                Idx(UInt32(o_b_stride)),
                Idx(UInt32(o_l_stride)),
                Idx(UInt32(o_h_stride)),
                Idx[1](),
            ),
        ),
    )

    var compiled = ctx.compile_function[
        fwd_kernel[
            dtype,
            head_dim,
            type_of(q_tt).LayoutType,
            type_of(k_tt).LayoutType,
            type_of(v_tt).LayoutType,
            type_of(o_tt).LayoutType,
        ],
        fwd_kernel[
            dtype,
            head_dim,
            type_of(q_tt).LayoutType,
            type_of(k_tt).LayoutType,
            type_of(v_tt).LayoutType,
            type_of(o_tt).LayoutType,
        ],
    ]()
    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            softmax_scale,
            q_tt.as_immut(),
            k_tt.as_immut(),
            v_tt.as_immut(),
            o_tt,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            softmax_scale,
            q_tt.as_immut(),
            k_tt.as_immut(),
            v_tt.as_immut(),
            o_tt,
            grid_dim=grid,
            block_dim=(kNThreads,),
        )
        ctx.synchronize()
