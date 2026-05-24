"""Launch helper for the fwd kernel.

Reconstructs a non-owning `DeviceContext` from a cached handle, computes
the dynamic-smem budget (q + k + v + scratch), JIT-compiles + enqueues
`fwd_kernel`.
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of

from kernel import fwd_kernel
from common import kNThreads, kBlockM, kBlockN, kBlockK, kWM, kWN


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
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    # Dynamic smem budget (bytes):
    #   Q tile:     BM * head_dim * size_of[dtype]
    #   K tile:     BN * head_dim * size_of[dtype]
    #   V tile:     BN * BN       * size_of[dtype]   (BN == head_dim for our shapes)
    #   scratch:    2 * num_warps_n * BM * size_of[accum=fp32]    (zero when num_warps_n == 1)
    # plus the output write-back reuses q_smem in place — same buffer, no extra.
    comptime q_bytes: Int = kBlockM * head_dim * size_of[dtype]()
    comptime k_bytes: Int = kBlockN * head_dim * size_of[dtype]()
    comptime v_bytes: Int = kBlockN * kBlockN * size_of[dtype]()
    # `p_smem` is reserved unconditionally so `multistage_mma`'s
    # a_smem_iter parameter (which must be in SHARED address space)
    # type-checks even when num_warps_n == 1 and P stays in registers.
    comptime p_bytes: Int = kBlockM * kBlockN * size_of[dtype]()
    comptime scratch_bytes: Int = 0  # num_warps_n == 1 ⇒ warp_scratch is empty
    comptime smem_bytes: Int = (
        q_bytes + k_bytes + v_bytes + p_bytes + scratch_bytes
    )

    var q_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    var compiled = ctx.compile_function[
        fwd_kernel[dtype, head_dim],
        fwd_kernel[dtype, head_dim],
    ](func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_bytes))

    var grid = (ceildiv(seqlen_int, Int(kBlockM)), nheads_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            grid_dim=grid,
            block_dim=(kNThreads,),
            shared_mem_bytes=smem_bytes,
        )
        ctx.synchronize()
