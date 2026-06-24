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
    use_external_stream: Bool,
    dump_assembly_into: StaticString = "",
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
    x_b_stride: UInt32,
    x_c_stride: UInt32,
    x_l_stride: UInt32,
    w_c_stride: UInt32,
    w_w_stride: UInt32,
    o_b_stride: UInt32,
    o_c_stride: UInt32,
    o_l_stride: UInt32,
    seq_idx_b_stride: UInt32,
    seq_idx_l_stride: UInt32,
    initial_states_b_stride: UInt32,
    initial_states_c_stride: UInt32,
    initial_states_l_stride: UInt32,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids the hipStreamCreate/Destroy that would happen with the
    # default `DeviceContext()` constructor each call.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    # `use_external_stream` is a *comptime* gate: CUDA/HIP variants pass
    # True (always wrap torch's stream and `enqueue_function` on it);
    # Metal variants pass False (Metal has no CUDA-style stream; we
    # `enqueue_function` on `ctx` directly and sync after). Keeping
    # this comptime instead of a runtime `if stream_handle_addr != 0`
    # check is load-bearing for NVIDIA wall-clock perf — even a
    # predictable runtime branch around enqueue_function adds ~30 μs
    # per call (probably because both branches' enqueue_function arg
    # packs get codegen'd, defeating some inlining).
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

    # The `contig_inner` fast path bakes `Idx[1]` into the inner stride
    # slot of each Layout, so the multiply on the innermost stride folds
    # out at comptime. The Layout *types* differ per branch (comptime
    # stride slot vs runtime), so the TileTensor construction has to
    # live inside the comptime if. The compile+enqueue is identical in
    # both arms — hoisted into `launch` below.
    # seq_idx (B, L) and initial_states (B, D, W-1) become TileTensors
    # so the kernel can do `seq_idx[b, t]` / `initial_states[b, c, i]`
    # instead of doing the address math by hand. All strides are kept
    # dynamic (no `Idx[const]`) — keeps the variant layout type fixed
    # across shapes so we don't accidentally explode the JIT cache.
    # When the corresponding `has_*` comptime flag is False the kernel
    # never indexes into the tensor, so the address (may be null) and
    # the strides (may be 0) don't matter — the construction is purely
    # type-level setup.
    var seq_idx_tt = TileTensor(
        seq_idx_ptr,
        Layout(
            (batch_int, seqlen_int),
            (seq_idx_b_stride, seq_idx_l_stride),
        ),
    )
    var initial_states_tt = TileTensor(
        initial_states_ptr,
        Layout(
            (batch_int, dim_int, Idx[width - 1]),
            (
                initial_states_b_stride,
                initial_states_c_stride,
                initial_states_l_stride,
            ),
        ),
    )

    @parameter
    def launch[
        XLT: TensorLayout,
        WLT: TensorLayout,
        OLT: TensorLayout,
        SLT: TensorLayout,
        ILT: TensorLayout,
    ](
        x_tt: TileTensor[dtype, XLT, ImmutAnyOrigin],
        w_tt: TileTensor[dtype, WLT, ImmutAnyOrigin],
        o_tt: TileTensor[mut=True, dtype, OLT, MutAnyOrigin],
        s_tt: TileTensor[DType.int32, SLT, ImmutAnyOrigin],
        i_tt: TileTensor[dtype, ILT, ImmutAnyOrigin],
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
                SLT,
                ILT,
            ],
            dump_asm=dump_assembly_into,
        ]()
        comptime if use_external_stream:
            var stream = ctx.create_external_stream(stream_opaque)
            stream.enqueue_function(
                compiled,
                seqlen_int,
                x_tt,
                w_tt,
                b_ptr,
                s_tt,
                i_tt,
                o_tt,
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
                s_tt,
                i_tt,
                o_tt,
                grid_dim=grid,
                block_dim=(kNThreads,),
            )
            ctx.synchronize()

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
                (batch_int, dim_int, seqlen_int),
                (
                    x_b_stride,
                    x_c_stride,
                    Idx[1],
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (dim_int, Idx[width]),
                (w_c_stride, Idx[1]),
            ),
        )
        var o_tt = TileTensor(
            o_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    o_b_stride,
                    o_c_stride,
                    Idx[1],
                ),
            ),
        )
        launch(
            x_tt.as_immut(),
            w_tt.as_immut(),
            o_tt,
            seq_idx_tt.as_immut(),
            initial_states_tt.as_immut(),
        )
    else:
        var x_tt = TileTensor(
            x_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    x_b_stride,
                    x_c_stride,
                    x_l_stride,
                ),
            ),
        )
        var w_tt = TileTensor(
            w_ptr,
            Layout(
                (dim_int, Idx[width]),
                (w_c_stride, w_w_stride),
            ),
        )
        var o_tt = TileTensor(
            o_ptr,
            Layout(
                (batch_int, dim_int, seqlen_int),
                (
                    o_b_stride,
                    o_c_stride,
                    o_l_stride,
                ),
            ),
        )
        launch(
            x_tt.as_immut(),
            w_tt.as_immut(),
            o_tt,
            seq_idx_tt.as_immut(),
            initial_states_tt.as_immut(),
        )
