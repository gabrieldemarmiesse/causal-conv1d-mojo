"""GPU forward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_fwd.cu`. The launcher lives in
`causal_conv1d_native.mojo` (the dispatcher); this file holds only
the kernel itself.

Refactored to use `TileTensor` at the kernel boundary for the three
main tensors (x, weight, output). Each gets its own comptime
`LayoutType: TensorLayout` — the dispatcher picks the LayoutType
based on whether the relevant innermost stride is 1 (the
"contig_inner" fast path). When stride=1 is baked into the Layout
via `Idx[1]()`, LLVM folds out the inner-stride multiply just like
the previous hand-written `@parameter if contig_inner` branches.

Bias / seq_idx / initial_states stay as raw pointers — bias is 1-D
(no stride to worry about), and seq_idx / initial_states are rarely
used in the perf-critical path.
"""

from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
)
from layout import TileTensor, TensorLayout

from causal_conv1d_common import _silu_f32, kNElts, kNThreads


fn fwd_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    OLayoutType: TensorLayout,
](
    seqlen: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin],
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin].flat_rank == 3
):
    """Causal conv1d forward, GPU.

    The comptime fast path "innermost stride == 1" is now encoded by
    the dispatcher in the Layout types of `x`, `weight`, `output`: when
    the inner stride slot is `Idx[1]()`, the multiply is folded out at
    compile time.

    `has_seq_idx`: when True, `seq_idx_ptr` is a `(B, L)` int32 tensor of
    sequence ids. For each output position `t`, historical reads from
    `src_t < t` are masked to 0 unless `seq_idx[b, src_t] == seq_idx[b, t]`,
    so packed mini-batches don't bleed across sequence boundaries.
    `seq_idx[b, t] < 0` marks a padding position — its output is forced
    to 0.

    `has_initial_states`: when True, `initial_states_ptr` is a
    `(B, D, W-1)` tensor that supplies the historical context before
    `t = 0`. For `src_t in [-(W-1), 0)`, we read
    `initial_states[b, c, src_t + W - 1]` instead of treating the
    out-of-range position as zero. Mutually exclusive with `has_seq_idx`.
    """
    comptime accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)

    comptime for k in range(width):
        weights[k] = weight[channel_id, k].cast[accum_t]()

    var cur_bias: Scalar[accum_t] = 0

    comptime if has_bias:
        cur_bias = bias_ptr[channel_id].cast[accum_t]()

    var seq_start = chunk_id * kNThreads * kNElts + tidx * kNElts
    if seq_start >= seqlen:
        return

    var seq_idx_base: Int = batch_id * seq_idx_b_stride
    var initial_states_base: Int = (
        batch_id * initial_states_b_stride
        + channel_id * initial_states_c_stride
    )

    comptime for i in range(kNElts):
        var t = seq_start + i
        if t >= seqlen:
            break
        var acc: Scalar[accum_t] = cur_bias

        var cur_id: Int32 = 0

        comptime if has_seq_idx:
            cur_id = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]

        comptime for k in range(width):
            var src_t = t + k - (width - 1)
            var val: Scalar[accum_t] = 0
            if src_t >= 0:
                val = x[batch_id, channel_id, src_t].cast[accum_t]()

                comptime if has_seq_idx:
                    var src_id: Int32 = seq_idx_ptr[
                        seq_idx_base + src_t * seq_idx_l_stride
                    ]
                    if src_id != cur_id:
                        val = 0
            else:

                comptime if has_initial_states:
                    # src_t in [-(W-1), 0); index 0..W-2 of initial_states.
                    var is_idx: Int = src_t + (width - 1)
                    val = initial_states_ptr[
                        initial_states_base + is_idx * initial_states_l_stride
                    ].cast[accum_t]()
            acc += val * weights[k]

        comptime if apply_silu:
            acc = _silu_f32(Float32(acc))

        # Padding tokens (seq_idx[t] < 0): output is forced to 0,
        # mirroring upstream's behaviour. Done after activation so the
        # bias/silu don't leak into a "padding" row.
        comptime if has_seq_idx:
            if cur_id < 0:
                acc = 0

        output[batch_id, channel_id, t] = acc.cast[dtype]()
