"""GPU forward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_fwd.cu`. The launcher lives in
`causal_conv1d_native.mojo` (the dispatcher); this file holds only
the kernel itself.
"""

from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
)

from causal_conv1d_common import _silu_f32, kNElts, kNThreads


fn fwd_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
](
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    """Causal conv1d forward, GPU.

    `contig_inner` is the comptime fast path: when True, the innermost
    axes of x / weight / out have stride=1 and we drop the
    `* x_l_stride` / `* weight_w_stride` / `* out_l_stride` multiplies
    so the compiler can constant-fold the index math (~2× kernel time
    on memory-bound shapes if we don't).

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
    alias accum_t = DType.float32

    var tidx: Int = thread_idx.x
    var batch_id: Int = block_idx.z
    var channel_id: Int = block_idx.y
    var chunk_id: Int = block_idx.x

    var weights = InlineArray[Scalar[accum_t], width](uninitialized=True)
    var weight_base = channel_id * weight_c_stride

    @parameter
    for k in range(width):

        @parameter
        if contig_inner:
            weights[k] = weight_ptr[weight_base + k].cast[accum_t]()
        else:
            weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[
                accum_t
            ]()

    var cur_bias: Scalar[accum_t] = 0

    @parameter
    if has_bias:
        cur_bias = bias_ptr[channel_id].cast[accum_t]()

    var seq_start = chunk_id * kNThreads * kNElts + tidx * kNElts
    if seq_start >= seqlen:
        return

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var out_base = batch_id * out_batch_stride + channel_id * out_c_stride
    var seq_idx_base: Int = batch_id * seq_idx_b_stride
    var initial_states_base: Int = (
        batch_id * initial_states_b_stride
        + channel_id * initial_states_c_stride
    )

    @parameter
    for i in range(kNElts):
        var t = seq_start + i
        if t >= seqlen:
            break
        var acc: Scalar[accum_t] = cur_bias

        var cur_id: Int32 = 0

        @parameter
        if has_seq_idx:
            cur_id = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]

        @parameter
        for k in range(width):
            var src_t = t + k - (width - 1)
            var val: Scalar[accum_t] = 0
            if src_t >= 0:

                @parameter
                if contig_inner:
                    val = x_ptr[x_base + src_t].cast[accum_t]()
                else:
                    val = x_ptr[x_base + src_t * x_l_stride].cast[accum_t]()

                @parameter
                if has_seq_idx:
                    var src_id: Int32 = seq_idx_ptr[
                        seq_idx_base + src_t * seq_idx_l_stride
                    ]
                    if src_id != cur_id:
                        val = 0
            else:

                @parameter
                if has_initial_states:
                    # src_t in [-(W-1), 0); index 0..W-2 of initial_states.
                    var is_idx: Int = src_t + (width - 1)
                    val = initial_states_ptr[
                        initial_states_base + is_idx * initial_states_l_stride
                    ].cast[accum_t]()
            acc += val * weights[k]

        @parameter
        if apply_silu:
            acc = _silu_f32(Float32(acc))

        # Padding tokens (seq_idx[t] < 0): output is forced to 0,
        # mirroring upstream's behaviour. Done after activation so the
        # bias/silu don't leak into a "padding" row.
        @parameter
        if has_seq_idx:
            if cur_id < 0:
                acc = 0

        @parameter
        if contig_inner:
            output_ptr[out_base + t] = acc.cast[dtype]()
        else:
            output_ptr[out_base + t * out_l_stride] = acc.cast[dtype]()
