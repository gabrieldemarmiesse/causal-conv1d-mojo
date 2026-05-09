"""GPU single-step update kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_update.cu`. This is the autoregressive-
decode op: for each batch element, take a tiny slice of new tokens
`x[b, :, 0..seqlen)` (typically `seqlen=1` for one-token-at-a-time
decoding), update the rolling `conv_state[b, :, 0..state_len)` buffer,
and emit the conv output.

State semantics (non-circular):
- `conv_state[b, c, :]` holds the most recent `state_len` x values for
  channel `c` of batch `b`, with the oldest at index 0 and the most
  recent at index `state_len-1`. After this call: state's oldest
  `seqlen` values are dropped, the new `seqlen` x values are appended
  on the right.
- Only the last `width-1` values of state matter for the conv math
  (they form the historical context for the new x).

Each thread handles one (batch, channel). One block covers
`kNThreadsUpdate=64` channels for a given batch.

The circular-buffer mode (`cache_seqlens != None`) and per-batch state
indirection (`conv_state_indices != None`) are out of scope here — the
public API raises `NotImplementedError` for those.
"""

from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
)

from causal_conv1d_common import _silu_f32


# Threads per block for the update kernel. Smaller than the 128 used by
# the full forward because update only touches `seqlen` outputs per
# (B, C) — extra parallelism across channels matters more.
comptime kNThreadsUpdate: Int = 64


fn update_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    state_len: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    conv_state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    state_batch_stride: Int,
    state_c_stride: Int,
    state_l_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    """Causal conv1d single-step update, GPU.

    Comptime params:
        has_bias: load `bias_ptr[d]` per channel, or skip and use 0.
        apply_silu: apply silu (= swish) to each output, or skip.
    Both pointers are never dereferenced when their gate is False.
    """
    alias accum_t = DType.float32

    var batch_id: Int = block_idx.x
    var channel_id: Int = block_idx.y * kNThreadsUpdate + thread_idx.x
    if channel_id >= dim:
        return

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var out_base = batch_id * out_batch_stride + channel_id * out_c_stride
    var state_base = batch_id * state_batch_stride + channel_id * state_c_stride
    var weight_base = channel_id * weight_c_stride

    var weights = SIMD[accum_t, width](0)

    @parameter
    for k in range(width):
        weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[
            accum_t
        ]()

    var bias_v: Scalar[accum_t] = 0

    @parameter
    if has_bias:
        bias_v = bias_ptr[channel_id].cast[accum_t]()

    # Phase 1: shift state left by `seqlen` (drop the oldest `seqlen`
    # values). Only positions [0, state_len - seqlen - (W-1)) actually
    # need to be re-written here — the rest is overwritten by the
    # subsequent read+writeback loop. Mirrors upstream's first loop.
    var advance_len = seqlen
    for i in range(state_len - advance_len - (width - 1)):
        conv_state_ptr[state_base + i * state_l_stride] = conv_state_ptr[
            state_base + (i + advance_len) * state_l_stride
        ]

    # Phase 2: read the trailing W-1 historical values into x_vals[0..W-2].
    # These are the conv's history before the new x. While reading, also
    # write each value back into the post-shift slot when the destination
    # exists (small state_len edge case from upstream).
    var x_vals = SIMD[accum_t, width](0)

    @parameter
    for i in range(width - 1):
        var read_idx: Int = state_len - (width - 1) + i
        var state_val = conv_state_ptr[state_base + read_idx * state_l_stride]
        var write_idx: Int = state_len - advance_len - (width - 1) + i
        if i < advance_len + (width - 1) and write_idx >= 0:
            conv_state_ptr[state_base + write_idx * state_l_stride] = state_val
        x_vals[i] = state_val.cast[accum_t]()

    # Phase 3: walk new x, write into state's tail, accumulate output.
    for i in range(seqlen):
        var x_val = x_ptr[x_base + i * x_l_stride]
        var write_idx: Int = state_len - advance_len + i
        if i < advance_len and write_idx >= 0:
            conv_state_ptr[state_base + write_idx * state_l_stride] = x_val
        x_vals[width - 1] = x_val.cast[accum_t]()

        var out_val: Scalar[accum_t] = bias_v

        @parameter
        for k in range(width):
            out_val += weights[k] * x_vals[k]

        @parameter
        if apply_silu:
            out_val = _silu_f32(Float32(out_val))

        output_ptr[out_base + i * out_l_stride] = out_val.cast[dtype]()

        # Slide x_vals left by 1 for the next output position.
        @parameter
        for k in range(width - 1):
            x_vals[k] = x_vals[k + 1]
