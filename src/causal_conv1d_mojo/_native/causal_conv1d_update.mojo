"""GPU single-step update kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_update.cu`. This is the autoregressive-
decode op: for each batch element, take a tiny slice of new tokens
`x[b, :, 0..seqlen)` (typically `seqlen=1` for one-token-at-a-time
decoding), update the rolling `conv_state[b, :, 0..state_len)` buffer,
and emit the conv output.

State semantics (linear / non-circular):
- `conv_state[b, c, :]` holds the most recent `state_len` x values for
  channel `c` of batch `b`, with the oldest at index 0 and the most
  recent at index `state_len-1`. After this call: state's oldest
  `seqlen` values are dropped, the new `seqlen` x values are appended
  on the right.

State semantics (circular / `is_circular=True`):
- `conv_state[b, c, :]` is a circular buffer; `cache_seqlens[b]` is
  the per-batch write head (modulo `state_len`). The kernel reads
  the W-1 historical values from positions `[cache_seqlen-(W-1),
  cache_seqlen)` (mod state_len) and writes new x values starting at
  `cache_seqlen` (advancing mod state_len). State is mutated in place
  but `cache_seqlens` is NOT updated by the kernel — caller advances
  by `seqlen` between calls.

`has_state_indices=True` redirects the state row: the conv state for
batch element `b` is read/written at `state_indices[b]` instead of
`b`. If `state_indices[b] < 0`, the output for that batch element is
forced to zero and state is not touched (padding token in vLLM-style
serving). cache_seqlens is still indexed by `b`, not the redirected
coord — matching upstream.

Each thread handles one (batch, channel). One block covers
`kNThreadsUpdate=64` channels for a given batch.
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
    has_state_indices: Bool,
    is_circular: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    state_len: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    conv_state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
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
        has_bias / apply_silu: standard.
        has_state_indices: redirect the state row via state_indices[b].
            `state_indices[b] < 0` → output zeros for that batch element.
        is_circular: treat conv_state as a circular buffer; cache_seqlens
            holds the per-batch write head.
    Pointers gated False by comptime are never dereferenced.
    """
    comptime accum_t = DType.float32

    var batch_id: Int = block_idx.x
    var channel_id: Int = block_idx.y * kNThreadsUpdate + thread_idx.x
    if channel_id >= dim:
        return

    var x_base = batch_id * x_batch_stride + channel_id * x_c_stride
    var out_base = batch_id * out_batch_stride + channel_id * out_c_stride
    var weight_base = channel_id * weight_c_stride

    # Resolve the state-row coordinate. With has_state_indices=False this
    # is just batch_id; otherwise we look it up. A negative index marks
    # a padding token: zero the output and skip state mutation entirely.
    var state_batch_coord: Int = batch_id

    comptime if has_state_indices:
        var idx_val: Int = Int(state_indices_ptr[batch_id])
        if idx_val < 0:
            for i in range(seqlen):
                output_ptr[out_base + i * out_l_stride] = Scalar[dtype](0)
            return
        state_batch_coord = idx_val

    var state_base = (
        state_batch_coord * state_batch_stride + channel_id * state_c_stride
    )

    var weights = SIMD[accum_t, width](0)

    comptime for k in range(width):
        weights[k] = weight_ptr[weight_base + k * weight_w_stride].cast[
            accum_t
        ]()

    var bias_v: Scalar[accum_t] = 0

    comptime if has_bias:
        bias_v = bias_ptr[channel_id].cast[accum_t]()

    # Circular-mode: cache_seqlens is the per-batch write head. Reads
    # start `(width-1)` slots to its left (with wrap); writes happen
    # AT the head and advance.
    var update_idx: Int = 0

    comptime if is_circular:
        var cs: Int = Int(cache_seqlens_ptr[batch_id]) % state_len
        update_idx = cs - (width - 1)
        if update_idx < 0:
            update_idx += state_len

    var advance_len = seqlen
    var x_vals = SIMD[accum_t, width](0)

    comptime if not is_circular:
        # Phase 1 (linear): shift state left by `seqlen`.
        for i in range(state_len - advance_len - (width - 1)):
            conv_state_ptr[state_base + i * state_l_stride] = conv_state_ptr[
                state_base + (i + advance_len) * state_l_stride
            ]

        # Phase 2 (linear): read trailing W-1 history into x_vals (with
        # writeback for the small-state_len edge case).
        comptime for i in range(width - 1):
            var read_idx: Int = state_len - (width - 1) + i
            var state_val = conv_state_ptr[
                state_base + read_idx * state_l_stride
            ]
            var write_idx: Int = state_len - advance_len - (width - 1) + i
            if i < advance_len + (width - 1) and write_idx >= 0:
                conv_state_ptr[
                    state_base + write_idx * state_l_stride
                ] = state_val
            x_vals[i] = state_val.cast[accum_t]()
    else:
        # Phase 1+2 (circular): read W-1 history starting at update_idx,
        # advancing with wrap. After this, update_idx = cache_seqlen %
        # state_len — the position where the new x[0] will be written.
        comptime for i in range(width - 1):
            var state_val = conv_state_ptr[
                state_base + update_idx * state_l_stride
            ]
            x_vals[i] = state_val.cast[accum_t]()
            update_idx += 1
            if update_idx >= state_len:
                update_idx -= state_len

    # Phase 3: walk new x, write into state, emit output.
    for i in range(seqlen):
        var x_val = x_ptr[x_base + i * x_l_stride]

        comptime if not is_circular:
            var write_idx: Int = state_len - advance_len + i
            if i < advance_len and write_idx >= 0:
                conv_state_ptr[state_base + write_idx * state_l_stride] = x_val
        else:
            conv_state_ptr[state_base + update_idx * state_l_stride] = x_val
            update_idx += 1
            if update_idx >= state_len:
                update_idx -= state_len

        x_vals[width - 1] = x_val.cast[accum_t]()

        var out_val: Scalar[accum_t] = bias_v

        comptime for k in range(width):
            out_val += weights[k] * x_vals[k]

        comptime if apply_silu:
            out_val = _silu_f32(Float32(out_val))

        output_ptr[out_base + i * out_l_stride] = out_val.cast[dtype]()

        # Slide x_vals left by 1 for the next output position.
        comptime for k in range(width - 1):
            x_vals[k] = x_vals[k + 1]
