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

Implementation note: this kernel takes **raw pointers + Int32 strides**
(not TileTensor, unlike `fwd/` and `bwd_full/`). The decode shapes we
care about (B=1..32, D=256..4096, seqlen=1) put the kernel at ~2-8μs
total wall-clock per call — small enough that *prologue* cost is a
meaningful fraction of runtime, and that's exactly where TileTensor
has two unavoidable overheads:

1. **i64 address math by default.** `TileTensor.linear_idx_type` is
   keyword-overridable but defaults to `DType.int64` for global-memory
   tensors with any dynamic dim (see `_get_index_type` in
   `layout/tile_tensor.mojo`). Every `tensor[b, c, i]` then lowers to
   `mul.lo.s64` for the stride×index multiplies, which on sm_89
   becomes several SASS instructions per multiply versus a single
   `IMAD` for the equivalent i32 multiply. Passing strides as
   `UInt32` in the Layout *doesn't* help — Mojo widens them back to
   i64 before the multiply because the index argument is `Int`.
   Forcing `linear_idx_type=DType.int32` does fix the multiply width
   (we verified via PTX diff), but exposes the second issue:

2. **Layout passed as a packed kernarg struct.** A `TileTensor`
   parameter lowers to a `.align 8 .b8 [N]` kernarg blob holding the
   {base ptr, shape, strides} struct. Reading a stride out is an
   offsetted load: `ld.param.b32 [..._param_6+32]`,
   `ld.param.b32 [..._param_6+36]`, etc. — and for some nested
   layouts (1-D tensors in particular) it's a register-indirect
   `mov.b64 %rd, param_X; ld.param.b64 [%rd]`. The raw-pointer
   version gets every stride as its own top-level `.u32` param,
   loaded directly into a register in one cycle. The difference is
   ~5-10 extra `ld.param.b32` plus 1-2 indirect loads in the
   prologue. Cheap individually, but serialised on the kernarg bus
   and not hidden by anything else in a 2μs kernel.

Empirically (RTX 2000 Ada, fp16, w=4, silu, with-bias, decode shapes):
TileTensor with `linear_idx_type=DType.int32` regressed wall-clock by
+0.15-0.30μs across every shape vs the raw-ptr version. The 16-byte
`global_load_dwordx2`/v4 vec loads were preserved (those just need
the `alignment=` promise on the load call), so the regression is
purely from the prologue, not from worse memory traffic. `fwd/` and
`bwd_full/` don't see this because their kernels run for tens to
hundreds of microseconds — the same prologue cost is in the noise.

So this kernel keeps raw pointers + Int32 strides, and indexes by
hand. Matches upstream Tri Dao's `index_t = uint32_t` approach. If
TileTensor ever grows a way to pass the layout as scalar kernargs
(not a packed struct), revisit.
"""

from std.gpu import block_idx, thread_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.sys import size_of
from std.utils.index import StaticTuple

from _silu import _silu_f32


# Threads per block for the update kernel. Smaller than the 128 used by
# the full forward because update only touches `seqlen` outputs per
# (B, C) — extra parallelism across channels matters more.
comptime kNThreadsUpdate: Int = 64


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreadsUpdate))
)
def update_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    has_state_indices: Bool,
    is_circular: Bool,
](
    dim: Int32,
    seqlen: Int32,
    state_len: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_b_stride: Int32,
    x_c_stride: Int32,
    x_l_stride: Int32,
    w_c_stride: Int32,
    state_b_stride: Int32,
    state_c_stride: Int32,
    state_l_stride: Int32,
    o_b_stride: Int32,
    o_c_stride: Int32,
    o_l_stride: Int32,
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

    var batch_id: Int32 = Int32(block_idx.x)
    var channel_id: Int32 = Int32(block_idx.y) * Int32(kNThreadsUpdate) + Int32(
        thread_idx.x
    )
    if channel_id >= dim:
        return

    # Resolve the state-row coordinate. With has_state_indices=False this
    # is just batch_id; otherwise we look it up. A negative index marks
    # a padding token: zero the output and skip state mutation entirely.
    var state_batch_coord: Int32 = batch_id

    comptime if has_state_indices:
        var idx_val: Int32 = state_indices_ptr[Int(batch_id)]
        if idx_val < 0:
            var out_row = o_ptr + Int(
                batch_id * o_b_stride + channel_id * o_c_stride
            )
            for i in range(Int(seqlen)):
                out_row[i * Int(o_l_stride)] = Scalar[dtype](0)
            return
        state_batch_coord = idx_val

    # Base pointers for the (batch, channel) lane. Folding b_stride +
    # c_stride into a single offset lets the kernel index by `i *
    # l_stride` only in the per-position loops below.
    var x_lane = x_ptr + Int(batch_id * x_b_stride + channel_id * x_c_stride)
    var out_lane = o_ptr + Int(batch_id * o_b_stride + channel_id * o_c_stride)
    var state_lane = state_ptr + Int(
        state_batch_coord * state_b_stride + channel_id * state_c_stride
    )
    var w_lane = w_ptr + Int(channel_id * w_c_stride)

    # Force a single wide load for all `width` weight values of this
    # channel. amdgcn won't fold (width=4) `global_load_ushort`s into a
    # single `global_load_dwordx2` without a hand-typed vec load.
    var weights = SIMD[accum_t, width](0)
    comptime if width == 2 or width == 4:
        var w_vec = w_lane.load[width=width, alignment = size_of[dtype]() * width](0)
        comptime for k in range(width):
            weights[k] = w_vec[k].cast[accum_t]()
    else:
        comptime for k in range(width):
            weights[k] = w_lane.load(k).cast[accum_t]()

    var bias_v: Scalar[accum_t] = 0
    comptime if has_bias:
        bias_v = bias_ptr[Int(channel_id)].cast[accum_t]()

    var sl: Int32 = state_len
    var advance_len: Int32 = seqlen
    var x_vals = SIMD[accum_t, width](0)
    var update_idx: Int32 = 0

    comptime if is_circular:
        var cs: Int32 = cache_seqlens_ptr[Int(batch_id)] % sl
        update_idx = cs - Int32(width - 1)
        if update_idx < 0:
            update_idx += sl

        # Read W-1 history starting at update_idx, advancing with wrap.
        # After: update_idx = cache_seqlen % state_len → where x[0] writes.
        comptime for i in range(width - 1):
            var state_val = state_lane[Int(update_idx * state_l_stride)]
            x_vals[i] = state_val.cast[accum_t]()
            update_idx += 1
            if update_idx >= sl:
                update_idx -= sl
    else:
        # Phase 1 (linear): shift state left by `seqlen`.
        var n_shift: Int32 = sl - advance_len - Int32(width - 1)
        if n_shift > 0:
            var sj: Int32 = 0
            while True:
                state_lane[Int(sj * state_l_stride)] = state_lane[
                    Int((sj + advance_len) * state_l_stride)
                ]
                sj = sj + Int32(1)
                if not (sj < n_shift):
                    break

        # Phase 2 (linear): read trailing W-1 history into x_vals (with
        # writeback for the small-state_len edge case).
        var state_vals = SIMD[dtype, width - 1](0)
        comptime if width == 3 or width == 4:
            var s_vec = state_lane.load[width = width - 1, alignment=2](
                Int((sl - Int32(width - 1)) * state_l_stride)
            )
            comptime for i in range(width - 1):
                state_vals[i] = s_vec[i]
        else:
            comptime for i in range(width - 1):
                var read_idx: Int32 = sl - Int32(width - 1) + Int32(i)
                state_vals[i] = state_lane[Int(read_idx * state_l_stride)]

        comptime for i in range(width - 1):
            var write_idx: Int32 = sl - advance_len - Int32(width - 1) + Int32(i)
            if Int32(i) < advance_len + Int32(width - 1) and write_idx >= 0:
                state_lane[Int(write_idx * state_l_stride)] = state_vals[i]
            x_vals[i] = state_vals[i].cast[accum_t]()

    # Phase 3: walk new x, write into state, emit output.
    # The bench shapes always use seqlen=1 so the loop runs once. We
    # express it as a do-while-loop with int32 counter to avoid Mojo
    # promoting the loop index to i64.
    if advance_len < 1:
        return
    var i: Int32 = 0
    while True:
        var x_val = x_lane[Int(i * x_l_stride)]

        comptime if not is_circular:
            var write_idx: Int32 = sl - advance_len + i
            if i < advance_len and write_idx >= 0:
                state_lane[Int(write_idx * state_l_stride)] = x_val
        else:
            state_lane[Int(update_idx * state_l_stride)] = x_val
            update_idx += 1
            if update_idx >= sl:
                update_idx -= sl

        x_vals[width - 1] = x_val.cast[accum_t]()

        var out_val: Scalar[accum_t] = bias_v

        comptime for k in range(width):
            out_val += weights[k] * x_vals[k]

        comptime if apply_silu:
            out_val = _silu_f32(Float32(out_val))

        out_lane[Int(i * o_l_stride)] = out_val.cast[dtype]()

        # Slide x_vals left by 1 for the next output position.
        comptime for k in range(width - 1):
            x_vals[k] = x_vals[k + 1]

        i = i + Int32(1)
        if not (i < advance_len):
            break
