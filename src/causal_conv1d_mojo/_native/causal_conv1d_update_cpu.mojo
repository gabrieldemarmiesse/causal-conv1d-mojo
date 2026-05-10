"""Pure-mojo CPU single-step update kernel for causal_conv1d.

No upstream analogue — this exists so the package's autoregressive-
decode path works on a machine without a GPU. The real product is the
GPU kernel in `causal_conv1d_update.mojo`.

Algorithm matches the GPU kernel: depending on `is_circular`, either
shift `conv_state` left by `seqlen` and write at the tail, or treat
state as a circular buffer with `cache_seqlens` as the per-batch
write head. `has_state_indices` redirects the state row via
`state_indices[b]`; a negative index zeroes the output for that batch.
"""

from std.algorithm import sync_parallelize
from layout import TileTensor, TensorLayout

from causal_conv1d_common import _silu_f32


fn update_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
    has_state_indices: Bool,
    is_circular: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    SLayoutType: TensorLayout,
    OLayoutType: TensorLayout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    state_len: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    conv_state: TileTensor[mut=True, dtype, SLayoutType, MutAnyOrigin],
    state_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin],
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[mut=True, dtype, SLayoutType, MutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin].flat_rank == 3
):
    comptime accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var state_batch_coord: Int = b

        comptime if has_state_indices:
            var idx_val: Int = Int(state_indices_ptr[b])
            if idx_val < 0:
                for i in range(seqlen):
                    output[b, d, i] = Scalar[dtype](0)
                return
            state_batch_coord = idx_val

        var weights = SIMD[accum_t, width](0)

        comptime for k in range(width):
            weights[k] = weight[d, k].cast[accum_t]()

        var bias_v: Scalar[accum_t] = 0

        comptime if has_bias:
            bias_v = bias_ptr[d].cast[accum_t]()

        var update_idx: Int = 0

        comptime if is_circular:
            var cs: Int = Int(cache_seqlens_ptr[b]) % state_len
            update_idx = cs - (width - 1)
            if update_idx < 0:
                update_idx += state_len

        var advance_len = seqlen
        var x_vals = SIMD[accum_t, width](0)

        comptime if not is_circular:
            # Phase 1 (linear): shift state left by `seqlen`.
            for i in range(state_len - advance_len - (width - 1)):
                conv_state[state_batch_coord, d, i] = conv_state[
                    state_batch_coord, d, i + advance_len
                ]

            # Phase 2 (linear): read trailing W-1 history (with writeback
            # for the small-state_len edge case).
            comptime for i in range(width - 1):
                var read_idx: Int = state_len - (width - 1) + i
                var state_val = conv_state[state_batch_coord, d, read_idx]
                var write_idx: Int = state_len - advance_len - (width - 1) + i
                if i < advance_len + (width - 1) and write_idx >= 0:
                    conv_state[state_batch_coord, d, write_idx] = state_val
                x_vals[i] = state_val.cast[accum_t]()
        else:
            # Circular: read W-1 history starting at update_idx (mod state_len).
            comptime for i in range(width - 1):
                var state_val = conv_state[state_batch_coord, d, update_idx]
                x_vals[i] = state_val.cast[accum_t]()
                update_idx += 1
                if update_idx >= state_len:
                    update_idx -= state_len

        # Phase 3: walk new x.
        for i in range(seqlen):
            var x_val = x[b, d, i]

            comptime if not is_circular:
                var write_idx: Int = state_len - advance_len + i
                if i < advance_len and write_idx >= 0:
                    conv_state[state_batch_coord, d, write_idx] = x_val
            else:
                conv_state[state_batch_coord, d, update_idx] = x_val
                update_idx += 1
                if update_idx >= state_len:
                    update_idx -= state_len

            x_vals[width - 1] = x_val.cast[accum_t]()

            var out_val: Scalar[accum_t] = bias_v

            comptime for k in range(width):
                out_val += weights[k] * x_vals[k]

            comptime if apply_silu:
                out_val = _silu_f32(Float32(out_val))

            output[b, d, i] = out_val.cast[dtype]()

            comptime for k in range(width - 1):
                x_vals[k] = x_vals[k + 1]

    sync_parallelize[process_bc](batch * dim)
