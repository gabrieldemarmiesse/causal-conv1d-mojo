"""Pure-mojo CPU single-step update kernel for causal_conv1d.

No upstream analogue — this exists so the package's autoregressive-
decode path works on a machine without a GPU. The real product is the
GPU kernel under `update/`.
Input/state/output use `dtype`; weight/bias use `wdtype`, with parameter
loads converted to fp32 before the FMA chain.

Algorithm matches the GPU kernel: depending on `is_circular`, either
shift `conv_state` left by `seqlen` and write at the tail, or treat
state as a circular buffer with `cache_seqlens` as the per-batch
write head. `has_state_indices` redirects the state row via
`state_indices[b]`; a negative index zeroes the output for that batch.

The per-row work is a handful of scalar ops (decode is seqlen=1..few),
so per-task dispatch overhead dominates if every (b, d) row is its own
task — rows are dealt to at most `8 * num_logical_cores()` contiguous
row-chunks instead (see fwd_cpu).

The conv accumulates with the exact fma chain the fwd kernel uses
(ascending-k, f32, `_silu_f32` on top), keeping decode bit-identical
to the one-shot forward — tests pin that at zero tolerance for
fp16/bf16.
"""

from max.algorithm import sync_parallelize
from std.bit import next_power_of_two
from std.sys.info import num_logical_cores

from _silu import _silu_f32

comptime TASKS_PER_CORE = 8


def update_kernel_cpu[
    dtype: DType,
    wdtype: DType,
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
    x_bs: Int,
    x_cs: Int,
    x_ls: Int,
    w_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    w_cs: Int,
    w_ws: Int,
    bias_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_bs: Int,
    state_cs: Int,
    state_ls: Int,
    state_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_bs: Int,
    o_cs: Int,
    o_ls: Int,
):
    comptime accum_t = DType.float32

    var n_rows = batch * dim
    var n_tasks = min(n_rows, TASKS_PER_CORE * num_logical_cores())
    var rows_per_task = (n_rows + n_tasks - 1) // n_tasks

    @parameter
    def process_chunk(task_idx: Int):
        var row_lo = task_idx * rows_per_task
        var row_hi = min(row_lo + rows_per_task, n_rows)
        for bc_idx in range(row_lo, row_hi):
            var b = bc_idx // dim
            var d = bc_idx % dim

            var out_row = o_ptr + (b * o_bs + d * o_cs)
            var state_batch_coord: Int = b

            comptime if has_state_indices:
                var idx_val: Int = Int(state_indices_ptr[b])
                if idx_val < 0:
                    # Padding token: zero output, leave state untouched.
                    for i in range(seqlen):
                        out_row[i * o_ls] = Scalar[dtype](0)
                    continue
                state_batch_coord = idx_val

            var x_row = x_ptr + (b * x_bs + d * x_cs)
            var state_row = state_ptr + (
                state_batch_coord * state_bs + d * state_cs
            )

            var weights = SIMD[accum_t, next_power_of_two(width)](0)

            comptime for k in range(width):
                weights[k] = w_ptr[d * w_cs + k * w_ws].cast[accum_t]()

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
            var x_vals = SIMD[accum_t, next_power_of_two(width)](0)

            comptime if not is_circular:
                # Phase 1 (linear): shift state left by `seqlen`.
                for i in range(state_len - advance_len - (width - 1)):
                    state_row[i * state_ls] = state_row[
                        (i + advance_len) * state_ls
                    ]

                # Phase 2 (linear): read trailing W-1 history (with
                # writeback for the small-state_len edge case).
                comptime for i in range(width - 1):
                    var read_idx: Int = state_len - (width - 1) + i
                    var state_val = state_row[read_idx * state_ls]
                    var write_idx: Int = (
                        state_len - advance_len - (width - 1) + i
                    )
                    if i < advance_len + (width - 1) and write_idx >= 0:
                        state_row[write_idx * state_ls] = state_val
                    x_vals[i] = state_val.cast[accum_t]()
            else:
                # Circular: read W-1 history at update_idx (mod state_len).
                comptime for i in range(width - 1):
                    var state_val = state_row[update_idx * state_ls]
                    x_vals[i] = state_val.cast[accum_t]()
                    update_idx += 1
                    if update_idx >= state_len:
                        update_idx -= state_len

            # Phase 3: walk new x.
            for i in range(seqlen):
                var x_val = x_row[i * x_ls]

                comptime if not is_circular:
                    var write_idx: Int = state_len - advance_len + i
                    if i < advance_len and write_idx >= 0:
                        state_row[write_idx * state_ls] = x_val
                else:
                    state_row[update_idx * state_ls] = x_val
                    update_idx += 1
                    if update_idx >= state_len:
                        update_idx -= state_len

                x_vals[width - 1] = x_val.cast[accum_t]()

                # Same fma chain as the fwd kernel (bit-exact contract).
                var out_val: Scalar[accum_t] = bias_v

                comptime for k in range(width):
                    out_val = x_vals[k].fma(weights[k], out_val)

                comptime if apply_silu:
                    out_val = _silu_f32(Float32(out_val))

                out_row[i * o_ls] = out_val.cast[dtype]()

                comptime for k in range(width - 1):
                    x_vals[k] = x_vals[k + 1]

    sync_parallelize[process_chunk](n_tasks)
