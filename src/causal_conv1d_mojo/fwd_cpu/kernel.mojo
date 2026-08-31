"""Pure-mojo CPU forward for causal_conv1d.

No upstream analogue — this exists so the package works on a machine
without a GPU without forcing users to `pip install causal-conv1d`
(which needs a C++ toolchain to source-build). The GPU kernels under
`fwd/` are the real product; this is the fallback — but a fast one.
Input/state/output use `dtype`; weight/bias use `wdtype`, with both
converted to fp32 for the existing accumulation chain.

Structure per (b, d) row:
  * boundary region `t < W-1`: scalar; history reaches back into
    `initial_states` (or implicit zeros), with seq_idx gating. When
    both are present, the virtual initial positions carry
    `seq_idx[b, 0]`, so only the first packed sequence sees them.
  * main region `t >= W-1`: every tap is in-range, so the tap loop is
    branch-free. When x/out are unit-stride along t and there is no
    seq_idx, the loop is explicitly vectorized: `kV` outputs per step,
    one unaligned vector load of x per tap — the W loads overlap and
    hit L1, so DRAM sees each element once. LLVM does not
    auto-vectorize the strided scalar form of this loop, and the
    difference is ~the whole memory-bandwidth roofline on wide CPUs.
    Strided / seq_idx rows fall back to the scalar tap loop.

Parallelism: `batch*dim` rows are dealt to at most
`8 * num_logical_cores()` contiguous row-chunks via `sync_parallelize`
(row-sized tasks would drown small-row shapes in per-task scheduling
overhead; 8x gives the scheduler slack to balance uneven chunks).
"""

from max.algorithm import sync_parallelize
from std.bit import next_power_of_two
from std.sys import size_of
from std.sys.info import num_logical_cores

from _silu import _silu_f32

# Row-chunks per core handed to sync_parallelize. Swept 1/2/4/8/16 on an
# M4 (10 cores): differences were within run-to-run noise, so keep the
# conventional 8 (enough slack for load balancing without drowning short
# rows in per-task dispatch overhead).
comptime TASKS_PER_CORE = 8


@always_inline
def _fwd_scalar_at[
    dtype: DType,
    width: Int,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    boundary: Bool,
](
    t: Int,
    bias_v: Float32,
    weights: SIMD[DType.float32, next_power_of_two(width)],
    x_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ls: Int,
    si_row: UnsafePointer[Int32, MutAnyOrigin],
    si_ls: Int,
    init_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    is_ls: Int,
    o_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ls: Int,
):
    """One output position, scalar. `boundary=True` keeps the
    `src_t < 0` handling (initial_states / implicit zeros); the main
    region compiles it out entirely.

    Accumulates with explicit `fma` in ascending-k order — the exact
    chain the vector main loop uses — so every output element is
    bit-identical no matter which path produced it (the update kernel
    relies on this: its decode loop must reproduce the fwd outputs
    exactly, and tests pin that at zero tolerance for fp16/bf16)."""
    var pre: Float32 = bias_v
    var cur_id: Int32 = 0

    comptime if has_seq_idx:
        cur_id = si_row[t * si_ls]

    comptime for k in range(width):
        var src_t = t + k - (width - 1)

        comptime if boundary:
            if src_t >= 0:
                var include: Bool = True

                comptime if has_seq_idx:
                    include = si_row[src_t * si_ls] == cur_id
                if include:
                    pre = (
                        x_row[src_t * x_ls]
                        .cast[DType.float32]()
                        .fma(weights[k], pre)
                    )
            else:

                comptime if has_initial_states:
                    # src_t in [-(W-1), 0); index 0..W-2 of initial_states.
                    var is_idx: Int = src_t + (width - 1)

                    comptime if has_seq_idx:
                        if si_row[0] == cur_id:
                            pre = (
                                init_row[is_idx * is_ls]
                                .cast[DType.float32]()
                                .fma(weights[k], pre)
                            )
                    else:
                        pre = (
                            init_row[is_idx * is_ls]
                            .cast[DType.float32]()
                            .fma(weights[k], pre)
                        )
        else:
            # Main region: src_t is always in [0, seqlen).
            var include: Bool = True

            comptime if has_seq_idx:
                include = si_row[src_t * si_ls] == cur_id
            if include:
                pre = (
                    x_row[src_t * x_ls]
                    .cast[DType.float32]()
                    .fma(weights[k], pre)
                )

    var out_v: Float32 = pre

    comptime if apply_silu:
        out_v = _silu_f32(pre)

    comptime if has_seq_idx:
        if cur_id < 0:
            out_v = 0

    o_row[t * o_ls] = out_v.cast[dtype]()


def fwd_kernel_cpu[
    dtype: DType,
    wdtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_bs: Int,
    x_cs: Int,
    x_ls: Int,
    w_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    w_cs: Int,
    w_ws: Int,
    bias_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    si_bs: Int,
    si_ls: Int,
    init_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    is_bs: Int,
    is_cs: Int,
    is_ls: Int,
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_bs: Int,
    o_cs: Int,
    o_ls: Int,
):
    """Causal conv1d forward, CPU path.

    Comptime params:
        has_bias: load `bias_ptr[d]` per channel, or skip and use 0.
        has_seq_idx: gate historical reads on `seq_idx[b, src_t] ==
            seq_idx[b, t]`; force output to 0 when `seq_idx[b, t] < 0`
            (padding). If initial_states is also present, its virtual
            positions use `seq_idx[b, 0]` for this gate.
        apply_silu: apply silu (= swish) on the output, or skip.
    When the gate is False, the corresponding pointer is never
    dereferenced — caller may pass null from the Python wrapper.
    Strides are in elements. `bias` is implicitly unit-stride.
    """
    comptime accum_t = DType.float32
    # Elements per vector step of the main loop: 32 bytes of x per tap
    # load (8 x f32 / 16 x f16-bf16).
    comptime kV = 32 // size_of[dtype]()

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

            var x_row = x_ptr + (b * x_bs + d * x_cs)
            var o_row = o_ptr + (b * o_bs + d * o_cs)
            # Only dereferenced when the matching comptime gate is on.
            var si_row = seq_idx_ptr + b * si_bs
            var init_row = init_ptr + (b * is_bs + d * is_cs)

            var bias_v: Scalar[accum_t] = 0

            comptime if has_bias:
                bias_v = bias_ptr[d].cast[accum_t]()

            var weights = SIMD[accum_t, next_power_of_two(width)](0)

            comptime for k in range(width):
                weights[k] = w_ptr[d * w_cs + k * w_ws].cast[accum_t]()

            # Boundary region: history reaches initial_states / zeros.
            var t0 = min(width - 1, seqlen)
            for t in range(t0):
                _fwd_scalar_at[
                    dtype,
                    width,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    True,
                ](
                    t,
                    bias_v,
                    weights,
                    x_row,
                    x_ls,
                    si_row,
                    si_ls,
                    init_row,
                    is_ls,
                    o_row,
                    o_ls,
                )

            # Main region: every tap in [0, seqlen).
            var t = t0

            comptime if not has_seq_idx:
                if x_ls == 1 and o_ls == 1:
                    while t + kV <= seqlen:
                        var acc = SIMD[accum_t, kV](bias_v)

                        comptime for k in range(width):
                            var xv = (
                                x_row + (t + k - (width - 1))
                            ).load[width=kV]()
                            acc = xv.cast[accum_t]().fma(
                                SIMD[accum_t, kV](weights[k]), acc
                            )

                        comptime if apply_silu:
                            acc = _silu_f32[kV](acc)
                        (o_row + t).store(acc.cast[dtype]())
                        t += kV

            for tt in range(t, seqlen):
                _fwd_scalar_at[
                    dtype,
                    width,
                    has_seq_idx,
                    has_initial_states,
                    apply_silu,
                    False,
                ](
                    tt,
                    bias_v,
                    weights,
                    x_row,
                    x_ls,
                    si_row,
                    si_ls,
                    init_row,
                    is_ls,
                    o_row,
                    o_ls,
                )

    sync_parallelize[process_chunk](n_tasks)
