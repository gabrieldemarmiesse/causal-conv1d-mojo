"""Pure-mojo CPU fused backward for causal_conv1d.

No upstream analogue — this exists so the package works on a machine
without a GPU. The GPU kernels under `bwd_full/` are the real product;
this is the fallback — but a fast one.
Input/state/dout/dx use `dtype`; weight/bias use `wdtype`, and parameter
loads convert to fp32 before arithmetic.

Computes `dx, dweight, dbias, dinitial_states` from
`x, weight, bias, dout`:

    dpre[t]     = dout[t] * silu'(pre[t])   (or dout[t] when no silu)
    dx[t]       = sum_k weight[W-1-k] * dpre[t + k]
    dweight[k] += sum_t x[t + k - (W-1)] * dpre[t]
    dbias      += sum_t dpre[t]

Fast path (no seq_idx, x/dout/dx unit-stride along t): per row, walk
seqlen in kChunk-sized pieces with a small stack buffer of dpre:

  * pass A (vectorized, kV lanes/step): recompute `pre` from the x tap
    loads, form `dpre`, store it to the buffer — and fold the
    dweight/dbias partial sums in the same step, reusing the x taps
    already in registers (dweight is nearly free);
  * pass B (vectorized): `dx` from the buffered dpre window.

The buffer extends `W-1` past the chunk (recomputed next chunk) so
pass B never crosses a chunk seam; positions past seqlen buffer as 0.
The first `W-1` positions of the row are handled scalar (history
reaches initial_states / zeros), as is the last partial vector.
When seq_idx and initial_states are both present, the virtual initial
positions carry `seq_idx[b, 0]`; pre recomputation, boundary dweight,
and dinitial_states all apply that id gate so later packed sequences
cannot consume or differentiate the state.

Rows with seq_idx or non-unit strides take the scalar sliding-window
path (same recurrence, one dpre window in registers).

Parallelised over (batch*dim) rows dealt to at most
`8 * num_logical_cores()` contiguous row-chunks (see fwd_cpu). Workers
may share a `d` across batches, so per-row dweight/dbias partials are
flushed with relaxed atomics by default. Deterministic variants instead
plain-store each row's partial into a private batch-major fp32 workspace;
Python reduces it over batch in a fixed order.
"""

from max.algorithm import sync_parallelize
from std.bit import next_power_of_two
from std.atomic import Atomic, Ordering
from std.math import exp, recip
from std.sys import size_of
from std.sys.info import num_logical_cores

comptime TASKS_PER_CORE = 8
# dpre values buffered per chunk of the fast path (fp32; +width slack for
# the cross-seam extension). 512 keeps the buffer + row working set deep
# in L1 while amortizing the per-chunk scalar seams.
comptime kChunk = 512


@always_inline
def _silu_grad_v[w: Int](
    pre: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """d(silu)/dpre = sig * (1 + pre * (1 - sig)), elementwise."""
    var one = SIMD[DType.float32, w](1)
    var sig = recip(one + exp(-pre))
    return sig * (one + pre * (one - sig))


@always_inline
def _dpre_scalar[
    dtype: DType,
    width: Int,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
](
    t: Int,
    seqlen: Int,
    bias_v: Float32,
    weights: SIMD[DType.float32, next_power_of_two(width)],
    x_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_ls: Int,
    dout_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_ls: Int,
    si_row: UnsafePointer[Int32, MutAnyOrigin],
    si_ls: Int,
    init_row: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    is_ls: Int,
) -> Float32:
    """`dpre[t]`, 0 outside [0, seqlen). Scalar; mirrors the forward's
    boundary/seq_idx gating when recomputing `pre` for the silu grad."""
    if t < 0 or t >= seqlen:
        return 0

    var cur_id: Int32 = 0

    comptime if has_seq_idx:
        cur_id = si_row[t * si_ls]
        if cur_id < 0:
            # Padding: forward forced out=0, so dpre is zero too.
            return 0

    comptime if not apply_silu:
        return dout_row[t * dout_ls].cast[DType.float32]()

    var pre: Float32 = bias_v

    comptime for k in range(width):
        var src_t = t + k - (width - 1)
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
    var dout_v = dout_row[t * dout_ls].cast[DType.float32]()
    return dout_v * _silu_grad_v[1](pre)


def bwd_kernel_cpu[
    dtype: DType,
    wdtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    deterministic: Bool,
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
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_bs: Int,
    dout_cs: Int,
    dout_ls: Int,
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    si_bs: Int,
    si_ls: Int,
    init_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    is_bs: Int,
    is_cs: Int,
    is_ls: Int,
    dx_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dx_bs: Int,
    dx_cs: Int,
    dx_ls: Int,
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dinit_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dis_bs: Int,
    dis_cs: Int,
    dis_ls: Int,
):
    """Causal conv1d fused backward, CPU path.

    With `deterministic=False`, `dweight_acc_ptr` / `dbias_acc_ptr` are
    dense fp32 `(dim, width)` / `(dim,)` accumulators zero-initialised by
    the caller. With `deterministic=True`, they are dense batch-major
    `(batch, dim, width)` / `(batch, dim)` workspaces whose row-private
    slots are written with plain stores. Everything else is arbitrarily
    strided. `dinitial_states` is written iff `has_initial_states`.
    Pointers gated off by a comptime flag are never dereferenced.
    """
    comptime accum_t = DType.float32
    comptime kV = 32 // size_of[dtype]()

    var n_rows = batch * dim
    var n_tasks = min(n_rows, TASKS_PER_CORE * num_logical_cores())
    var rows_per_task = (n_rows + n_tasks - 1) // n_tasks

    @parameter
    def process_chunk(task_idx: Int):
        var row_lo = task_idx * rows_per_task
        var row_hi = min(row_lo + rows_per_task, n_rows)
        # dpre chunk buffer, reused across rows of this task (fast path).
        var dpre_buf = InlineArray[Float32, kChunk + width](fill=0)
        var buf = dpre_buf.unsafe_ptr()

        for bc_idx in range(row_lo, row_hi):
            var b = bc_idx // dim
            var d = bc_idx % dim

            var x_row = x_ptr + (b * x_bs + d * x_cs)
            var dout_row = dout_ptr + (b * dout_bs + d * dout_cs)
            var dx_row = dx_ptr + (b * dx_bs + d * dx_cs)
            # Only dereferenced when the matching comptime gate is on.
            var si_row = seq_idx_ptr + b * si_bs
            var init_row = init_ptr + (b * is_bs + d * is_cs)
            var dinit_row = dinit_ptr + (b * dis_bs + d * dis_cs)

            var bias_v: Scalar[accum_t] = 0

            comptime if has_bias:
                bias_v = bias_ptr[d].cast[accum_t]()

            var weights = SIMD[accum_t, next_power_of_two(width)](0)

            comptime for k in range(width):
                weights[k] = w_ptr[d * w_cs + k * w_ws].cast[accum_t]()

            var local_dweight = SIMD[accum_t, next_power_of_two(width)](0)
            var local_dbias: Scalar[accum_t] = 0

            # ---- initial_states contributions ----
            # Need dpre[0..W-2]: dinit[i] = sum_{k<=i} weight[k]*dpre[i-k],
            # and the boundary dweight terms where the forward read
            # initial_states instead of x. With seq_idx, both terms are
            # gated against seq_idx[b, 0], matching the virtual ids used
            # by forward even when the first fragment is shorter than W-1.
            comptime if has_initial_states:
                var dpre_head = SIMD[accum_t, next_power_of_two(width)](0)

                comptime for i in range(width - 1):
                    dpre_head[i] = _dpre_scalar[
                        dtype, width, has_seq_idx, has_initial_states,
                        apply_silu,
                    ](
                        i, seqlen, bias_v, weights, x_row, x_ls,
                        dout_row, dout_ls, si_row, si_ls, init_row, is_ls,
                    )

                comptime for i in range(width - 1):
                    var dinit_v: Scalar[accum_t] = 0

                    comptime for k in range(width):

                        comptime if i - k >= 0:

                            comptime if has_seq_idx:
                                if i - k < seqlen:
                                    if si_row[(i - k) * si_ls] == si_row[0]:
                                        dinit_v = dpre_head[i - k].fma(
                                            weights[k], dinit_v
                                        )
                            else:
                                dinit_v = dpre_head[i - k].fma(
                                    weights[k], dinit_v
                                )
                    dinit_row[i * dis_ls] = dinit_v.cast[dtype]()

                # dweight[k] += sum_{t=0..W-2-k} dpre[t]*initial_states[t+k]
                comptime for k in range(width):

                    comptime for t in range(width - 1 - k):
                        var is_v = init_row[(t + k) * is_ls].cast[accum_t]()

                        comptime if has_seq_idx:
                            if t < seqlen:
                                if si_row[t * si_ls] == si_row[0]:
                                    local_dweight[k] = dpre_head[t].fma(
                                        is_v, local_dweight[k]
                                    )
                        else:
                            local_dweight[k] = dpre_head[t].fma(
                                is_v, local_dweight[k]
                            )

            # ---- main loop over t ----
            var fast = x_ls == 1 and dout_ls == 1 and dx_ls == 1

            comptime if has_seq_idx:
                fast = False

            if fast:
                # dweight/dbias vector partial sums; reduced at row end.
                var dw_v = InlineArray[SIMD[accum_t, kV], width](
                    fill=SIMD[accum_t, kV](0)
                )
                var db_v = SIMD[accum_t, kV](0)

                var chunk_start = 0
                while chunk_start < seqlen:
                    var n = min(kChunk, seqlen - chunk_start)
                    var n_ext = n + width - 1

                    # -- pass A: fill dpre_buf[0..n_ext) = dpre[cs+j],
                    #    folding dweight/dbias for j < n as we go. --
                    var j = 0
                    # scalar head: t < W-1 (boundary taps) — first chunk
                    # only.
                    while j < n and chunk_start + j < width - 1:
                        var t = chunk_start + j
                        var dp = _dpre_scalar[
                            dtype, width, has_seq_idx,
                            has_initial_states, apply_silu,
                        ](
                            t, seqlen, bias_v, weights, x_row, x_ls,
                            dout_row, dout_ls, si_row, si_ls,
                            init_row, is_ls,
                        )
                        buf[j] = dp

                        comptime if has_bias:
                            local_dbias += dp

                        comptime for k in range(width):
                            var src_t = t + k - (width - 1)
                            if src_t >= 0:
                                local_dweight[k] = dp.fma(
                                    x_row[src_t].cast[accum_t](),
                                    local_dweight[k],
                                )
                        j += 1

                    # vector body over the in-chunk region [j, n).
                    while j + kV <= n:
                        var t = chunk_start + j
                        var dpre_v: SIMD[accum_t, kV]
                        # x taps stay live for the dweight fold below.
                        var xt = InlineArray[SIMD[accum_t, kV], width](
                            fill=SIMD[accum_t, kV](0)
                        )

                        comptime for k in range(width):
                            xt[k] = (
                                (x_row + (t + k - (width - 1)))
                                .load[width=kV]()
                                .cast[accum_t]()
                            )
                        var dout_v = (
                            (dout_row + t).load[width=kV]().cast[accum_t]()
                        )

                        comptime if apply_silu:
                            var pre = SIMD[accum_t, kV](bias_v)

                            comptime for k in range(width):
                                pre = xt[k].fma(
                                    SIMD[accum_t, kV](weights[k]), pre
                                )
                            dpre_v = dout_v * _silu_grad_v[kV](pre)
                        else:
                            dpre_v = dout_v

                        (buf + j).store(dpre_v)

                        comptime if has_bias:
                            db_v += dpre_v

                        comptime for k in range(width):
                            dw_v[k] = dpre_v.fma(xt[k], dw_v[k])
                        j += kV

                    # scalar tail of the in-chunk region [j, n): every tap
                    # is in-range (t >= W-1 here — the scalar head ended
                    # the boundary), so no gating needed beyond dpre's own.
                    while j < n:
                        var t = chunk_start + j
                        var dp = _dpre_scalar[
                            dtype, width, has_seq_idx,
                            has_initial_states, apply_silu,
                        ](
                            t, seqlen, bias_v, weights, x_row, x_ls,
                            dout_row, dout_ls, si_row, si_ls,
                            init_row, is_ls,
                        )
                        buf[j] = dp

                        comptime if has_bias:
                            local_dbias += dp

                        comptime for k in range(width):
                            local_dweight[k] = dp.fma(
                                x_row[t + k - (width - 1)].cast[accum_t](),
                                local_dweight[k],
                            )
                        j += 1

                    # extension [n, n_ext): buffer-only (next chunk's
                    # positions; their dweight/dbias fold happens there).
                    while j < n_ext:
                        buf[j] = _dpre_scalar[
                            dtype, width, has_seq_idx,
                            has_initial_states, apply_silu,
                        ](
                            chunk_start + j, seqlen, bias_v, weights,
                            x_row, x_ls, dout_row, dout_ls, si_row,
                            si_ls, init_row, is_ls,
                        )
                        j += 1

                    # -- pass B: dx[t] = sum_k w[W-1-k] * dpre[t+k]. --
                    var i = 0
                    while i + kV <= n:
                        var acc = SIMD[accum_t, kV](0)

                        comptime for k in range(width):
                            var dpv = (buf + (i + k)).load[width=kV]()
                            acc = dpv.fma(
                                SIMD[accum_t, kV](weights[width - 1 - k]),
                                acc,
                            )
                        (dx_row + (chunk_start + i)).store(
                            acc.cast[dtype]()
                        )
                        i += kV
                    while i < n:
                        var acc: Scalar[accum_t] = 0

                        comptime for k in range(width):
                            acc = buf[i + k].fma(
                                weights[width - 1 - k], acc
                            )
                        dx_row[chunk_start + i] = acc.cast[dtype]()
                        i += 1

                    chunk_start += n

                # reduce the vector partial sums into the row totals.
                comptime if has_bias:
                    local_dbias += db_v.reduce_add()

                comptime for k in range(width):
                    local_dweight[k] += dw_v[k].reduce_add()
            else:
                # ---- scalar sliding-window fallback (seq_idx and/or
                # non-unit strides). dpre_win[k] = dpre[t + k]. ----
                var dpre_win = SIMD[accum_t, next_power_of_two(width)](0)

                comptime for k in range(width):
                    dpre_win[k] = _dpre_scalar[
                        dtype, width, has_seq_idx, has_initial_states,
                        apply_silu,
                    ](
                        k, seqlen, bias_v, weights, x_row, x_ls,
                        dout_row, dout_ls, si_row, si_ls, init_row, is_ls,
                    )

                for t in range(seqlen):
                    var cur_id_t: Int32 = 0

                    comptime if has_seq_idx:
                        cur_id_t = si_row[t * si_ls]

                    # dx[t] = sum_k weights[W-1-k] * dpre_win[k], each
                    # term gated on seq_idx[t] == seq_idx[t+k].
                    var dx_v: Scalar[accum_t] = 0

                    comptime for k in range(width):
                        var include: Bool = True

                        comptime if has_seq_idx:
                            var pos_k = t + k
                            if pos_k < seqlen:
                                include = si_row[pos_k * si_ls] == cur_id_t
                            else:
                                include = False
                        if include:
                            dx_v = dpre_win[k].fma(
                                weights[width - 1 - k], dx_v
                            )
                    dx_row[t * dx_ls] = dx_v.cast[dtype]()

                    # dweight[k] += dpre[t] * x[t+k-(W-1)]; dbias += dpre[t]
                    var dpre_t: Scalar[accum_t] = dpre_win[0]

                    comptime if has_bias:
                        local_dbias += dpre_t

                    comptime for k in range(width):
                        var src_t = t + k - (width - 1)
                        if src_t >= 0:
                            var include: Bool = True

                            comptime if has_seq_idx:
                                include = si_row[src_t * si_ls] == cur_id_t
                            if include:
                                local_dweight[k] = dpre_t.fma(
                                    x_row[src_t * x_ls].cast[accum_t](),
                                    local_dweight[k],
                                )

                    # slide the window, append dpre[t + W] (0 past end).
                    comptime for k in range(width - 1):
                        dpre_win[k] = dpre_win[k + 1]
                    dpre_win[width - 1] = _dpre_scalar[
                        dtype, width, has_seq_idx, has_initial_states,
                        apply_silu,
                    ](
                        t + width, seqlen, bias_v, weights, x_row, x_ls,
                        dout_row, dout_ls, si_row, si_ls, init_row, is_ls,
                    )

            # Workers may share a `d` across batches. Deterministic rows
            # own `(b, d)` workspace slots; the default path accumulates
            # into the shared per-channel tensors with relaxed atomics.
            comptime if deterministic:
                var workspace_row = b * dim + d

                comptime for k in range(width):
                    dweight_acc_ptr[workspace_row * width + k] = local_dweight[k]

                comptime if has_bias:
                    dbias_acc_ptr[workspace_row] = local_dbias
            else:

                comptime for k in range(width):
                    _ = Atomic[DType.float32].fetch_add[ordering=Ordering.RELAXED](
                        dweight_acc_ptr + d * width + k, local_dweight[k]
                    )

                comptime if has_bias:
                    _ = Atomic[DType.float32].fetch_add[ordering=Ordering.RELAXED](
                        dbias_acc_ptr + d, local_dbias
                    )

    sync_parallelize[process_chunk](n_tasks)
