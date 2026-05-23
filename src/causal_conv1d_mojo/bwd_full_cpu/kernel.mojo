"""Pure-mojo CPU fused backward for causal_conv1d.

No upstream analogue — this exists so the package works on a machine
without a GPU. The GPU kernels in `causal_conv1d_fwd.mojo` /
`causal_conv1d_bwd.mojo` are the real product; the CPU paths are the
slow fallback.

Computes `dx, dweight, dbias` from `x, weight, bias, dout`. Uses a
sliding window of `width` dpre values to avoid materialising the full
`dpre` tensor:

    dx[t]       = sum_k weight[W-1-k] * dpre[t + k]
    dweight[w] += sum_t x[t + w - (W-1)] * dpre[t]
    dbias      += sum_t dpre[t]

where `dpre[t] = silu'(pre[t]) * dout[t]`.
"""

from std.algorithm import sync_parallelize
from std.math import exp
from std.atomic import Atomic, Ordering
from layout import TileTensor, TensorLayout


@always_inline
def _cpu_dpre_at[
    dtype: DType,
    width: Int,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    XLayoutType: TensorLayout,
    DoutLayoutType: TensorLayout,
    SLayoutType: TensorLayout,
    ILayoutType: TensorLayout,
](
    t: Int,
    b: Int,
    d: Int,
    seqlen: Int,
    bias_v: Float32,
    weights: SIMD[DType.float32, width],
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    dout: TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin],
    seq_idx: TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin],
    initial_states: TileTensor[dtype, ILayoutType, ImmutAnyOrigin],
) -> Float32 where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, ILayoutType, ImmutAnyOrigin].flat_rank == 3
):
    """`dpre[t]` for the CPU backward, 0 if `t` is out of [0, seqlen).

    With `apply_silu`, `dpre[t] = silu'(pre[t]) * dout[t]` (the bias-aware
    sigmoid-derivative path). Without it, `dpre[t] = dout[t]` directly —
    bias-only forward has identity gradient w.r.t. pre.

    With `has_seq_idx`, returns 0 if `seq_idx[t] < 0` (padding token whose
    output was forced to 0 in the forward); the silu_grad recomputation
    of `pre[t]` masks historical x reads on
    `seq_idx[src_t] == seq_idx[t]` to mirror the forward gate.

    With `has_initial_states`, the silu_grad pre recomputation reads
    `initial_states[src_t + W - 1]` for `src_t ∈ [-(W-1), 0)` instead of
    treating those positions as zero (mirrors the forward).
    """
    if t < 0 or t >= seqlen:
        return 0

    var cur_id: Int32 = 0

    comptime if has_seq_idx:
        cur_id = seq_idx[b, t]
        if cur_id < 0:
            # Padding: forward forced out=0, so dpre is zero too.
            return 0

    comptime if not apply_silu:
        return dout[b, d, t].cast[DType.float32]()

    var pre: Float32 = bias_v

    comptime for k in range(width):
        var src_t = t + k - (width - 1)
        if src_t >= 0:
            var include: Bool = True

            comptime if has_seq_idx:
                var src_id: Int32 = seq_idx[b, src_t]
                include = src_id == cur_id
            if include:
                pre += weights[k] * x[b, d, src_t].cast[DType.float32]()
        else:

            comptime if has_initial_states:
                var is_idx: Int = src_t + (width - 1)
                pre += (
                    weights[k]
                    * initial_states[b, d, is_idx].cast[DType.float32]()
                )
    var sig: Float32 = 1.0 / (1.0 + exp(-pre))
    var silu_grad: Float32 = sig * (1.0 + pre * (1.0 - sig))
    var dout_v = dout[b, d, t].cast[DType.float32]()
    return dout_v * silu_grad


def bwd_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    DoutLayoutType: TensorLayout,
    DxLayoutType: TensorLayout,
    SLayoutType: TensorLayout,
    ILayoutType: TensorLayout,
    DILayoutType: TensorLayout,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout: TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin],
    seq_idx: TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin],
    initial_states: TileTensor[dtype, ILayoutType, ImmutAnyOrigin],
    dx: TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dinitial_states: TileTensor[
        mut=True, dtype, DILayoutType, MutAnyOrigin
    ],
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, DoutLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, DxLayoutType, MutAnyOrigin].flat_rank == 3
    and TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[dtype, ILayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[mut=True, dtype, DILayoutType, MutAnyOrigin].flat_rank == 3
):
    """Causal conv1d backward, CPU path.

    With `has_seq_idx`, gates dpre, dx, and dweight contributions by
    sequence-id equality (mirroring the forward gate). dbias's per-token
    sum is unchanged because dpre is already zero for padding tokens.

    Parallelised across (batch, channel) workers via `sync_parallelize`.
    Workers may share a `d` (across batches) so the per-channel
    `dweight` / `dbias` accumulators are atomic-added at the end.
    """
    comptime accum_t = DType.float32

    @parameter
    def process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var bias_v: Scalar[accum_t] = 0

        comptime if has_bias:
            bias_v = bias_ptr[d].cast[accum_t]()

        var weights = SIMD[accum_t, width](0)

        comptime for k in range(width):
            weights[k] = weight[d, k].cast[accum_t]()

        # Sliding window: dpre_win[k] = dpre[t + k]. Prefill with dpre[0..W-1].
        var dpre_win = SIMD[accum_t, width](0)

        comptime for k in range(width):
            dpre_win[k] = _cpu_dpre_at[
                dtype,
                width,
                has_seq_idx,
                has_initial_states,
                apply_silu,
                XLayoutType,
                DoutLayoutType,
                SLayoutType,
                ILayoutType,
            ](
                k,
                b,
                d,
                seqlen,
                bias_v,
                weights,
                x,
                dout,
                seq_idx,
                initial_states,
            )

        var local_dweight = SIMD[accum_t, width](0)
        var local_dbias: Scalar[accum_t] = 0

        # ---- initial_states-only contributions ----
        # dpre_win[0..W-2] = dpre[0..W-2] right now (the prefill); these
        # are exactly the dpre values that flow into dinitial_states and
        # the "boundary" dweight terms (where the forward conv read
        # initial_states instead of x). Compute both before the main
        # loop slides the window away.
        comptime if has_initial_states:
            # dinit[i] = sum_{k=0..i} weight[k] * dpre[i - k]   for i in [0, W-1)
            comptime for i in range(width - 1):
                var dinit_v: Scalar[accum_t] = 0

                comptime for k in range(width):

                    comptime if i - k >= 0:
                        dinit_v += weights[k] * dpre_win[i - k]
                dinitial_states[b, d, i] = dinit_v.cast[dtype]()

            # dweight[k] += sum_{t=0..W-2-k} dpre[t] * initial_states[t + k]
            # — the part of the conv that read initial_states in the forward.
            comptime for k in range(width):

                comptime for t in range(width - 1 - k):
                    var is_v = initial_states[b, d, t + k].cast[accum_t]()
                    local_dweight[k] += dpre_win[t] * is_v

        for t in range(seqlen):
            var cur_id_t: Int32 = 0

            comptime if has_seq_idx:
                cur_id_t = seq_idx[b, t]

            # dx[t] = sum_k weights[W-1-k] * dpre_win[k]
            # With seq_idx: each term is gated on
            # `seq_idx[t] == seq_idx[t+k]` (forward used x[t] in position
            # t+k's conv only when ids matched).
            var dx_v: Scalar[accum_t] = 0

            comptime for k in range(width):
                var include: Bool = True

                comptime if has_seq_idx:
                    var pos_k = t + k
                    if pos_k < seqlen:
                        var sid: Int32 = seq_idx[b, pos_k]
                        include = sid == cur_id_t
                    else:
                        include = False
                if include:
                    dx_v += weights[width - 1 - k] * dpre_win[k]
            dx[b, d, t] = dx_v.cast[dtype]()

            # dweight[k] += dpre[t] * x[t + k - (W-1)];  dbias += dpre[t]
            # dpre_win[0] = dpre[t]; already zero if seq_idx[t] < 0.
            # For dweight, additionally gate on
            # `seq_idx[src_t] == seq_idx[t]` (= cur_id_t).
            var dpre_t: Scalar[accum_t] = dpre_win[0]

            comptime if has_bias:
                local_dbias += dpre_t

            comptime for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    var include: Bool = True

                    comptime if has_seq_idx:
                        var sid: Int32 = seq_idx[b, src_t]
                        include = sid == cur_id_t
                    if include:
                        var x_v = x[b, d, src_t].cast[accum_t]()
                        local_dweight[k] += dpre_t * x_v

            # Slide window left, append dpre[t + W] (or 0 past seqlen).
            comptime for k in range(width - 1):
                dpre_win[k] = dpre_win[k + 1]
            dpre_win[width - 1] = _cpu_dpre_at[
                dtype,
                width,
                has_seq_idx,
                has_initial_states,
                apply_silu,
                XLayoutType,
                DoutLayoutType,
                SLayoutType,
                ILayoutType,
            ](
                t + width,
                b,
                d,
                seqlen,
                bias_v,
                weights,
                x,
                dout,
                seq_idx,
                initial_states,
            )

        # Atomic-add the (b, d) block's contribution. Multiple parallel
        # workers may target the same `d` across different batches.
        comptime for k in range(width):
            _ = Atomic[DType.float32].fetch_add[ordering=Ordering.RELAXED](
                dweight_acc_ptr + d * width + k, local_dweight[k]
            )

        comptime if has_bias:
            _ = Atomic[DType.float32].fetch_add[ordering=Ordering.RELAXED](
                dbias_acc_ptr + d, local_dbias
            )

    sync_parallelize[process_bc](batch * dim)
