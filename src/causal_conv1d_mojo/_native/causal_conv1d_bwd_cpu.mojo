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
from std.os.atomic import Atomic, Consistency


@always_inline
fn _cpu_dpre_at[
    dtype: DType,
    width: Int,
    has_seq_idx: Bool,
    apply_silu: Bool,
](
    t: Int,
    seqlen: Int,
    bias_v: Float32,
    weights: SIMD[DType.float32, width],
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_base: Int,
    x_l_stride: Int,
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_base: Int,
    dout_l_stride: Int,
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    seq_idx_base: Int,
    seq_idx_l_stride: Int,
) -> Float32:
    """`dpre[t]` for the CPU backward, 0 if `t` is out of [0, seqlen).

    With `apply_silu`, `dpre[t] = silu'(pre[t]) * dout[t]` (the bias-aware
    sigmoid-derivative path). Without it, `dpre[t] = dout[t]` directly —
    bias-only forward has identity gradient w.r.t. pre.

    With `has_seq_idx`, returns 0 if `seq_idx[t] < 0` (padding token whose
    output was forced to 0 in the forward); the silu_grad recomputation
    of `pre[t]` masks historical x reads on
    `seq_idx[src_t] == seq_idx[t]` to mirror the forward gate.
    """
    if t < 0 or t >= seqlen:
        return 0

    var cur_id: Int32 = 0

    @parameter
    if has_seq_idx:
        cur_id = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]
        if cur_id < 0:
            # Padding: forward forced out=0, so dpre is zero too.
            return 0

    @parameter
    if not apply_silu:
        return dout_ptr[dout_base + t * dout_l_stride].cast[DType.float32]()

    var pre: Float32 = bias_v

    @parameter
    for k in range(width):
        var src_t = t + k - (width - 1)
        if src_t >= 0:
            var include: Bool = True

            @parameter
            if has_seq_idx:
                var src_id: Int32 = seq_idx_ptr[
                    seq_idx_base + src_t * seq_idx_l_stride
                ]
                include = src_id == cur_id
            if include:
                pre += (
                    weights[k]
                    * x_ptr[x_base + src_t * x_l_stride].cast[DType.float32]()
                )
    var sig: Float32 = 1.0 / (1.0 + exp(-pre))
    var silu_grad: Float32 = sig * (1.0 + pre * (1.0 - sig))
    var dout_v = dout_ptr[dout_base + t * dout_l_stride].cast[DType.float32]()
    return dout_v * silu_grad


fn bwd_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    dx_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    dout_batch_stride: Int,
    dout_c_stride: Int,
    dout_l_stride: Int,
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    dx_batch_stride: Int,
    dx_c_stride: Int,
    dx_l_stride: Int,
):
    """Causal conv1d backward, CPU path.

    With `has_seq_idx`, gates dpre, dx, and dweight contributions by
    sequence-id equality (mirroring the forward gate). dbias's per-token
    sum is unchanged because dpre is already zero for padding tokens.

    Parallelised across (batch, channel) workers via `sync_parallelize`.
    Workers may share a `d` (across batches) so the per-channel
    `dweight` / `dbias` accumulators are atomic-added at the end.
    """
    alias accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var bias_v: Scalar[accum_t] = 0

        @parameter
        if has_bias:
            bias_v = bias_ptr[d].cast[accum_t]()

        var weights = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            weights[k] = weight_ptr[
                d * weight_c_stride + k * weight_w_stride
            ].cast[accum_t]()

        var x_base = b * x_batch_stride + d * x_c_stride
        var dout_base = b * dout_batch_stride + d * dout_c_stride
        var dx_base = b * dx_batch_stride + d * dx_c_stride
        var seq_idx_base: Int = b * seq_idx_b_stride

        # Sliding window: dpre_win[k] = dpre[t + k]. Prefill with dpre[0..W-1].
        var dpre_win = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            dpre_win[k] = _cpu_dpre_at[dtype, width, has_seq_idx, apply_silu](
                k,
                seqlen,
                bias_v,
                weights,
                x_ptr,
                x_base,
                x_l_stride,
                dout_ptr,
                dout_base,
                dout_l_stride,
                seq_idx_ptr,
                seq_idx_base,
                seq_idx_l_stride,
            )

        var local_dweight = SIMD[accum_t, width](0)
        var local_dbias: Scalar[accum_t] = 0

        for t in range(seqlen):
            var cur_id_t: Int32 = 0

            @parameter
            if has_seq_idx:
                cur_id_t = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]

            # dx[t] = sum_k weights[W-1-k] * dpre_win[k]
            # With seq_idx: each term is gated on
            # `seq_idx[t] == seq_idx[t+k]` (forward used x[t] in position
            # t+k's conv only when ids matched).
            var dx_v: Scalar[accum_t] = 0

            @parameter
            for k in range(width):
                var include: Bool = True

                @parameter
                if has_seq_idx:
                    var pos_k = t + k
                    if pos_k < seqlen:
                        var sid: Int32 = seq_idx_ptr[
                            seq_idx_base + pos_k * seq_idx_l_stride
                        ]
                        include = sid == cur_id_t
                    else:
                        include = False
                if include:
                    dx_v += weights[width - 1 - k] * dpre_win[k]
            dx_ptr[dx_base + t * dx_l_stride] = dx_v.cast[dtype]()

            # dweight[k] += dpre[t] * x[t + k - (W-1)];  dbias += dpre[t]
            # dpre_win[0] = dpre[t]; already zero if seq_idx[t] < 0.
            # For dweight, additionally gate on
            # `seq_idx[src_t] == seq_idx[t]` (= cur_id_t).
            var dpre_t: Scalar[accum_t] = dpre_win[0]

            @parameter
            if has_bias:
                local_dbias += dpre_t

            @parameter
            for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    var include: Bool = True

                    @parameter
                    if has_seq_idx:
                        var sid: Int32 = seq_idx_ptr[
                            seq_idx_base + src_t * seq_idx_l_stride
                        ]
                        include = sid == cur_id_t
                    if include:
                        var x_v = x_ptr[x_base + src_t * x_l_stride].cast[
                            accum_t
                        ]()
                        local_dweight[k] += dpre_t * x_v

            # Slide window left, append dpre[t + W] (or 0 past seqlen).
            @parameter
            for k in range(width - 1):
                dpre_win[k] = dpre_win[k + 1]
            dpre_win[width - 1] = _cpu_dpre_at[
                dtype, width, has_seq_idx, apply_silu
            ](
                t + width,
                seqlen,
                bias_v,
                weights,
                x_ptr,
                x_base,
                x_l_stride,
                dout_ptr,
                dout_base,
                dout_l_stride,
                seq_idx_ptr,
                seq_idx_base,
                seq_idx_l_stride,
            )

        # Atomic-add the (b, d) block's contribution. Multiple parallel
        # workers may target the same `d` across different batches.
        @parameter
        for k in range(width):
            _ = Atomic[DType.float32].fetch_add[ordering=Consistency.MONOTONIC](
                dweight_acc_ptr + d * width + k, local_dweight[k]
            )

        @parameter
        if has_bias:
            _ = Atomic[DType.float32].fetch_add[ordering=Consistency.MONOTONIC](
                dbias_acc_ptr + d, local_dbias
            )

    sync_parallelize[process_bc](batch * dim)
