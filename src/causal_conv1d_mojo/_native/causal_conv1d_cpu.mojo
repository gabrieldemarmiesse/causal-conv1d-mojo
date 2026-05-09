"""Pure-mojo CPU forward + backward for causal_conv1d.

No upstream analogue — this exists so the package works on a machine
without a GPU without forcing users to `pip install causal-conv1d`
(which needs a C++ toolchain to source-build). These are the slow
path; the GPU kernels in `causal_conv1d_fwd.mojo` /
`causal_conv1d_bwd.mojo` are the real product.

Pattern follows max/kernels/src/state_space/causal_conv1d.mojo:
parallelise over (batch, channel) work items via `sync_parallelize`.
Each worker pre-loads its row of weights into a register, then walks
seqlen.
"""

from std.algorithm import sync_parallelize
from std.math import exp
from std.os.atomic import Atomic, Consistency

from causal_conv1d_common import _silu_f32


@always_inline
fn _cpu_dpre_at[
    dtype: DType,
    width: Int,
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
) -> Float32:
    """`dpre[t]` for the CPU backward, 0 if `t` is out of [0, seqlen).

    With `apply_silu`, `dpre[t] = silu'(pre[t]) * dout[t]` (the bias-aware
    sigmoid-derivative path). Without it, `dpre[t] = dout[t]` directly —
    bias-only forward has identity gradient w.r.t. pre.
    """
    if t < 0 or t >= seqlen:
        return 0

    @parameter
    if not apply_silu:
        return dout_ptr[dout_base + t * dout_l_stride].cast[DType.float32]()

    var pre: Float32 = bias_v

    @parameter
    for k in range(width):
        var src_t = t + k - (width - 1)
        if src_t >= 0:
            pre += (
                weights[k]
                * x_ptr[x_base + src_t * x_l_stride].cast[DType.float32]()
            )
    var sig: Float32 = 1.0 / (1.0 + exp(-pre))
    var silu_grad: Float32 = sig * (1.0 + pre * (1.0 - sig))
    var dout_v = dout_ptr[dout_base + t * dout_l_stride].cast[DType.float32]()
    return dout_v * silu_grad


fn fwd_kernel_cpu[
    dtype: DType,
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
    """Causal conv1d forward, CPU path.

    Comptime params:
        has_bias: load `bias_ptr[d]` per channel, or skip and use 0.
        has_seq_idx: gate historical reads on `seq_idx[b, src_t] ==
            seq_idx[b, t]`; force output to 0 when `seq_idx[b, t] < 0`
            (padding).
        apply_silu: apply silu (= swish) on the output, or skip.
    When the gate is False, the corresponding pointer is never
    dereferenced — caller may pass null from the Python wrapper.
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
        var out_base = b * out_batch_stride + d * out_c_stride
        var seq_idx_base: Int = b * seq_idx_b_stride
        var initial_states_base: Int = (
            b * initial_states_b_stride + d * initial_states_c_stride
        )

        for t in range(seqlen):
            var pre: Scalar[accum_t] = bias_v

            var cur_id: Int32 = 0

            @parameter
            if has_seq_idx:
                cur_id = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]

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
                            * x_ptr[x_base + src_t * x_l_stride].cast[accum_t]()
                        )
                else:

                    @parameter
                    if has_initial_states:
                        # src_t in [-(W-1), 0); index 0..W-2 of initial_states.
                        var is_idx: Int = src_t + (width - 1)
                        pre += (
                            weights[k]
                            * initial_states_ptr[
                                initial_states_base
                                + is_idx * initial_states_l_stride
                            ].cast[accum_t]()
                        )

            var out_v: Scalar[accum_t]

            @parameter
            if apply_silu:
                out_v = _silu_f32(pre)
            else:
                out_v = pre

            @parameter
            if has_seq_idx:
                if cur_id < 0:
                    out_v = 0

            output_ptr[out_base + t * out_l_stride] = out_v.cast[dtype]()

    sync_parallelize[process_bc](batch * dim)


fn bwd_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    dx_batch_stride: Int,
    dx_c_stride: Int,
    dx_l_stride: Int,
):
    """Causal conv1d backward, CPU path. Comptime: dtype, width, has_bias, apply_silu.

    Computes `dx, dweight, dbias` from `x, weight, bias, dout`. Uses a
    sliding window of `width` dpre values to avoid materialising the full
    `dpre` tensor:

        dx[t]       = sum_k weight[W-1-k] * dpre[t + k]
        dweight[w] += sum_t x[t + w - (W-1)] * dpre[t]
        dbias      += sum_t dpre[t]

    where `dpre[t] = silu'(pre[t]) * dout[t]`.

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

        # Sliding window: dpre_win[k] = dpre[t + k]. Prefill with dpre[0..W-1].
        var dpre_win = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            dpre_win[k] = _cpu_dpre_at[dtype, width, apply_silu](
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
            )

        var local_dweight = SIMD[accum_t, width](0)
        var local_dbias: Scalar[accum_t] = 0

        for t in range(seqlen):
            # dx[t] = sum_k weights[W-1-k] * dpre_win[k]
            var dx_v: Scalar[accum_t] = 0

            @parameter
            for k in range(width):
                dx_v += weights[width - 1 - k] * dpre_win[k]
            dx_ptr[dx_base + t * dx_l_stride] = dx_v.cast[dtype]()

            # dweight[k] += dpre[t] * x[t + k - (W-1)];  dbias += dpre[t]
            var dpre_t: Scalar[accum_t] = dpre_win[0]

            @parameter
            if has_bias:
                local_dbias += dpre_t

            @parameter
            for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    var x_v = x_ptr[x_base + src_t * x_l_stride].cast[accum_t]()
                    local_dweight[k] += dpre_t * x_v

            # Slide window left, append dpre[t + W] (or 0 past seqlen).
            @parameter
            for k in range(width - 1):
                dpre_win[k] = dpre_win[k + 1]
            dpre_win[width - 1] = _cpu_dpre_at[dtype, width, apply_silu](
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
