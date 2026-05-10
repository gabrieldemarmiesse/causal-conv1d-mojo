"""Pure-mojo CPU forward kernel for flash_attn_func.

No upstream analogue — this exists so the package works on a
GPU-less machine. Naive online-softmax implementation, parallelised
over (batch, head, q_position). Performance is not the goal here;
the GPU kernel in `flash_fwd.mojo` is the real product.

Algorithm — standard "FlashAttention" online softmax recurrence:

    m, l, o = -inf, 0, zeros(D)
    for kj in range(seqlen_k):
        s = (q . k_j) * softmax_scale
        m_new = max(m, s)
        alpha = exp(m - m_new)
        p     = exp(s - m_new)
        l     = alpha * l + p
        o     = alpha * o + p * v_j
        m     = m_new
    out = o / l

The first iteration (m = -inf, l = 0, o = 0) handles cleanly:
m_new = s, alpha = exp(-inf - s) = 0, l = 1, o = v_0, m = s.
"""

from std.algorithm import sync_parallelize
from std.math import exp, inf


fn fwd_kernel_cpu[
    dtype: DType,
    headdim: Int,
](
    batch: Int,
    seqlen_q: Int,
    seqlen_k: Int,
    nheads: Int,
    softmax_scale: Float32,
    q_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    q_b_stride: Int,
    q_s_stride: Int,
    q_h_stride: Int,
    q_d_stride: Int,
    k_b_stride: Int,
    k_s_stride: Int,
    k_h_stride: Int,
    k_d_stride: Int,
    v_b_stride: Int,
    v_s_stride: Int,
    v_h_stride: Int,
    v_d_stride: Int,
    out_b_stride: Int,
    out_s_stride: Int,
    out_h_stride: Int,
    out_d_stride: Int,
):
    """Causal conv1d... no wait, flash-attn forward, CPU path.

    Comptime params:
        dtype:   element type (only fp16 in phase 1.1)
        headdim: per-head dimension (only 64 in phase 1.1)

    Tensor layout (matches upstream `flash_attn_func`): all of q, k, v,
    out are `(batch, seqlen, nheads, headdim)`. Strides are passed
    explicitly so non-contiguous tensors work too.

    Parallelised across (batch, head, q_position) workers via
    `sync_parallelize` — every output position is computed
    independently in fp32 accumulators with online softmax.
    """
    alias accum_t = DType.float32
    var neg_inf: Float32 = -inf[accum_t]()

    @parameter
    fn process_bhq(idx: Int):
        # Decompose idx into (b, h, q_idx). The work axis is the
        # combined size batch * nheads * seqlen_q so each worker is one
        # output row.
        var b = idx // (nheads * seqlen_q)
        var rem = idx % (nheads * seqlen_q)
        var h = rem // seqlen_q
        var q_idx = rem % seqlen_q

        var q_base = b * q_b_stride + q_idx * q_s_stride + h * q_h_stride
        var out_base = (
            b * out_b_stride + q_idx * out_s_stride + h * out_h_stride
        )
        var k_b_h_base = b * k_b_stride + h * k_h_stride
        var v_b_h_base = b * v_b_stride + h * v_h_stride

        # Load q vector into fp32 registers once.
        var q_vec = SIMD[accum_t, headdim](0)

        @parameter
        for d in range(headdim):
            q_vec[d] = q_ptr[q_base + d * q_d_stride].cast[accum_t]()

        # Online softmax state.
        var m: Float32 = neg_inf
        var l: Float32 = 0
        var o = SIMD[accum_t, headdim](0)

        for kj in range(seqlen_k):
            var k_base = k_b_h_base + kj * k_s_stride
            var v_base = v_b_h_base + kj * v_s_stride

            # score = (q . k_j) * scale
            var score: Scalar[accum_t] = 0

            @parameter
            for d in range(headdim):
                score += (
                    q_vec[d] * k_ptr[k_base + d * k_d_stride].cast[accum_t]()
                )
            score *= softmax_scale

            var m_new = max(m, score)
            var alpha = exp(m - m_new)
            var p = exp(score - m_new)
            l = alpha * l + p

            # o = alpha * o + p * v_j  (per-element, vectorised over D)
            @parameter
            for d in range(headdim):
                var v_d = v_ptr[v_base + d * v_d_stride].cast[accum_t]()
                o[d] = alpha * o[d] + p * v_d
            m = m_new

        # Final normalise + writeback. l is guaranteed > 0 because
        # exp(0) = 1 was added on at least one iteration (seqlen_k >= 1).
        var inv_l: Float32 = 1.0 / l

        @parameter
        for d in range(headdim):
            out_ptr[out_base + d * out_d_stride] = (o[d] * inv_l).cast[dtype]()

    sync_parallelize[process_bhq](batch * nheads * seqlen_q)
