"""Pure-mojo CPU forward kernel for flash_attn_func.

No upstream analogue — this exists so the package works on a
GPU-less machine. Naive online-softmax implementation, parallelised
over (batch, head, q_position). Performance is not the goal here;
the GPU kernel in `flash_fwd.mojo` is the real product.

Algorithm — standard "FlashAttention" online softmax recurrence:

    m, l, o = -inf, 0, zeros(D)
    for kj in range(k_max + 1):
        s = (q . k_j) * softmax_scale
        m_new = max(m, s)
        alpha = exp(m - m_new)
        p     = exp(s - m_new)
        l     = alpha * l + p
        o     = alpha * o + p * v_j
        m     = m_new
    out = o / l
    lse = m + log(l)            # log-sum-exp; needed by backward

`k_max` is `seqlen_k - 1` for non-causal, or
`(seqlen_k - seqlen_q) + q_idx` for causal — bottom-right alignment,
matching upstream `flash_attn_func`. If `k_max < 0` (only possible
when `seqlen_k < seqlen_q` with causal), the row attends to nothing
and the output is zero.

The first valid iteration (m = -inf) handles cleanly:
m_new = s, alpha = exp(-inf - s) = 0, l = 1, o = v_first, m = s.
"""

from std.algorithm import sync_parallelize
from std.math import exp, inf, log, tanh


fn fwd_kernel_cpu[
    dtype: DType,
    headdim: Int,
    causal: Bool,
](
    batch: Int,
    seqlen_q: Int,
    seqlen_k: Int,
    nheads_q: Int,
    nheads_kv: Int,
    softmax_scale: Float32,
    # Sliding-window bounds (raw upstream values: -1 means "infinite"
    # on that side, 0+ means a finite window of that many tokens).
    window_left: Int,
    window_right: Int,
    # ALiBi bias: bias_ij = -alibi_slope[b,h] * |pos - j|. NULL ptr
    # (and zero strides) disables the bias.
    has_alibi: Bool,
    alibi_b_stride: Int,
    alibi_ptr: UnsafePointer[Float32, MutAnyOrigin],
    # Dropout: when has_dropout is False, mask_ptr is unused. When True,
    # mask is fp32 of shape (batch, nheads_q, seqlen_q, seqlen_k), each
    # element ∈ {0, 1/(1-p)} (already scale-baked, so the kernel just
    # multiplies in).
    has_dropout: Bool,
    dropout_mask_ptr: UnsafePointer[Float32, MutAnyOrigin],
    # Per-batch effective k length for the KV-cache path. When
    # has_cache_seqlens is False, the kernel uses the global seqlen_k.
    # When True, batch element b only attends over k positions
    # [0, cache_seqlens_ptr[b]).
    has_cache_seqlens: Bool,
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
    # Logit softcap (Gemma2-style). Zero disables; positive c replaces
    # `score` with `c * tanh(score / c)` before alibi/mask/softmax.
    softcap: Float32,
    q_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    # lse_ptr is fp32 of shape (batch, nheads_q, seqlen_q), contiguous.
    lse_ptr: UnsafePointer[Float32, MutAnyOrigin],
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
    """flash-attn forward, CPU path.

    Comptime params:
        dtype:   element type (currently only fp16)
        headdim: per-head dimension (currently 64, 96, or 128)
        causal:  apply causal mask with bottom-right alignment

    Tensor layout (matches upstream `flash_attn_func`):
        q   : (batch, seqlen_q, nheads_q,  headdim)
        k, v: (batch, seqlen_k, nheads_kv, headdim)
        out : (batch, seqlen_q, nheads_q,  headdim)
    For MQA/GQA, `nheads_q % nheads_kv == 0` and a kv head is shared
    by `nheads_q / nheads_kv` consecutive q heads. Strides are passed
    explicitly so non-contiguous tensors work too.

    Parallelised across (batch, head, q_position) workers via
    `sync_parallelize` — every output position is computed
    independently in fp32 accumulators with online softmax.
    """
    alias accum_t = DType.float32
    var neg_inf: Float32 = -inf[accum_t]()

    var heads_per_kv = nheads_q // nheads_kv

    @parameter
    fn process_bhq(idx: Int):
        # Decompose idx into (b, h_q, q_idx). The work axis is
        # batch * nheads_q * seqlen_q, one worker per output row.
        var b = idx // (nheads_q * seqlen_q)
        var rem = idx % (nheads_q * seqlen_q)
        var h_q = rem // seqlen_q
        var q_idx = rem % seqlen_q
        # GQA: each KV head is shared by `heads_per_kv` consecutive Q heads.
        var h_kv = h_q // heads_per_kv

        var q_base = b * q_b_stride + q_idx * q_s_stride + h_q * q_h_stride
        var out_base = (
            b * out_b_stride + q_idx * out_s_stride + h_q * out_h_stride
        )
        var k_b_h_base = b * k_b_stride + h_kv * k_h_stride
        var v_b_h_base = b * v_b_stride + h_kv * v_h_stride

        # Load q vector into fp32 registers once.
        var q_vec = SIMD[accum_t, headdim](0)

        @parameter
        for d in range(headdim):
            q_vec[d] = q_ptr[q_base + d * q_d_stride].cast[accum_t]()

        # ALiBi slope for this (b, h_q) row.
        var alibi_slope: Float32 = 0
        if has_alibi:
            alibi_slope = alibi_ptr[b * alibi_b_stride + h_q]

        # Online softmax state.
        var m: Float32 = neg_inf
        var l: Float32 = 0
        var o = SIMD[accum_t, headdim](0)

        # Per-batch effective k length: full seqlen_k by default, or
        # cache_seqlens[b] in the kvcache path.
        var seqlen_k_eff: Int = seqlen_k
        if has_cache_seqlens:
            seqlen_k_eff = Int(cache_seqlens_ptr[b])
        var local_seq_offset = seqlen_k_eff - seqlen_q

        # Half-open k range for this query row: [kj_start, kj_end).
        var kj_start: Int = 0
        var kj_end: Int = seqlen_k_eff
        var pos: Int = local_seq_offset + q_idx  # bottom-right query position

        @parameter
        if causal:
            var k_max = pos
            if k_max < 0:
                k_max = -1  # ensures empty range below
            if k_max + 1 < kj_end:
                kj_end = k_max + 1

        # Sliding window bounds (skipped at runtime when both are -1).
        if window_left >= 0:
            var lo = pos - window_left
            if lo > kj_start:
                kj_start = lo
        if window_right >= 0:
            var hi = pos + window_right + 1
            if hi < kj_end:
                kj_end = hi
        if kj_start < 0:
            kj_start = 0
        if kj_end > seqlen_k_eff:
            kj_end = seqlen_k_eff

        # lse output index: contiguous (b, h_q, q_idx).
        var lse_idx = (b * nheads_q + h_q) * seqlen_q + q_idx

        if kj_start >= kj_end:
            # Row attends to nothing — write zeros and lse=-inf so the
            # backward sees zero gradient through this row.
            @parameter
            for d in range(headdim):
                out_ptr[out_base + d * out_d_stride] = Scalar[dtype](0)
            lse_ptr[lse_idx] = neg_inf
            return

        for kj in range(kj_start, kj_end):
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
            if softcap > 0:
                score = softcap * tanh(score / softcap)
            if has_alibi:
                # bias = -slope * |pos - kj|, distance is non-negative.
                var dist = pos - kj
                if dist < 0:
                    dist = -dist
                score -= alibi_slope * Float32(dist)

            var m_new = max(m, score)
            var alpha = exp(m - m_new)
            var p = exp(score - m_new)
            l = alpha * l + p

            # Dropout: read pre-scaled mask weight; non-dropout path reuses 1.
            var mask_weight: Float32 = 1
            if has_dropout:
                var mask_idx = (
                    (b * nheads_q + h_q) * seqlen_q + q_idx
                ) * seqlen_k + kj
                mask_weight = dropout_mask_ptr[mask_idx]

            # o = alpha * o + (mask * p) * v_j  (vectorised over D)
            var p_eff = p * mask_weight

            @parameter
            for d in range(headdim):
                var v_d = v_ptr[v_base + d * v_d_stride].cast[accum_t]()
                o[d] = alpha * o[d] + p_eff * v_d
            m = m_new

        # Final normalise + writeback. l is guaranteed > 0 because
        # exp(0) = 1 was added on at least one iteration (seqlen_k >= 1).
        var inv_l: Float32 = 1.0 / l

        @parameter
        for d in range(headdim):
            out_ptr[out_base + d * out_d_stride] = (o[d] * inv_l).cast[dtype]()

        # lse = m + log(l)  — the log-sum-exp of un-shifted scores.
        lse_ptr[lse_idx] = m + log(l)

    sync_parallelize[process_bhq](batch * nheads_q * seqlen_q)
