"""GPU forward kernel for flash_attn_func — naive online-softmax port.

Phase 2.1: this is the *correctness* version of the GPU forward. One
thread per (batch, head, q_idx) output row, walking the K dimension
sequentially. No tile loading, no shared memory, no tensor cores — just
the same online-softmax recurrence as `flash_fwd_cpu.mojo`, with global
loads from K and V on every iteration. Slow, but correct, and a place
to start.

Optimisation passes will follow:
  - phase 2.2: q-tile + k-tile blocking with shared-memory K/V cache.
  - phase 2.3: warp-level reduction across headdim for the dot products.
  - phase 2.4: tensor-core MMAs for the QK^T and PV matmuls.
  - phase 2.5: async copy / TMA + pipelined K/V loads.

The kernel signature mirrors `fwd_kernel_cpu` exactly so the launcher
can share argument-parsing code.
"""

from std.gpu import block_dim, block_idx_int as block_idx, thread_idx_int as thread_idx
from std.math import exp, inf, log, tanh
from std.memory import stack_allocation


alias MAX_HEADDIM = 256


fn fwd_kernel_gpu[
    dtype: DType,
](
    batch: Int,
    seqlen_q: Int,
    seqlen_k: Int,
    nheads_q: Int,
    nheads_kv: Int,
    headdim: Int,
    # Booleans are passed as Int to keep the kernel `DevicePassable`-friendly
    # (DeviceContext.enqueue_function rejects raw Bool args). 0 = False,
    # nonzero = True.
    causal_i: Int,
    softmax_scale: Float32,
    window_left: Int,
    window_right: Int,
    has_alibi_i: Int,
    alibi_b_stride: Int,
    alibi_ptr: UnsafePointer[Float32, MutAnyOrigin],
    has_dropout_i: Int,
    dropout_mask_ptr: UnsafePointer[Float32, MutAnyOrigin],
    has_cache_seqlens_i: Int,
    cache_seqlens_ptr: UnsafePointer[Int32, MutAnyOrigin],
    has_cache_batch_idx_i: Int,
    cache_batch_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    softcap: Float32,
    q_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    """flash-attn forward, GPU.

    Grid: (seqlen_q, nheads_q, batch). One thread per row, blockDim = 1.
    Each thread computes the online-softmax output for its (b, h_q, q_idx).
    """
    alias accum_t = DType.float32
    var neg_inf: Float32 = -inf[accum_t]()

    # Unpack Int → Bool flags up-front for readability.
    var causal: Bool = causal_i != 0
    var has_alibi: Bool = has_alibi_i != 0
    var has_dropout: Bool = has_dropout_i != 0
    var has_cache_seqlens: Bool = has_cache_seqlens_i != 0
    var has_cache_batch_idx: Bool = has_cache_batch_idx_i != 0

    var q_idx: Int = block_idx.x
    var h_q: Int = block_idx.y
    var b: Int = block_idx.z

    if q_idx >= seqlen_q or h_q >= nheads_q or b >= batch:
        return

    var heads_per_kv = nheads_q // nheads_kv
    var h_kv = h_q // heads_per_kv

    # KV batch index — same as q's `b` unless cache_batch_idx redirects.
    var b_kv: Int = b
    if has_cache_batch_idx:
        b_kv = Int(cache_batch_idx_ptr[b])

    var q_base = b * q_b_stride + q_idx * q_s_stride + h_q * q_h_stride
    var out_base = b * out_b_stride + q_idx * out_s_stride + h_q * out_h_stride
    var k_b_h_base = b_kv * k_b_stride + h_kv * k_h_stride
    var v_b_h_base = b_kv * v_b_stride + h_kv * v_h_stride

    # Per-thread scratch in registers (mojo lifts these into the local
    # frame; for headdim ≤ 256 that's 1 KB / buffer).
    var q_buf = stack_allocation[MAX_HEADDIM, accum_t]()
    var o_buf = stack_allocation[MAX_HEADDIM, accum_t]()

    for d in range(headdim):
        q_buf[d] = q_ptr[q_base + d * q_d_stride].cast[accum_t]()
        o_buf[d] = 0

    var alibi_slope: Float32 = 0
    if has_alibi:
        alibi_slope = alibi_ptr[b * alibi_b_stride + h_q]

    var m: Float32 = neg_inf
    var l: Float32 = 0

    var seqlen_k_eff: Int = seqlen_k
    if has_cache_seqlens:
        seqlen_k_eff = Int(cache_seqlens_ptr[b])
    var local_seq_offset = seqlen_k_eff - seqlen_q

    var kj_start: Int = 0
    var kj_end: Int = seqlen_k_eff
    var pos: Int = local_seq_offset + q_idx

    if causal:
        var k_max = pos
        if k_max < 0:
            k_max = -1
        if k_max + 1 < kj_end:
            kj_end = k_max + 1
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

    var lse_idx = (b * nheads_q + h_q) * seqlen_q + q_idx

    if kj_start >= kj_end:
        for d in range(headdim):
            out_ptr[out_base + d * out_d_stride] = Scalar[dtype](0)
        lse_ptr[lse_idx] = neg_inf
        return

    for kj in range(kj_start, kj_end):
        var k_base = k_b_h_base + kj * k_s_stride
        var v_base = v_b_h_base + kj * v_s_stride

        var score: Float32 = 0
        for d in range(headdim):
            score += (
                q_buf[d] * k_ptr[k_base + d * k_d_stride].cast[accum_t]()
            )
        score *= softmax_scale
        if softcap > 0:
            score = softcap * tanh(score / softcap)
        if has_alibi:
            var dist = pos - kj
            if dist < 0:
                dist = -dist
            score -= alibi_slope * Float32(dist)

        var m_new = max(m, score)
        var alpha = exp(m - m_new)
        var p = exp(score - m_new)
        l = alpha * l + p

        var mask_weight: Float32 = 1
        if has_dropout:
            var mask_idx = (
                (b * nheads_q + h_q) * seqlen_q + q_idx
            ) * seqlen_k + kj
            mask_weight = dropout_mask_ptr[mask_idx]

        var p_eff = p * mask_weight
        for d in range(headdim):
            var v_d = v_ptr[v_base + d * v_d_stride].cast[accum_t]()
            o_buf[d] = alpha * o_buf[d] + p_eff * v_d
        m = m_new

    var inv_l: Float32 = 1.0 / l
    for d in range(headdim):
        out_ptr[out_base + d * out_d_stride] = (
            o_buf[d] * inv_l
        ).cast[dtype]()
    lse_ptr[lse_idx] = m + log(l)
