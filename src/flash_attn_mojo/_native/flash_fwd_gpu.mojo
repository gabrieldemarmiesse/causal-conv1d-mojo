"""GPU forward kernel for flash_attn_func — block-cooperative on D axis.

Phase 2.2: same grid as 2.1 (one block per (B, H_q, S_q) output row),
but the block is 64 threads cooperating on the headdim axis. The big
wins over phase 2.1:

- Coalesced global loads. Each kj iteration reads the K row across
  64 threads issuing consecutive d-offsets in parallel — one (or
  few) DRAM transactions instead of `headdim` sequential ones.
- Parallel dot products. The `score = sum_d q[d] * k[d]` sum is a
  butterfly reduction across threads (5-step warp shuffle, plus a
  smem hop for the second warp).
- Parallel V accumulation. `o[d] = alpha*o[d] + p*v[d]` is split
  across threads — each thread owns its d-chunk in registers.

What still happens serially (per-block):
- The K dimension. Threads in a block iterate kj together; each
  iteration does one block-reduce + one mask-broadcast. The serial
  dependence on (m, l) over kj prevents parallelising kj here.

Phase 2.3 will add Q_TILE > 1 to amortize K/V loads across multiple
q rows in the same block. Phase 2.4 will switch to tensor-core MMAs.

The block-level fp32 sum + warp shuffle helpers are copied from
`causal_conv1d_bwd.mojo` (same butterfly pattern, broadcast-True so
all threads hold the result).
"""

from std.gpu import (
    block_idx_int as block_idx,
    thread_idx_int as thread_idx,
    barrier,
)
from std.gpu.memory import AddressSpace
from std.math import ceildiv, exp, inf, log, tanh
from std.memory import stack_allocation
from std.sys import llvm_intrinsic


alias MAX_HEADDIM = 256
alias BLOCK_DIM = 64
alias D_PER_THREAD = ceildiv(MAX_HEADDIM, BLOCK_DIM)  # = 4


@always_inline
fn _shfl_xor_f32(val: Float32, offset: UInt32) -> Float32:
    return llvm_intrinsic["llvm.nvvm.shfl.sync.bfly.f32", Float32](
        Int32(-1), val, offset, Int32(31)
    )


@always_inline
fn _warp_sum_f32(val: Float32) -> Float32:
    """5-step butterfly warp reduction, fp32; all lanes hold the sum."""
    var v = val
    v += _shfl_xor_f32(v, UInt32(16))
    v += _shfl_xor_f32(v, UInt32(8))
    v += _shfl_xor_f32(v, UInt32(4))
    v += _shfl_xor_f32(v, UInt32(2))
    v += _shfl_xor_f32(v, UInt32(1))
    return v


@always_inline
fn _block_sum_f32_broadcast[block_size: Int](val: Float32) -> Float32:
    """Block-level fp32 sum; *every* thread receives the result.

    Two-warp version (block_size=64): each warp reduces to its lane-0;
    the two partials go to a 2-element smem; warp 0 reads both,
    butterfly-reduces, writes the broadcast value back to smem[0];
    barrier; everyone reads smem[0].
    """
    constrained[
        block_size >= 32 and block_size % 32 == 0,
        "block_size must be a multiple of 32",
    ]()
    alias n_warps: Int = block_size // 32

    var warp_result = _warp_sum_f32(val)

    @parameter
    if n_warps == 1:
        return warp_result

    var tid: Int = thread_idx.x
    var lane: Int = tid & 31
    var warp: Int = tid >> 5

    # Use n_warps + 1 slots: n_warps for per-warp partials, 1 for the
    # final broadcast value (separates write/read, simplifies the
    # broadcast step).
    var smem = stack_allocation[
        n_warps + 1, DType.float32, address_space=AddressSpace.SHARED
    ]()
    if lane == 0:
        smem[warp] = warp_result
    barrier()

    if warp == 0:
        var v: Float32 = 0
        if lane < n_warps:
            v = smem[lane]
        v = _warp_sum_f32(v)
        if lane == 0:
            smem[n_warps] = v
    barrier()

    return smem[n_warps]


fn fwd_kernel_gpu[
    dtype: DType,
](
    batch: Int,
    seqlen_q: Int,
    seqlen_k: Int,
    nheads_q: Int,
    nheads_kv: Int,
    headdim: Int,
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
    """flash-attn forward, GPU. Grid (S_q, H_q, B) × block BLOCK_DIM."""
    alias accum_t = DType.float32
    var neg_inf: Float32 = -inf[accum_t]()

    var causal: Bool = causal_i != 0
    var has_alibi: Bool = has_alibi_i != 0
    var has_dropout: Bool = has_dropout_i != 0
    var has_cache_seqlens: Bool = has_cache_seqlens_i != 0
    var has_cache_batch_idx: Bool = has_cache_batch_idx_i != 0

    var q_idx: Int = block_idx.x
    var h_q: Int = block_idx.y
    var b: Int = block_idx.z
    var t: Int = thread_idx.x

    if q_idx >= seqlen_q or h_q >= nheads_q or b >= batch:
        return

    var heads_per_kv = nheads_q // nheads_kv
    var h_kv = h_q // heads_per_kv

    var b_kv: Int = b
    if has_cache_batch_idx:
        b_kv = Int(cache_batch_idx_ptr[b])

    var q_base = b * q_b_stride + q_idx * q_s_stride + h_q * q_h_stride
    var out_base = b * out_b_stride + q_idx * out_s_stride + h_q * out_h_stride
    var k_b_h_base = b_kv * k_b_stride + h_kv * k_h_stride
    var v_b_h_base = b_kv * v_b_stride + h_kv * v_h_stride

    # Per-thread d-range: contiguous chunk [t*D_PER_THREAD, ...).
    var d_start: Int = t * D_PER_THREAD

    # Load this thread's q chunk + init o chunk (both fp32, in registers).
    var q_local = SIMD[accum_t, D_PER_THREAD](0)
    var o_local = SIMD[accum_t, D_PER_THREAD](0)

    @parameter
    for di in range(D_PER_THREAD):
        var d = d_start + di
        if d < headdim:
            q_local[di] = q_ptr[q_base + d * q_d_stride].cast[accum_t]()

    # ALiBi slope (every thread loads — same address, L1 hit after first).
    var alibi_slope: Float32 = 0
    if has_alibi:
        alibi_slope = alibi_ptr[b * alibi_b_stride + h_q]

    # Online softmax state — every thread keeps the same value.
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
        # Row attends to nothing — write zeros for this thread's chunk;
        # thread 0 writes lse=-inf.
        @parameter
        for di in range(D_PER_THREAD):
            var d = d_start + di
            if d < headdim:
                out_ptr[out_base + d * out_d_stride] = Scalar[dtype](0)
        if t == 0:
            lse_ptr[lse_idx] = neg_inf
        return

    for kj in range(kj_start, kj_end):
        var k_base = k_b_h_base + kj * k_s_stride
        var v_base = v_b_h_base + kj * v_s_stride

        # Each thread computes its partial dot product. Threads with
        # `d_start + di >= headdim` contribute 0.
        var partial: Float32 = 0

        @parameter
        for di in range(D_PER_THREAD):
            var d = d_start + di
            if d < headdim:
                partial += (
                    q_local[di]
                    * k_ptr[k_base + d * k_d_stride].cast[accum_t]()
                )

        # Block reduction with broadcast — all threads receive `score`.
        var score = _block_sum_f32_broadcast[BLOCK_DIM](partial)
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

        @parameter
        for di in range(D_PER_THREAD):
            var d = d_start + di
            if d < headdim:
                var v_d = v_ptr[v_base + d * v_d_stride].cast[accum_t]()
                o_local[di] = alpha * o_local[di] + p_eff * v_d
        m = m_new

    var inv_l: Float32 = 1.0 / l

    @parameter
    for di in range(D_PER_THREAD):
        var d = d_start + di
        if d < headdim:
            out_ptr[out_base + d * out_d_stride] = (
                o_local[di] * inv_l
            ).cast[dtype]()

    if t == 0:
        lse_ptr[lse_idx] = m + log(l)
