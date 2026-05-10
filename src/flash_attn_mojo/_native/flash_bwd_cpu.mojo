"""Pure-mojo CPU backward kernel for flash_attn_func.

Two-pass implementation (recompute approach — same trick the GPU
backward uses):

  Pass A — parallelise over (b, h_q, q_idx). For each q row:
      - D[i] = dO[i] · O[i]                (used by both passes)
      - For each k_j:
          s_j = (q_i · k_j) * scale
          P_j = exp(s_j - lse[i])
          dP_j = dO[i] · V[j]
          dS_j = P_j * (dP_j - D[i])
          dQ[i] += dS_j * K[j] * scale

  Pass B — parallelise over (b, h_kv, k_idx). For each k row, sum
  over the q_heads sharing this kv head and over all valid q
  positions:
      For each (h_q in group, q_idx) with k_idx in row's valid range:
          recompute s, P, dP, dS as above
          dV[j] += P_j * dO[i]
          dK[j] += dS_j * Q[i] * scale

Both passes recompute the score matrix from saved (q, k, lse). This
costs 2× the forward FLOPs but avoids materialising the (S_q × S_k)
attention matrix.

The two passes write to disjoint output tensors (dQ in A, dK/dV in B),
so they can run sequentially with no atomics. D[i] is recomputed in
each pass — it's just one D-dim dot product per row, cheap.
"""

from std.algorithm import sync_parallelize
from std.math import exp, inf


fn bwd_kernel_cpu[
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
    q_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dq_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
    dout_b_stride: Int,
    dout_s_stride: Int,
    dout_h_stride: Int,
    dout_d_stride: Int,
    dq_b_stride: Int,
    dq_s_stride: Int,
    dq_h_stride: Int,
    dq_d_stride: Int,
    dk_b_stride: Int,
    dk_s_stride: Int,
    dk_h_stride: Int,
    dk_d_stride: Int,
    dv_b_stride: Int,
    dv_s_stride: Int,
    dv_h_stride: Int,
    dv_d_stride: Int,
):
    """flash-attn backward, CPU path. See module docstring for math.

    All stride arguments mirror their forward counterparts. dQ/dK/dV
    are output buffers, expected to be zero-initialised by the caller
    (we write — never accumulate — so this is just the convention).
    """
    alias accum_t = DType.float32
    var neg_inf: Float32 = -inf[accum_t]()
    var seq_offset = seqlen_k - seqlen_q
    var heads_per_kv = nheads_q // nheads_kv

    # ---- Pass A: dQ ----
    @parameter
    fn pass_a(idx: Int):
        var b = idx // (nheads_q * seqlen_q)
        var rem = idx % (nheads_q * seqlen_q)
        var h_q = rem // seqlen_q
        var q_idx = rem % seqlen_q
        var h_kv = h_q // heads_per_kv

        var q_base = b * q_b_stride + q_idx * q_s_stride + h_q * q_h_stride
        var dq_base = b * dq_b_stride + q_idx * dq_s_stride + h_q * dq_h_stride
        var o_base = (
            b * out_b_stride + q_idx * out_s_stride + h_q * out_h_stride
        )
        var do_base = (
            b * dout_b_stride + q_idx * dout_s_stride + h_q * dout_h_stride
        )
        var k_b_h_base = b * k_b_stride + h_kv * k_h_stride
        var v_b_h_base = b * v_b_stride + h_kv * v_h_stride
        var lse_idx = (b * nheads_q + h_q) * seqlen_q + q_idx

        var lse = lse_ptr[lse_idx]

        # k_max bound for this row.
        var kj_end: Int = seqlen_k

        @parameter
        if causal:
            var k_max = seq_offset + q_idx
            if k_max < 0:
                # Row attends to nothing — dQ stays zero.
                @parameter
                for d in range(headdim):
                    dq_ptr[dq_base + d * dq_d_stride] = Scalar[dtype](0)
                return
            kj_end = k_max + 1
            if kj_end > seqlen_k:
                kj_end = seqlen_k

        # Load q, dO, O into fp32 registers once.
        var q_vec = SIMD[accum_t, headdim](0)
        var do_vec = SIMD[accum_t, headdim](0)
        var o_vec = SIMD[accum_t, headdim](0)

        @parameter
        for d in range(headdim):
            q_vec[d] = q_ptr[q_base + d * q_d_stride].cast[accum_t]()
            do_vec[d] = dout_ptr[do_base + d * dout_d_stride].cast[accum_t]()
            o_vec[d] = out_ptr[o_base + d * out_d_stride].cast[accum_t]()

        # D[i] = dO[i] · O[i]
        var D_i: Float32 = 0

        @parameter
        for d in range(headdim):
            D_i += do_vec[d] * o_vec[d]

        # Accumulate dQ row.
        var dq_acc = SIMD[accum_t, headdim](0)

        for kj in range(kj_end):
            var k_base = k_b_h_base + kj * k_s_stride
            var v_base = v_b_h_base + kj * v_s_stride

            var s: Float32 = 0

            @parameter
            for d in range(headdim):
                s += (
                    q_vec[d] * k_ptr[k_base + d * k_d_stride].cast[accum_t]()
                )
            s *= softmax_scale
            var p = exp(s - lse)

            # dP_j = dO · V_j
            var dp: Float32 = 0

            @parameter
            for d in range(headdim):
                dp += (
                    do_vec[d] * v_ptr[v_base + d * v_d_stride].cast[accum_t]()
                )

            var ds = p * (dp - D_i)

            @parameter
            for d in range(headdim):
                dq_acc[d] += (
                    ds
                    * k_ptr[k_base + d * k_d_stride].cast[accum_t]()
                    * softmax_scale
                )

        @parameter
        for d in range(headdim):
            dq_ptr[dq_base + d * dq_d_stride] = dq_acc[d].cast[dtype]()

    sync_parallelize[pass_a](batch * nheads_q * seqlen_q)

    # ---- Pass B: dK and dV ----
    @parameter
    fn pass_b(idx: Int):
        var b = idx // (nheads_kv * seqlen_k)
        var rem = idx % (nheads_kv * seqlen_k)
        var h_kv = rem // seqlen_k
        var k_idx = rem % seqlen_k

        var k_base = b * k_b_stride + k_idx * k_s_stride + h_kv * k_h_stride
        var v_base = b * v_b_stride + k_idx * v_s_stride + h_kv * v_h_stride
        var dk_base = (
            b * dk_b_stride + k_idx * dk_s_stride + h_kv * dk_h_stride
        )
        var dv_base = (
            b * dv_b_stride + k_idx * dv_s_stride + h_kv * dv_h_stride
        )

        # Load k_j and v_j once.
        var k_vec = SIMD[accum_t, headdim](0)
        var v_vec = SIMD[accum_t, headdim](0)

        @parameter
        for d in range(headdim):
            k_vec[d] = k_ptr[k_base + d * k_d_stride].cast[accum_t]()
            v_vec[d] = v_ptr[v_base + d * v_d_stride].cast[accum_t]()

        var dk_acc = SIMD[accum_t, headdim](0)
        var dv_acc = SIMD[accum_t, headdim](0)

        # Sweep over all q heads sharing this kv head, then all q positions.
        for h_off in range(heads_per_kv):
            var h_q = h_kv * heads_per_kv + h_off

            for q_idx in range(seqlen_q):
                # Skip rows masked out by causal.
                @parameter
                if causal:
                    var k_max = seq_offset + q_idx
                    if k_idx > k_max:
                        continue
                    if k_max < 0:
                        # Row contributes nothing for this q (and all earlier).
                        continue

                var q_base = (
                    b * q_b_stride + q_idx * q_s_stride + h_q * q_h_stride
                )
                var o_base = (
                    b * out_b_stride
                    + q_idx * out_s_stride
                    + h_q * out_h_stride
                )
                var do_base = (
                    b * dout_b_stride
                    + q_idx * dout_s_stride
                    + h_q * dout_h_stride
                )
                var lse_idx = (b * nheads_q + h_q) * seqlen_q + q_idx
                var lse = lse_ptr[lse_idx]

                # Skip fully-masked rows (lse stored as -inf).
                if lse == neg_inf:
                    continue

                # s = q · k * scale, P = exp(s - lse)
                var s: Float32 = 0

                @parameter
                for d in range(headdim):
                    s += (
                        q_ptr[q_base + d * q_d_stride].cast[accum_t]()
                        * k_vec[d]
                    )
                s *= softmax_scale
                var p = exp(s - lse)

                # dP = dO · V_j ; D_i = dO · O_i
                var dp: Float32 = 0
                var D_i: Float32 = 0

                @parameter
                for d in range(headdim):
                    var do_d = dout_ptr[
                        do_base + d * dout_d_stride
                    ].cast[accum_t]()
                    dp += do_d * v_vec[d]
                    D_i += (
                        do_d
                        * out_ptr[o_base + d * out_d_stride].cast[accum_t]()
                    )

                var ds = p * (dp - D_i)

                @parameter
                for d in range(headdim):
                    var do_d = dout_ptr[
                        do_base + d * dout_d_stride
                    ].cast[accum_t]()
                    var q_d = q_ptr[q_base + d * q_d_stride].cast[accum_t]()
                    dv_acc[d] += p * do_d
                    dk_acc[d] += ds * q_d * softmax_scale

        @parameter
        for d in range(headdim):
            dk_ptr[dk_base + d * dk_d_stride] = dk_acc[d].cast[dtype]()
            dv_ptr[dv_base + d * dv_d_stride] = dv_acc[d].cast[dtype]()

    sync_parallelize[pass_b](batch * nheads_kv * seqlen_k)
