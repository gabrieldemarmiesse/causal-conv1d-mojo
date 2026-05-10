"""Flash-Attention, Mojo port (WIP — Phase 1).

Companion package to `causal_conv1d_mojo` in this repo, mirroring the
Tri Dao `flash-attn` 2.x public API. Three entry points planned:

- ``flash_attn_func(q, k, v, ...)``                       — standard MHA / MQA / GQA attention.
- ``flash_attn_qkvpacked_func(qkv, ...)``                 — Q/K/V packed into one tensor.
- ``flash_attn_with_kvcache(q, k_cache, v_cache, ...)``   — autoregressive decode with KV-cache.

Phase 1.x (current): minimal CPU forward for ``flash_attn_func`` —
fp16, headdim ∈ {64, 96, 128}, MHA / MQA / GQA, optional causal mask.
Everything else still raises.

Tests compare correctness against the upstream ``flash_attn`` PyPI
package (pinned to a prebuilt wheel via pyproject.toml).
"""

from __future__ import annotations

import math

import torch

# `mojo.importer` registers a Python import hook that compiles
# `flash_attn_native.mojo` to a shared lib on first import.
import mojo.importer  # noqa: F401

from flash_attn_mojo._native import flash_attn_native as _native_mod


__version__ = "0.0.0"


# Must match the dispatch in the Mojo entry points.
_DTYPE_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}


def _strides_4d(t):
    """Flatten a 4-D tensor's strides into a 4-tuple in (b, s, h, d) order."""
    return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))


def _apply_rotary(x, cos, sin, positions, interleaved):
    """Apply rotary embedding to a (B, S, H, D) tensor.

    `cos` / `sin` are (max_pos, rotary_dim // 2) (matches upstream's
    flash_attn shape — half-size, since RoPE rotates *pairs* of dims).
    `positions` is (B, S) int — actual token position per query row.
    Only the first ``rotary_dim`` dims of x are touched; the remaining
    headdim - rotary_dim dims pass through unchanged.

    Layouts:
        interleaved=True: pairs are (x[2i], x[2i+1]) — Llama / GPT-J style
        interleaved=False: split — pairs are (x[i], x[D/2 + i])
    """
    if cos.shape != sin.shape:
        raise ValueError(
            f"rotary_cos and rotary_sin must have the same shape; got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.dim() != 2:
        raise ValueError(
            f"rotary_cos must be 2-D (max_pos, rotary_dim/2); got {tuple(cos.shape)}"
        )
    rotary_half = cos.shape[1]
    rotary_dim = rotary_half * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim ({rotary_dim}) > headdim ({x.shape[-1]})")
    cos_at = cos[positions].unsqueeze(2)  # (B, S, 1, rotary_half)
    sin_at = sin[positions].unsqueeze(2)
    out = x.clone()
    if interleaved:
        # x[..., :rotary_dim] viewed as (..., rotary_half, 2) pairs
        x_pairs = out[..., :rotary_dim].view(*x.shape[:-1], rotary_half, 2)
        x0 = x_pairs[..., 0].clone()
        x1 = x_pairs[..., 1].clone()
        x_pairs[..., 0] = x0 * cos_at.to(x0.dtype) - x1 * sin_at.to(x0.dtype)
        x_pairs[..., 1] = x0 * sin_at.to(x0.dtype) + x1 * cos_at.to(x0.dtype)
    else:
        x0 = out[..., :rotary_half].clone()
        x1 = out[..., rotary_half:rotary_dim].clone()
        out[..., :rotary_half] = x0 * cos_at.to(x0.dtype) - x1 * sin_at.to(x0.dtype)
        out[..., rotary_half:rotary_dim] = x0 * sin_at.to(x0.dtype) + x1 * cos_at.to(
            x0.dtype
        )
    return out


def _normalise_alibi(alibi_slopes, nheads_q, batch):
    """Return a contiguous fp32 alibi tensor (or None) — the kernel reads
    it as a flat fp32 buffer indexed by `b * batch_stride + h_q`.

    Caller must hold the returned tensor alive for the duration of the
    native call so its storage isn't freed.
    """
    if alibi_slopes is None:
        return None, 0
    if alibi_slopes.dtype != torch.float32:
        raise ValueError(f"alibi_slopes must be fp32 (got {alibi_slopes.dtype})")
    a = alibi_slopes.contiguous()
    if a.dim() == 1:
        if a.shape[0] != nheads_q:
            raise ValueError(
                f"alibi_slopes shape {tuple(a.shape)} doesn't match nheads_q={nheads_q}"
            )
        return a, 0
    if a.dim() == 2:
        if a.shape != (batch, nheads_q):
            raise ValueError(
                f"alibi_slopes shape {tuple(a.shape)} doesn't match "
                f"(batch={batch}, nheads_q={nheads_q})"
            )
        return a, a.shape[1]
    raise ValueError(
        f"alibi_slopes must be 1-D (nheads,) or 2-D (batch, nheads); "
        f"got shape {tuple(a.shape)}"
    )


def _native_fwd_cpu(
    q,
    k,
    v,
    out,
    lse,
    softmax_scale,
    causal,
    window,
    alibi,
    dropout_mask,
    cache_seqlens=None,
    softcap=0.0,
    cache_batch_idx=None,
):
    alibi_t, alibi_stride = _normalise_alibi(alibi, q.shape[2], q.shape[0])
    alibi_addr = alibi_t.data_ptr() if alibi_t is not None else 0
    dropout_addr = dropout_mask.data_ptr() if dropout_mask is not None else 0
    cache_seqlens_addr = cache_seqlens.data_ptr() if cache_seqlens is not None else 0
    cache_batch_idx_addr = (
        cache_batch_idx.data_ptr() if cache_batch_idx is not None else 0
    )
    _native_mod.flash_attn_fwd_cpu(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        lse.data_ptr(),
        q.shape[0],  # batch
        q.shape[1],  # seqlen_q
        k.shape[1],  # seqlen_k
        q.shape[2],  # nheads_q
        k.shape[2],  # nheads_kv
        *_strides_4d(q),
        *_strides_4d(k),
        *_strides_4d(v),
        *_strides_4d(out),
        float(softmax_scale),
        _DTYPE_CODE[q.dtype],
        q.shape[3],  # headdim
        1 if causal else 0,
        int(window[0]),
        int(window[1]),
        alibi_addr,
        alibi_stride,
        dropout_addr,
        cache_seqlens_addr,
        float(softcap),
        cache_batch_idx_addr,
    )


def _native_bwd_cpu(
    q,
    k,
    v,
    out,
    dout,
    lse,
    dq,
    dk,
    dv,
    softmax_scale,
    causal,
    window,
    alibi,
    dropout_mask,
    softcap=0.0,
):
    alibi_t, alibi_stride = _normalise_alibi(alibi, q.shape[2], q.shape[0])
    alibi_addr = alibi_t.data_ptr() if alibi_t is not None else 0
    dropout_addr = dropout_mask.data_ptr() if dropout_mask is not None else 0
    _native_mod.flash_attn_bwd_cpu(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        dout.data_ptr(),
        lse.data_ptr(),
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        q.shape[0],  # batch
        q.shape[1],  # seqlen_q
        k.shape[1],  # seqlen_k
        q.shape[2],  # nheads_q
        k.shape[2],  # nheads_kv
        *_strides_4d(q),
        *_strides_4d(k),
        *_strides_4d(v),
        *_strides_4d(out),
        *_strides_4d(dout),
        *_strides_4d(dq),
        *_strides_4d(dk),
        *_strides_4d(dv),
        float(softmax_scale),
        _DTYPE_CODE[q.dtype],
        q.shape[3],  # headdim
        1 if causal else 0,
        int(window[0]),
        int(window[1]),
        alibi_addr,
        alibi_stride,
        dropout_addr,
        float(softcap),
    )


class _FlashAttnFunc(torch.autograd.Function):
    """torch.autograd.Function wrapping the native fwd/bwd calls."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal, window, alibi, dropout_p, softcap):
        out = torch.empty_like(q)
        # lse is fp32, shape (batch, nheads_q, seqlen_q), contiguous.
        lse = torch.empty(
            q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device
        )
        # Materialise dropout mask up-front. Pre-scaled by 1/(1-p) so the
        # kernel only needs to multiply.
        dropout_mask = None
        if dropout_p > 0.0:
            keep_prob = 1.0 - dropout_p
            dropout_mask = (
                torch.empty(
                    q.shape[0],
                    q.shape[2],  # nheads_q
                    q.shape[1],  # seqlen_q
                    k.shape[1],  # seqlen_k
                    dtype=torch.float32,
                    device=q.device,
                )
                .bernoulli_(keep_prob)
                .div_(keep_prob)
            )
        _native_fwd_cpu(
            q,
            k,
            v,
            out,
            lse,
            softmax_scale,
            causal,
            window,
            alibi,
            dropout_mask,
            softcap=softcap,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.dropout_mask = dropout_mask  # tensor or None — keeps it alive
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window = window
        ctx.alibi = alibi
        ctx.softcap = softcap
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        # Gradient layout follows q/k/v exactly — same shape and dtype.
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dout = dout.contiguous() if not dout.is_contiguous() else dout
        _native_bwd_cpu(
            q,
            k,
            v,
            out,
            dout,
            lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window,
            ctx.alibi,
            ctx.dropout_mask,
            softcap=ctx.softcap,
        )
        # 9 forward inputs (q, k, v, softmax_scale, causal, window, alibi,
        # dropout_p, softcap) — gradients only flow through the first three.
        return dq, dk, dv, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
):
    """Multi-head / multi-query / grouped-query attention.

    Mirrors ``flash_attn.flash_attn_func`` from upstream. Tensor layout
    is ``(batch, seqlen, nheads, headdim)``; the last dim must be
    contiguous.

    Currently: fp16, headdim=64, MHA (Q and K have the same nheads),
    optional ``causal``, no dropout / window / alibi. Everything else
    raises ``NotImplementedError`` with a phase pointer.
    """
    # ---- feature gates ----
    if not isinstance(dropout_p, (int, float)) or dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p!r}")
    if (
        not isinstance(window_size, tuple)
        or len(window_size) != 2
        or not all(isinstance(w, int) for w in window_size)
    ):
        raise ValueError(f"window_size must be a 2-tuple of ints; got {window_size!r}")
    # alibi_slopes is validated lazily inside _normalise_alibi (so we can
    # reuse the validation in the bwd path).
    # `deterministic` is a no-op: the CPU backward already writes dQ/dK/dV
    # from disjoint workers (pass A over q rows, pass B over k rows), so
    # there's no nondeterminism source to suppress. Accept the kwarg for
    # API compat with upstream and ignore it.
    del deterministic

    # ---- shape / dtype validation ----
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q, k, v must be 4-D (batch, seqlen, nheads, headdim); got "
            f"shapes {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    batch, seqlen_q, nheads_q, headdim = q.shape
    if k.shape != (batch, k.shape[1], k.shape[2], headdim):
        raise ValueError(
            f"k shape {tuple(k.shape)} doesn't match q's batch ({batch}) "
            f"or headdim ({headdim})"
        )
    if v.shape != k.shape:
        raise ValueError(
            f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}"
        )
    nheads_kv = k.shape[2]
    if nheads_q % nheads_kv != 0:
        raise ValueError(
            f"nheads_q ({nheads_q}) must be a multiple of nheads_kv "
            f"({nheads_kv}) for MQA/GQA"
        )
    if q.dtype not in _DTYPE_CODE:
        raise NotImplementedError(
            f"unsupported dtype {q.dtype}; supported: fp16, bf16, fp32"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q, k, v must share dtype (got {q.dtype}, {k.dtype}, {v.dtype})"
        )
    if headdim not in (64, 96, 128):
        raise NotImplementedError(
            f"currently only supports headdim ∈ (64, 96, 128); got {headdim}. "
            f"Other sizes (32, 160, 192, 224, 256) are upstream-supported "
            f"and can be added by extending the dispatch tree in "
            f"_native/flash_attn_native.mojo."
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q, k, v must be on the same device "
            f"(got {q.device}, {k.device}, {v.device})"
        )

    # ---- softmax scale default ----
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # CPU-only for now; GPU forward lands later in 1.x.
    if q.is_cuda:
        raise NotImplementedError(
            "currently CPU-only; GPU forward lands in a later 1.x step"
        )

    if not isinstance(softcap, (int, float)) or softcap < 0:
        raise ValueError(f"softcap must be ≥ 0; got {softcap!r}")
    return _FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        dropout_p,
        float(softcap),
    )


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
):
    """Same as ``flash_attn_func`` but takes Q/K/V stacked into one
    ``(batch, seqlen, 3, nheads, headdim)`` tensor.

    Forward is a thin wrapper around ``flash_attn_func``. Once we have
    a backward, the qkvpacked variant will get its own custom backward
    so gradients land back in qkv without an explicit concat.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(
            f"qkv must be (batch, seqlen, 3, nheads, headdim); "
            f"got shape {tuple(qkv.shape)}"
        )
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """Autoregressive-decode attention with an in-place updated KV cache.

    Phase 1.12 implements the basic case: q + k_cache + v_cache, with
    optional new-token (k, v) append, per-batch valid lengths via
    cache_seqlens, plus causal / window / alibi. Rotary, paged caches,
    and cache_batch_idx remain ``NotImplementedError`` (phases 1.13+).

    Tensor layout:
        q       : (batch, seqlen_q,    nheads_q,  headdim)
        k_cache : (batch, seqlen_kmax, nheads_kv, headdim)
        v_cache : (batch, seqlen_kmax, nheads_kv, headdim)
        k, v    : (batch, seqlen_q,    nheads_kv, headdim) — optional
        cache_seqlens : (batch,) int32, or scalar int (broadcast)

    Side effect: when k and v are provided, they are written into
    k_cache / v_cache at slots [cache_seqlens[b], cache_seqlens[b] +
    seqlen_q). cache_seqlens itself is NOT mutated — that's the
    caller's responsibility (matches upstream).

    Returns: out of shape ``(batch, seqlen_q, nheads_q, headdim)``.
    """
    if (rotary_cos is None) != (rotary_sin is None):
        raise ValueError("rotary_cos and rotary_sin must both be provided or both None")
    # cache_batch_idx is validated below after we know nheads.
    if block_table is not None:
        raise NotImplementedError(
            "paged kv-cache (block_table) is not yet implemented — phase 1.16"
        )
    if alibi_slopes is not None and (causal and window_size != (-1, -1)):
        # Combinations work — leave as-is. Just keeping the gate logic
        # honest. (Removed: this is supported.)
        pass
    if (
        not isinstance(window_size, tuple)
        or len(window_size) != 2
        or not all(isinstance(w, int) for w in window_size)
    ):
        raise ValueError(f"window_size must be a 2-tuple of ints; got {window_size!r}")

    # Shape validation
    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError(
            "q, k_cache, v_cache must be 4-D (batch, seqlen, nheads, headdim); "
            f"got {tuple(q.shape)}, {tuple(k_cache.shape)}, {tuple(v_cache.shape)}"
        )
    batch, seqlen_q, nheads_q, headdim = q.shape
    cache_batch = k_cache.shape[0]  # may differ from q's batch under cache_batch_idx
    seqlen_kmax = k_cache.shape[1]
    nheads_kv = k_cache.shape[2]
    if k_cache.shape != (cache_batch, seqlen_kmax, nheads_kv, headdim):
        raise ValueError(f"k_cache shape {tuple(k_cache.shape)} doesn't match expected")
    if cache_batch_idx is None and cache_batch != batch:
        raise ValueError(
            f"k_cache batch ({cache_batch}) must match q batch ({batch}) when "
            f"cache_batch_idx is not provided"
        )
    if v_cache.shape != k_cache.shape:
        raise ValueError(
            f"v_cache shape {tuple(v_cache.shape)} must match k_cache shape"
        )
    if nheads_q % nheads_kv != 0:
        raise ValueError(
            f"nheads_q ({nheads_q}) must be a multiple of nheads_kv "
            f"({nheads_kv}) for MQA/GQA"
        )
    if q.dtype not in _DTYPE_CODE or q.dtype != k_cache.dtype != v_cache.dtype:
        raise ValueError(
            f"q, k_cache, v_cache must share supported dtype; got "
            f"{q.dtype}, {k_cache.dtype}, {v_cache.dtype}"
        )
    if headdim not in (64, 96, 128):
        raise NotImplementedError(
            f"currently only supports headdim ∈ (64, 96, 128); got {headdim}"
        )
    if q.is_cuda:
        raise NotImplementedError("flash_attn_with_kvcache is CPU-only for now")

    # Resolve cache_seqlens to a contiguous int32 (B,) tensor.
    if cache_seqlens is None:
        cs = torch.zeros(batch, dtype=torch.int32, device=q.device)
    elif isinstance(cache_seqlens, int):
        cs = torch.full((batch,), cache_seqlens, dtype=torch.int32, device=q.device)
    else:
        if cache_seqlens.shape != (batch,):
            raise ValueError(
                f"cache_seqlens shape {tuple(cache_seqlens.shape)} must be ({batch},)"
            )
        cs = cache_seqlens.to(torch.int32).contiguous()

    # Resolve cache_batch_idx (optional). When set, q[b] reads
    # k_cache[cache_batch_idx[b]] / v_cache[cache_batch_idx[b]].
    cbi = None
    if cache_batch_idx is not None:
        if cache_batch_idx.shape != (batch,):
            raise ValueError(
                f"cache_batch_idx shape {tuple(cache_batch_idx.shape)} must "
                f"be ({batch},)"
            )
        cbi = cache_batch_idx.to(torch.int32).contiguous()

    # Optionally append new tokens to the cache.
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together (or neither)")

    # When rotary is given, apply it to q and (the new) k at their
    # absolute token positions. Positions for batch b, query offset i:
    #   pos[b, i] = cache_seqlens[b] + i
    q_for_attn = q
    if rotary_cos is not None:
        positions = cs.unsqueeze(1) + torch.arange(
            seqlen_q, dtype=torch.int32, device=q.device
        ).unsqueeze(0)
        q_for_attn = _apply_rotary(
            q, rotary_cos, rotary_sin, positions.long(), rotary_interleaved
        )
        if k is not None:
            k = _apply_rotary(
                k, rotary_cos, rotary_sin, positions.long(), rotary_interleaved
            )

    if k is not None:
        if k.shape != (batch, seqlen_q, nheads_kv, headdim) or v.shape != k.shape:
            raise ValueError(
                f"k/v new-token shape must be ({batch}, {seqlen_q}, "
                f"{nheads_kv}, {headdim}); got {tuple(k.shape)}/{tuple(v.shape)}"
            )
        for b in range(batch):
            n = int(cs[b].item())
            slot = int(cbi[b].item()) if cbi is not None else b
            if n + seqlen_q > seqlen_kmax:
                raise ValueError(
                    f"batch {b}: cache_seqlens[{b}]={n} + seqlen_q={seqlen_q} "
                    f"exceeds seqlen_kmax={seqlen_kmax}"
                )
            k_cache[slot, n : n + seqlen_q] = k[b]
            v_cache[slot, n : n + seqlen_q] = v[b]
        cs = cs + seqlen_q  # effective valid k length now includes new tokens

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # Allocate out + lse and call the kernel directly (no autograd —
    # decode-time inference doesn't need gradients).
    out = torch.empty_like(q)
    lse = torch.empty(batch, nheads_q, seqlen_q, dtype=torch.float32, device=q.device)
    _native_fwd_cpu(
        q_for_attn,
        k_cache,
        v_cache,
        out,
        lse,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        None,  # no dropout in kvcache path
        cs,
        softcap=float(softcap),
        cache_batch_idx=cbi,
    )
    return out
