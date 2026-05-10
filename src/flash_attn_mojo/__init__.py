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


def _native_fwd_cpu(q, k, v, out, softmax_scale, causal):
    _native_mod.flash_attn_fwd_cpu(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        q.shape[0],  # batch
        q.shape[1],  # seqlen_q
        k.shape[1],  # seqlen_k
        q.shape[2],  # nheads_q
        k.shape[2],  # nheads_kv
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        float(softmax_scale),
        _DTYPE_CODE[q.dtype],
        q.shape[3],  # headdim
        1 if causal else 0,
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
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
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p is not implemented yet — phase 1.8")
    if window_size != (-1, -1):
        raise NotImplementedError(
            "window_size (sliding-window/local) is not implemented yet — phase 1.9"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not implemented yet — phase 1.10")
    if deterministic:
        raise NotImplementedError(
            "deterministic backward is not implemented yet — phase 1.11"
        )

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
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise NotImplementedError(
            "phase 1.1 only supports fp16; bf16 + fp32 land in phase 1.17"
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

    # Phase 1.1 is CPU-only. GPU lands later in 1.x.
    if q.is_cuda:
        raise NotImplementedError(
            "phase 1.1 is CPU-only; GPU forward lands in a later 1.x step"
        )

    out = torch.empty_like(q)
    _native_fwd_cpu(q, k, v, out, softmax_scale, causal)
    return out


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
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
    rotary_interleaved=True,
    alibi_slopes=None,
):
    """Autoregressive-decode attention with an in-place updated KV cache.

    Lands in phase 1.12 (basic case) and is fleshed out through 1.16
    (rotary, paged kv cache, indirection).
    """
    raise NotImplementedError(
        "flash_attn_with_kvcache is not implemented yet — phase 1.12+."
    )
