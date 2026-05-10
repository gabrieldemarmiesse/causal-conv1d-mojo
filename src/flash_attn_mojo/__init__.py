"""Flash-Attention, Mojo port (WIP — Phase 1).

Companion package to `causal_conv1d_mojo` in this repo, mirroring the
Tri Dao `flash-attn` 2.x public API. Three entry points planned:

- ``flash_attn_func(q, k, v, ...)``                       — standard MHA / MQA / GQA attention.
- ``flash_attn_qkvpacked_func(qkv, ...)``                 — Q/K/V packed into one tensor.
- ``flash_attn_with_kvcache(q, k_cache, v_cache, ...)``   — autoregressive decode with KV-cache.

All three currently raise ``NotImplementedError``. Features land
incrementally; each commit moves one option from raises to working.
See README's "Flash Attention" section for the live phase status.

Tests compare correctness against the upstream ``flash_attn`` PyPI
package (pinned to a prebuilt wheel via pyproject.toml).
"""

from __future__ import annotations


__version__ = "0.0.0"


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
    contiguous and 16-byte-aligned.

    See ``flash-attention/README.md`` ("How to use FlashAttention")
    for the full argument semantics.
    """
    raise NotImplementedError(
        "flash_attn_func is not implemented yet — phase 1.1 lands the"
        " minimal forward (fp16, headdim=64, non-causal, MHA only)."
    )


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
    ``(batch, seqlen, 3, nheads, headdim)`` tensor. Backward avoids
    explicit gradient concatenation.

    Will be implemented as a thin wrapper around ``flash_attn_func``
    once that lands.
    """
    raise NotImplementedError(
        "flash_attn_qkvpacked_func is not implemented yet — depends on"
        " flash_attn_func (phase 1.6)."
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
