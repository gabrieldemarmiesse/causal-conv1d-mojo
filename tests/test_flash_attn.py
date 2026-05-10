"""Tests for the flash_attn_mojo package.

Compares output (and gradients, where applicable) against the upstream
``flash_attn`` PyPI package.

The test environment runs on whichever GPU pixi happens to be on. Some
features upstream are arch-gated (FA3 needs sm_90, FA4 needs sm_100);
those tests skip when the local GPU doesn't qualify. Likewise, every
test skips when ``flash_attn`` isn't importable.

Phase 1 lives behind ``pytest.xfail`` markers — each phase step
removes the xfail and adds a real correctness assertion. See
``flash_attn_mojo/__init__.py`` for the phase ladder.
"""

from __future__ import annotations

import pytest
import torch

import flash_attn_mojo


# ---- environment gates ------------------------------------------------------

try:
    import flash_attn  # noqa: F401  (just import-check)

    _UPSTREAM_AVAILABLE = True
except ImportError:
    _UPSTREAM_AVAILABLE = False


def _gpu_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def _skip_if_no_upstream() -> None:
    if not _UPSTREAM_AVAILABLE:
        pytest.skip("upstream flash_attn not installed")


def _skip_if_unsupported_arch() -> None:
    """flash-attn FA2 requires Ampere or newer (sm_80+). FA3 features
    require sm_90+. We test against the public API which auto-selects;
    skipping the whole test below sm_80 keeps the suite portable."""
    cap = _gpu_capability()
    if cap < (8, 0):
        pytest.skip(f"flash-attn requires sm_80+, GPU is sm_{cap[0]}{cap[1]}")


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


def _ref_attention(q, k, v, softmax_scale=None):
    """Reference scaled-dot-product attention computed in fp32.

    q/k/v are (B, S, H, D) — the same layout flash_attn_func uses.
    Internally we transpose to (B, H, S, D) for the matmuls, then back.
    """
    import math

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    # (B, H, S, D)
    q32 = q.transpose(1, 2).to(torch.float32)
    k32 = k.transpose(1, 2).to(torch.float32)
    v32 = v.transpose(1, 2).to(torch.float32)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * softmax_scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v32)
    # Back to (B, S, H, D), cast to input dtype.
    return out.transpose(1, 2).to(q.dtype)


# Phase 1.1: minimum forward — fp16, headdim=64, non-causal, MHA, CPU.
@pytest.mark.parametrize("seqlen", [1, 4, 16, 128])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_flash_attn_func_forward_minimum(batch, nheads, seqlen):
    """flash_attn_func forward matches the pytorch reference for the
    phase-1.1 supported shape: fp16, headdim=64, non-causal, MHA, CPU."""
    headdim = 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v)
    ref = _ref_attention(q, k, v)

    assert out.shape == ref.shape == q.shape
    assert out.dtype == ref.dtype == torch.float16
    diff = (out.float() - ref.float()).abs().max().item()
    # fp16 attention has roundoff in scores + softmax + V matmul; ~5e-3
    # is the tolerance upstream uses in their own tests.
    assert diff < 5e-3, f"max_diff={diff}"


def test_flash_attn_func_default_softmax_scale():
    """Default softmax_scale is 1/sqrt(headdim)."""
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    import math

    out_default = flash_attn_mojo.flash_attn_func(q, k, v)
    out_explicit = flash_attn_mojo.flash_attn_func(
        q, k, v, softmax_scale=1.0 / math.sqrt(64)
    )
    assert torch.equal(out_default, out_explicit)


def test_flash_attn_func_explicit_softmax_scale():
    """A non-default softmax_scale is honoured."""
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    scale = 0.25  # very far from 1/sqrt(64)=0.125 — noticeably different output
    out = flash_attn_mojo.flash_attn_func(q, k, v, softmax_scale=scale)
    ref = _ref_attention(q, k, v, softmax_scale=scale)
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3


# ---- "still raises" tests for features not yet implemented ----


def test_causal_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1.2"):
        flash_attn_mojo.flash_attn_func(q, k, v, causal=True)


def test_headdim_other_than_64_raises():
    q = torch.randn(1, 4, 1, 128, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 128, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 128, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1.3"):
        flash_attn_mojo.flash_attn_func(q, k, v)


def test_mqa_gqa_raises():
    """nheads_q != nheads_kv → MQA/GQA, not yet implemented."""
    q = torch.randn(1, 4, 4, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1.4"):
        flash_attn_mojo.flash_attn_func(q, k, v)


def test_dropout_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1.8"):
        flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.1)


def test_window_size_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1.9"):
        flash_attn_mojo.flash_attn_func(q, k, v, window_size=(2, 2))


def test_alibi_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    slopes = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="phase 1.10"):
        flash_attn_mojo.flash_attn_func(q, k, v, alibi_slopes=slopes)


def test_bf16_fp32_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="phase 1.17"):
        flash_attn_mojo.flash_attn_func(q, k, v)


def test_gpu_raises():
    """Phase 1.1 is CPU-only."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 1, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 4, 1, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(NotImplementedError, match="CPU-only"):
        flash_attn_mojo.flash_attn_func(q, k, v)


def test_flash_attn_qkvpacked_func_raises():
    qkv = torch.randn(1, 4, 3, 2, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1"):
        flash_attn_mojo.flash_attn_qkvpacked_func(qkv)


def test_flash_attn_with_kvcache_raises():
    q = torch.randn(1, 1, 2, 64, dtype=torch.float16)
    k_cache = torch.zeros(1, 16, 2, 64, dtype=torch.float16)
    v_cache = torch.zeros(1, 16, 2, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1"):
        flash_attn_mojo.flash_attn_with_kvcache(q, k_cache, v_cache)


# Sanity: upstream flash_attn is importable in this env (non-fatal).
def test_upstream_importable():
    _skip_if_no_upstream()
    import flash_attn

    assert flash_attn.__version__.startswith("2.")
