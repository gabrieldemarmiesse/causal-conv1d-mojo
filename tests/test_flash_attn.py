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


# ---- placeholder tests ------------------------------------------------------
#
# These confirm the public API surface exists and currently raises. As
# each phase step lands, the matching xfail/raises block is replaced
# with a real correctness assertion against upstream flash_attn.


def test_flash_attn_func_raises():
    """Phase 0: flash_attn_func is not implemented yet."""
    q = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="phase 1"):
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
