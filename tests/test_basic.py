"""Smoke tests for the scaffolding.

These don't exercise the (yet-unimplemented) Mojo kernels. They check
that the package imports cleanly, the public API surface is what we
expect, and the pure-PyTorch CPU fallback path works against itself.
The real kernel correctness tests will come once `fwd/` and `bwd/`
are filled in.
"""

import pytest
import torch

import flash_attn_mojo
from flash_attn_mojo.reference import flash_attn_ref


def test_package_importable():
    assert callable(flash_attn_mojo.flash_attn_func)
    assert callable(flash_attn_mojo.flash_attn_ref)
    assert isinstance(flash_attn_mojo.__version__, str)


def test_ref_basic_shape():
    """Reference path: any (B, L, H, D) input survives the round trip
    with the expected output shape."""
    B, L, H, D = 2, 8, 4, 16
    q = torch.randn(B, L, H, D, dtype=torch.float32)
    k = torch.randn(B, L, H, D, dtype=torch.float32)
    v = torch.randn(B, L, H, D, dtype=torch.float32)
    out = flash_attn_ref(q, k, v, causal=True)
    assert out.shape == (B, L, H, D)
    assert out.dtype == torch.float32


def test_ref_matches_pytorch_sdpa():
    """The non-flash-specific code path should agree with PyTorch's
    own SDPA on shapes that don't use alibi/softcap/window."""
    B, L, H, D = 1, 4, 2, 8
    q = torch.randn(B, L, H, D, dtype=torch.float32)
    k = torch.randn(B, L, H, D, dtype=torch.float32)
    v = torch.randn(B, L, H, D, dtype=torch.float32)

    out_ours = flash_attn_ref(q, k, v, causal=False)

    # Same op via torch SDPA directly
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2)
    v_h = v.transpose(1, 2)
    out_torch = (
        torch.nn.functional.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=False)
        .transpose(1, 2)
        .contiguous()
    )

    assert torch.allclose(out_ours, out_torch, rtol=1e-5, atol=1e-6)


def test_cpu_routes_through_ref():
    """`flash_attn_func` on a CPU tensor must fall back to the
    reference path (no GPU kernel needed)."""
    B, L, H, D = 1, 4, 2, 8
    q = torch.randn(B, L, H, D, dtype=torch.float32)
    k = torch.randn(B, L, H, D, dtype=torch.float32)
    v = torch.randn(B, L, H, D, dtype=torch.float32)
    # Should not raise — CPU dispatch goes to reference.
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    assert out.shape == (B, L, H, D)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_cuda_kernel_matches_reference():
    """Smallest-envelope CUDA call (fp16, head_dim=64, non-causal,
    no dropout) must produce results within fp16 tolerance of the
    pure-PyTorch reference."""
    B, L, H, D = 1, 32, 2, 64
    q = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")

    out = flash_attn_mojo.flash_attn_func(q, k, v)
    ref = flash_attn_mojo.flash_attn_ref(q, k, v)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max |diff| {diff:.3e} exceeds tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_cuda_outside_envelope_rejects_clearly():
    """Anything outside the current minimal envelope (causal, dropout,
    head_dim != 64, etc.) must error cleanly with NotImplementedError."""
    B, L, H, D = 1, 4, 2, 64
    q = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.float16, device="cuda")
    with pytest.raises(NotImplementedError, match="causal"):
        flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
