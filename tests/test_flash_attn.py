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


def _ref_attention(q, k, v, softmax_scale=None, causal=False):
    """Reference scaled-dot-product attention computed in fp32.

    q/k/v are (B, S, H, D) — the same layout flash_attn_func uses.
    Internally we transpose to (B, H, S, D) for the matmuls, then back.

    Causal mask uses bottom-right alignment, matching upstream
    `flash_attn_func`: q_i attends to k_j iff j <= (Sk - Sq) + i.
    """
    import math

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    # (B, H, S, D)
    q32 = q.transpose(1, 2).to(torch.float32)
    k32 = k.transpose(1, 2).to(torch.float32)
    v32 = v.transpose(1, 2).to(torch.float32)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * softmax_scale
    if causal:
        sq, sk = q.shape[1], k.shape[1]
        # mask[i, j] = True where attention is allowed.
        i = torch.arange(sq).unsqueeze(1)
        j = torch.arange(sk).unsqueeze(0)
        allowed = j <= (sk - sq) + i
        scores = scores.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # All-masked rows become NaN under softmax — mirror our kernel's
    # zero output in that case.
    probs = torch.nan_to_num(probs, nan=0.0)
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


# Phase 1.2: causal masking with bottom-right alignment.
@pytest.mark.parametrize("seqlen", [1, 4, 16, 128])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_flash_attn_func_causal(batch, nheads, seqlen):
    """Causal flash_attn_func matches the reference for seqlen_q=seqlen_k."""
    headdim = 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    ref = _ref_attention(q, k, v, causal=True)

    assert out.shape == ref.shape == q.shape
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max_diff={diff}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(1, 4), (3, 5), (4, 16), (8, 8)])
def test_flash_attn_func_causal_unequal_seqlens(seqlen_q, seqlen_k):
    """Causal with seqlen_q < seqlen_k: bottom-right aligned mask."""
    headdim, nheads, batch = 64, 2, 1
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen_k, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    ref = _ref_attention(q, k, v, causal=True)

    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max_diff={diff} (sq={seqlen_q}, sk={seqlen_k})"


def test_flash_attn_func_causal_q_longer_than_k():
    """Causal with seqlen_q > seqlen_k: rows above the bottom-right
    diagonal attend to nothing and produce zero output."""
    seqlen_q, seqlen_k, headdim, nheads, batch = 5, 3, 64, 1, 1
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen_k, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    # First two rows have k_max = (3-5) + i < 0 → output zero.
    assert torch.equal(out[:, :2], torch.zeros_like(out[:, :2]))
    # Last three rows (i ∈ {2, 3, 4}) have k_max ∈ {0, 1, 2} → match ref.
    ref = _ref_attention(q, k, v, causal=True)
    diff = (out[:, 2:].float() - ref[:, 2:].float()).abs().max().item()
    assert diff < 5e-3


# Phase 1.7: backward — gradients vs the fp32 reference.


def _grads_from_ref(q, k, v, dout, softmax_scale=None, causal=False):
    """Run the reference attention with autograd and return (dq, dk, dv)."""
    qg = q.detach().clone().to(torch.float32).requires_grad_(True)
    kg = k.detach().clone().to(torch.float32).requires_grad_(True)
    vg = v.detach().clone().to(torch.float32).requires_grad_(True)
    out = _ref_attention(
        qg.to(q.dtype),
        kg.to(k.dtype),
        vg.to(v.dtype),
        softmax_scale=softmax_scale,
        causal=causal,
    )
    out.float().backward(dout.float())
    return qg.grad, kg.grad, vg.grad


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("headdim", [64, 96, 128])
def test_flash_attn_func_backward(causal, headdim):
    """Backward through flash_attn_func matches the fp32 reference."""
    batch, seqlen, nheads = 2, 16, 2
    q = torch.randn(
        batch, seqlen, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    out.backward(dout)
    dq_ref, dk_ref, dv_ref = _grads_from_ref(q, k, v, dout, causal=causal)

    # fp16 backward accumulates more roundoff than fwd; bump tolerance to
    # ~1e-2 (matches what upstream's own bwd tests use).
    for name, got, ref in [
        ("dq", q.grad, dq_ref),
        ("dk", k.grad, dk_ref),
        ("dv", v.grad, dv_ref),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff}"


@pytest.mark.parametrize("nheads_q,nheads_kv", [(8, 1), (8, 2), (4, 2)])
def test_flash_attn_func_backward_gqa(nheads_q, nheads_kv):
    """GQA backward — gradients on the kv side sum across the q heads
    sharing each kv head."""
    batch, seqlen, headdim = 2, 8, 64
    q = torch.randn(
        batch, seqlen, nheads_q, headdim, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, nheads_kv, headdim, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, nheads_kv, headdim, dtype=torch.float16, requires_grad=True
    )
    dout = torch.randn(batch, seqlen, nheads_q, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    out.backward(dout)

    # Reference: tile k/v to nheads_q for fwd, then aggregate the per-q-head
    # k/v gradients back down to nheads_kv groups.
    repeat = nheads_q // nheads_kv
    k_full = k.detach().repeat_interleave(repeat, dim=2).requires_grad_(True)
    v_full = v.detach().repeat_interleave(repeat, dim=2).requires_grad_(True)
    qg = q.detach().clone().requires_grad_(True)
    out_ref = _ref_attention(qg, k_full, v_full, causal=True)
    out_ref.float().backward(dout.float())
    dk_ref = k_full.grad.view(batch, seqlen, nheads_kv, repeat, headdim).sum(dim=3)
    dv_ref = v_full.grad.view(batch, seqlen, nheads_kv, repeat, headdim).sum(dim=3)

    for name, got, ref in [
        ("dq", q.grad, qg.grad),
        ("dk", k.grad, dk_ref),
        ("dv", v.grad, dv_ref),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff}"


def test_flash_attn_func_backward_causal_q_longer_than_k():
    """Backward with seqlen_q > seqlen_k — fully-masked rows produce
    zero gradients on q (and contribute nothing on k/v)."""
    seqlen_q, seqlen_k, headdim, nheads, batch = 5, 3, 64, 1, 1
    q = torch.randn(
        batch, seqlen_q, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen_k, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen_k, nheads, headdim, dtype=torch.float16, requires_grad=True
    )
    dout = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    out.backward(dout)

    # Top two q rows are fully masked → dq is exactly zero there.
    assert torch.equal(q.grad[:, :2], torch.zeros_like(q.grad[:, :2]))


# Phase 1.11: deterministic backward.
def test_flash_attn_func_deterministic_kwarg_accepted():
    """`deterministic=True` is accepted (CPU bwd is already deterministic
    — pass A and pass B write disjoint outputs)."""
    q = torch.randn(1, 8, 1, 64, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 8, 1, 64, dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 8, 1, 64, dtype=torch.float16, requires_grad=True)
    dout = torch.randn(1, 8, 1, 64, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, deterministic=True)
    out.backward(dout)

    # Re-run with the same inputs and check bit-exact equality.
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out2 = flash_attn_mojo.flash_attn_func(q2, k2, v2, deterministic=True)
    out2.backward(dout)
    assert torch.equal(q.grad, q2.grad)
    assert torch.equal(k.grad, k2.grad)
    assert torch.equal(v.grad, v2.grad)


# ---- "still raises" tests for features not yet implemented ----


# Phase 1.3: headdim ∈ {64, 96, 128} — the three most common GQA sizes.
@pytest.mark.parametrize("headdim", [64, 96, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_headdim(headdim, causal):
    """Correctness for all three supported headdim values, both causal modes."""
    batch, seqlen, nheads = 2, 16, 2
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal=causal)

    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max_diff={diff} (headdim={headdim}, causal={causal})"


def test_unsupported_headdim_raises():
    """headdim=32 (and 160/192/224/256) aren't dispatched yet."""
    q = torch.randn(1, 4, 1, 32, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 32, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 32, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="headdim"):
        flash_attn_mojo.flash_attn_func(q, k, v)


# Phase 1.4: MQA / GQA — nheads_q is a multiple of nheads_kv.
@pytest.mark.parametrize(
    "nheads_q,nheads_kv",
    [(8, 1), (8, 2), (8, 4), (4, 2), (2, 1)],
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_gqa(nheads_q, nheads_kv, causal):
    """flash_attn_func with shared KV heads matches the broadcast reference."""
    batch, seqlen, headdim = 2, 16, 64
    q = torch.randn(batch, seqlen, nheads_q, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads_kv, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads_kv, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    # Reference: tile k and v from nheads_kv to nheads_q.
    repeat = nheads_q // nheads_kv
    k_full = k.repeat_interleave(repeat, dim=2)
    v_full = v.repeat_interleave(repeat, dim=2)
    ref = _ref_attention(q, k_full, v_full, causal=causal)

    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max_diff={diff} (q={nheads_q}, kv={nheads_kv})"


def test_gqa_non_divisible_raises():
    """nheads_q must be a multiple of nheads_kv."""
    q = torch.randn(1, 4, 6, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 4, 64, dtype=torch.float16)  # 6 % 4 != 0
    v = torch.randn(1, 4, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple"):
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


# Phase 1.17: bf16 + fp32 dispatch.
@pytest.mark.parametrize(
    "input_dtype,fwd_tol,bwd_tol",
    [
        (torch.float16, 5e-3, 1e-2),
        (torch.bfloat16, 1e-2, 5e-2),  # bf16 has 7 mantissa bits — looser
        (torch.float32, 5e-5, 5e-4),
    ],
    ids=["fp16", "bf16", "fp32"],
)
def test_flash_attn_func_dtypes(input_dtype, fwd_tol, bwd_tol):
    """Forward + backward correctness across fp16 / bf16 / fp32."""
    batch, seqlen, nheads, headdim = 2, 16, 2, 64
    q = torch.randn(
        batch, seqlen, nheads, headdim, dtype=input_dtype, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, nheads, headdim, dtype=input_dtype, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, nheads, headdim, dtype=input_dtype, requires_grad=True
    )
    dout = torch.randn(batch, seqlen, nheads, headdim, dtype=input_dtype)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    ref = _ref_attention(q.detach(), k.detach(), v.detach(), causal=True)
    assert out.dtype == input_dtype
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < fwd_tol, f"fwd max_diff={diff} ({input_dtype})"

    out.backward(dout)
    dq_ref, dk_ref, dv_ref = _grads_from_ref(q, k, v, dout, causal=True)
    for name, got, ref in [
        ("dq", q.grad, dq_ref),
        ("dk", k.grad, dk_ref),
        ("dv", v.grad, dv_ref),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < bwd_tol, f"{name} max_diff={diff} ({input_dtype})"


def test_unsupported_dtype_raises():
    """fp64 isn't dispatched."""
    q = torch.randn(1, 4, 1, 64, dtype=torch.float64)
    k = torch.randn(1, 4, 1, 64, dtype=torch.float64)
    v = torch.randn(1, 4, 1, 64, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="dtype"):
        flash_attn_mojo.flash_attn_func(q, k, v)


def test_mixed_dtype_raises():
    """q/k/v dtypes must match."""
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 1, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
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


# Phase 1.6: flash_attn_qkvpacked_func is a thin wrapper around
# flash_attn_func — same correctness, plus shape validation.
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_qkvpacked_func(causal):
    """qkvpacked path matches the unpacked one."""
    batch, seqlen, nheads, headdim = 2, 16, 2, 64
    qkv = torch.randn(batch, seqlen, 3, nheads, headdim, dtype=torch.float16)
    q, k, v = qkv.unbind(dim=2)

    out_packed = flash_attn_mojo.flash_attn_qkvpacked_func(qkv, causal=causal)
    out_unpacked = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)

    assert torch.equal(out_packed, out_unpacked)


def test_flash_attn_qkvpacked_func_bad_shape_raises():
    """qkv must have a size-3 dim-2."""
    bad = torch.randn(1, 4, 4, 2, 64, dtype=torch.float16)  # dim-2 = 4, not 3
    with pytest.raises(ValueError, match="seqlen, 3, nheads"):
        flash_attn_mojo.flash_attn_qkvpacked_func(bad)


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
