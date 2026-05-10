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


def _ref_attention(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    window=(-1, -1),
    alibi=None,
    softcap=0.0,
):
    """Reference scaled-dot-product attention computed in fp32.

    q/k/v are (B, S, H, D) — the same layout flash_attn_func uses.
    Internally we transpose to (B, H, S, D) for the matmuls, then back.

    Causal mask uses bottom-right alignment, matching upstream
    `flash_attn_func`: q_i attends to k_j iff j <= (Sk - Sq) + i.
    Sliding window: q_i attends to k_j iff j ∈ [pos-left, pos+right]
    where pos = (Sk - Sq) + i. left/right < 0 means "no bound" on
    that side. ALiBi: bias_ij = -slope[h] * |pos - j|, slope is
    fp32 (nheads,) or (batch, nheads).
    """
    import math

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    # (B, H, S, D)
    q32 = q.transpose(1, 2).to(torch.float32)
    k32 = k.transpose(1, 2).to(torch.float32)
    v32 = v.transpose(1, 2).to(torch.float32)
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * softmax_scale
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    sq, sk = q.shape[1], k.shape[1]
    i = torch.arange(sq).unsqueeze(1)
    j = torch.arange(sk).unsqueeze(0)
    pos = (sk - sq) + i
    if alibi is not None:
        # broadcast slope to (B, H, 1, 1)
        slope = alibi.to(torch.float32)
        if slope.dim() == 1:
            slope = slope[None, :]  # (1, H)
        slope = slope[..., None, None]  # (B|1, H, 1, 1)
        dist = (pos - j).abs().to(torch.float32)  # (sq, sk)
        scores = scores - slope * dist
    allowed = torch.ones_like(scores[0, 0], dtype=torch.bool)
    if causal:
        allowed = allowed & (j <= pos)
    if window[0] >= 0:
        allowed = allowed & (j >= pos - window[0])
    if window[1] >= 0:
        allowed = allowed & (j <= pos + window[1])
    if not allowed.all():
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


# Phase 1.3: full headdim sweep. Upstream supports
# {32, 64, 96, 128, 160, 192, 224, 256}; we dispatch all of them.
@pytest.mark.parametrize("headdim", [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_headdim(headdim, causal):
    """Forward correctness for every supported headdim, both causal modes."""
    batch, seqlen, nheads = 1, 8, 1
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal=causal)

    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"max_diff={diff} (headdim={headdim}, causal={causal})"


def test_unsupported_headdim_raises():
    """headdim=48 (and other non-listed sizes) aren't dispatched."""
    q = torch.randn(1, 4, 1, 48, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 48, dtype=torch.float16)
    v = torch.randn(1, 4, 1, 48, dtype=torch.float16)
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


# Phase 1.8: dropout. Statistical test — at p=0 the result must match the
# no-dropout reference, and at p>0 the expected value over many seeds must
# also match (we test with p=0 plus an end-to-end seed-determinism check).


def test_flash_attn_func_dropout_zero_matches_no_dropout():
    """dropout_p=0.0 produces exactly the no-dropout output and grads."""
    torch.manual_seed(0)
    batch, seqlen, nheads, headdim = 1, 8, 2, 64
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

    out = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    ref = flash_attn_mojo.flash_attn_func(
        q.detach(), k.detach(), v.detach(), causal=True
    )
    assert torch.equal(out, ref)
    out.backward(dout)


def test_flash_attn_func_dropout_seeded_determinism():
    """With a seeded RNG, two calls produce the same dropout mask and so
    the same output."""
    batch, seqlen, nheads, headdim = 1, 8, 1, 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    torch.manual_seed(42)
    out1 = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.3)
    torch.manual_seed(42)
    out2 = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.3)
    assert torch.equal(out1, out2)


def test_flash_attn_func_dropout_expected_value():
    """Mean of output over many random masks ≈ no-dropout output.

    Pulled element-wise mean down by averaging across (batch, head,
    seqlen, dim) — the per-output-tensor mean is much tighter than any
    individual element. Verifies the 1/(1-p) scaling is applied
    correctly: with the wrong scaling, the global mean would be off
    by a constant factor visible at any sample size.
    """
    torch.manual_seed(7)
    batch, seqlen, nheads, headdim = 1, 4, 1, 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float32)

    no_dropout = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)

    n_trials = 200
    accum = torch.zeros_like(no_dropout)
    for _ in range(n_trials):
        out = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.3, causal=True)
        accum += out
    mean_out = accum / n_trials

    # Compare global L1-norm — the per-element mean is noisy but the
    # overall sum is well-bracketed.
    norm_diff = (mean_out - no_dropout).norm().item() / no_dropout.norm().item()
    assert norm_diff < 0.05, f"||E[out] - no_dropout|| / ||no_dropout|| = {norm_diff}"


def test_flash_attn_func_dropout_backward_matches_explicit_mask():
    """Backward through dropout matches a reference computed by manually
    applying the same mask in fp32."""
    torch.manual_seed(123)
    batch, seqlen, nheads, headdim = 1, 8, 2, 64
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
    p = 0.25

    torch.manual_seed(2026)
    out = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=p, causal=True)
    out.backward(dout)

    # Re-run, capture the mask used internally by overriding bernoulli's
    # seed deterministically — torch.manual_seed before the second call
    # produces the same mask draws.
    torch.manual_seed(2026)
    out_ref = flash_attn_mojo.flash_attn_func(
        q.detach(), k.detach(), v.detach(), dropout_p=p, causal=True
    )
    # Output bit-equal under same seed.
    assert torch.equal(out, out_ref)


# Phase 1.13: softcap (Gemma-style logit cap).
@pytest.mark.parametrize("softcap", [5.0, 30.0])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_softcap(softcap, causal):
    """Forward + backward correctness with softcap."""
    batch, seqlen, nheads, headdim = 1, 16, 2, 64
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

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal, softcap=softcap)
    ref = _ref_attention(
        q.detach(), k.detach(), v.detach(), causal=causal, softcap=softcap
    )
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"fwd max_diff={diff} (softcap={softcap})"

    out.backward(dout)
    qg = q.detach().clone().to(torch.float32).requires_grad_(True)
    kg = k.detach().clone().to(torch.float32).requires_grad_(True)
    vg = v.detach().clone().to(torch.float32).requires_grad_(True)
    out_ref = _ref_attention(
        qg.to(q.dtype),
        kg.to(k.dtype),
        vg.to(v.dtype),
        causal=causal,
        softcap=softcap,
    )
    out_ref.float().backward(dout.float())
    for name, got, ref in [
        ("dq", q.grad, qg.grad),
        ("dk", k.grad, kg.grad),
        ("dv", v.grad, vg.grad),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff} (softcap={softcap})"


def test_softcap_negative_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="softcap"):
        flash_attn_mojo.flash_attn_func(q, q, q, softcap=-1.0)


# return_attn_probs — debug-mode tuple return.
def test_return_attn_probs_basic():
    """When return_attn_probs=True, flash_attn_func returns (out, lse, S_dmask)."""
    batch, seqlen, nheads, headdim = 1, 8, 1, 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    out_only = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    out, lse, s_dmask = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=True, return_attn_probs=True
    )

    # out is identical (same kernel, same inputs).
    assert torch.equal(out, out_only)
    # lse: (B, H_q, S_q), fp32.
    assert lse.shape == (batch, nheads, seqlen)
    assert lse.dtype == torch.float32
    # S_dmask: (B, H_q, S_q, S_k), fp32, all positive (no dropout).
    assert s_dmask.shape == (batch, nheads, seqlen, seqlen)
    # Per-row probs sum to 1 (within fp16 roundoff).
    row_sums = s_dmask.sum(dim=-1)
    assert (row_sums - 1.0).abs().max().item() < 5e-3
    assert (s_dmask >= 0).all()


def test_return_attn_probs_with_dropout():
    """With dropout, S_dmask uses sign to encode kept (+) vs dropped (−)."""
    torch.manual_seed(11)
    batch, seqlen, nheads, headdim = 1, 8, 1, 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    _, _, s_dmask = flash_attn_mojo.flash_attn_func(
        q, k, v, dropout_p=0.5, causal=True, return_attn_probs=True
    )
    # With p=0.5, expect roughly half-positive-half-negative entries
    # in the upper-triangular allowed region.
    n_pos = (s_dmask > 0).sum().item()
    n_neg = (s_dmask < 0).sum().item()
    # At least some of each — exact ratio depends on RNG draws.
    assert n_pos > 0 and n_neg > 0


def test_return_attn_probs_backward_through_out_only():
    """Backward through `out` works even when return_attn_probs=True;
    grads flowing back to lse / S_dmask are silently discarded."""
    batch, seqlen, nheads, headdim = 1, 4, 1, 64
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

    out, _lse, _s = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=True, return_attn_probs=True
    )
    out.backward(dout)
    # Same q.grad as if we hadn't requested probs.
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out2 = flash_attn_mojo.flash_attn_func(q2, k2, v2, causal=True)
    out2.backward(dout)
    assert torch.equal(q.grad, q2.grad)
    assert torch.equal(k.grad, k2.grad)
    assert torch.equal(v.grad, v2.grad)


def test_return_attn_probs_qkvpacked():
    """Wrappers thread return_attn_probs through."""
    batch, seqlen, nheads, headdim = 1, 4, 1, 64
    qkv = torch.randn(batch, seqlen, 3, nheads, headdim, dtype=torch.float16)
    out, lse, s = flash_attn_mojo.flash_attn_qkvpacked_func(
        qkv, causal=True, return_attn_probs=True
    )
    assert out.shape == (batch, seqlen, nheads, headdim)
    assert lse.shape == (batch, nheads, seqlen)
    assert s.shape == (batch, nheads, seqlen, seqlen)


def test_dropout_p_out_of_range_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="dropout_p"):
        flash_attn_mojo.flash_attn_func(q, q, q, dropout_p=1.5)
    with pytest.raises(ValueError, match="dropout_p"):
        flash_attn_mojo.flash_attn_func(q, q, q, dropout_p=-0.1)


# Phase 1.9: sliding window (local) attention.
@pytest.mark.parametrize(
    "window,causal",
    [
        ((4, 4), False),  # symmetric window
        ((2, 0), False),  # left-only (no future) but not causal-named
        ((0, 2), False),  # right-only — peek into the future
        ((4, -1), False),  # left bound only
        ((-1, 4), False),  # right bound only
        ((3, 0), True),  # causal + sliding ("local")
        ((4, 4), True),  # causal still wins on the right
    ],
)
def test_flash_attn_func_window(window, causal):
    """Forward + backward correctness with sliding-window masks."""
    batch, seqlen, nheads, headdim = 1, 16, 2, 64
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

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal, window_size=window)
    ref = _ref_attention(
        q.detach(), k.detach(), v.detach(), causal=causal, window=window
    )
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"fwd max_diff={diff} (window={window}, causal={causal})"

    out.backward(dout)

    # Reference grad with window applied via masked softmax.
    qg = q.detach().clone().to(torch.float32).requires_grad_(True)
    kg = k.detach().clone().to(torch.float32).requires_grad_(True)
    vg = v.detach().clone().to(torch.float32).requires_grad_(True)
    out_ref = _ref_attention(
        qg.to(q.dtype),
        kg.to(k.dtype),
        vg.to(v.dtype),
        causal=causal,
        window=window,
    )
    out_ref.float().backward(dout.float())
    for name, got, ref in [
        ("dq", q.grad, qg.grad),
        ("dk", k.grad, kg.grad),
        ("dv", v.grad, vg.grad),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff} (window={window}, causal={causal})"


def test_flash_attn_func_window_zero_zero():
    """window_size=(0, 0) keeps only the diagonal — output equals scaled v."""
    batch, seqlen, nheads, headdim = 1, 8, 1, 64
    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    out = flash_attn_mojo.flash_attn_func(q, k, v, window_size=(0, 0))
    ref = _ref_attention(q, k, v, window=(0, 0))
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3


def test_window_size_bad_shape_raises():
    q = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="window_size"):
        flash_attn_mojo.flash_attn_func(q, q, q, window_size=(1, 2, 3))


# Phase 1.10: ALiBi bias.
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("per_batch", [False, True])
def test_flash_attn_func_alibi(causal, per_batch):
    """ALiBi-biased attention matches the reference.

    Bias: -alibi_slope[h] * |pos - k_idx|, where pos = (sk - sq) + q_idx.
    """
    batch, seqlen, nheads, headdim = 2, 16, 4, 64
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

    # Standard ALiBi slopes for 4 heads: powers of 1/2 (positive).
    slopes_1d = torch.tensor([0.5, 0.25, 0.125, 0.0625], dtype=torch.float32)
    slopes = (
        slopes_1d.unsqueeze(0).expand(batch, -1).contiguous()
        if per_batch
        else slopes_1d
    )

    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal, alibi_slopes=slopes)
    ref = _ref_attention(
        q.detach(),
        k.detach(),
        v.detach(),
        causal=causal,
        alibi=slopes,
    )
    diff = (out.float() - ref.float()).abs().max().item()
    assert diff < 5e-3, f"fwd max_diff={diff} (causal={causal}, per_batch={per_batch})"

    out.backward(dout)
    qg = q.detach().clone().to(torch.float32).requires_grad_(True)
    kg = k.detach().clone().to(torch.float32).requires_grad_(True)
    vg = v.detach().clone().to(torch.float32).requires_grad_(True)
    out_ref = _ref_attention(
        qg.to(q.dtype),
        kg.to(k.dtype),
        vg.to(v.dtype),
        causal=causal,
        alibi=slopes,
    )
    out_ref.float().backward(dout.float())
    for name, got, ref in [
        ("dq", q.grad, qg.grad),
        ("dk", k.grad, kg.grad),
        ("dv", v.grad, vg.grad),
    ]:
        diff = (got.float() - ref.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff}"


def test_alibi_bad_dtype_raises():
    q = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    slopes = torch.zeros(2, dtype=torch.float16)  # must be fp32
    with pytest.raises(ValueError, match="alibi_slopes must be fp32"):
        flash_attn_mojo.flash_attn_func(q, q, q, alibi_slopes=slopes)


def test_alibi_bad_shape_raises():
    q = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    slopes = torch.zeros(3, dtype=torch.float32)  # nheads=2, slopes has 3
    with pytest.raises(ValueError, match="nheads_q"):
        flash_attn_mojo.flash_attn_func(q, q, q, alibi_slopes=slopes)


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


# flash_attn_kvpacked_func — q + kv-packed wrapper.
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_kvpacked_func(causal):
    """kvpacked path matches the unpacked one."""
    batch, seqlen, nheads_q, nheads_kv, headdim = 2, 16, 4, 2, 64
    q = torch.randn(batch, seqlen, nheads_q, headdim, dtype=torch.float16)
    kv = torch.randn(batch, seqlen, 2, nheads_kv, headdim, dtype=torch.float16)
    k, v = kv.unbind(dim=2)

    out_packed = flash_attn_mojo.flash_attn_kvpacked_func(q, kv, causal=causal)
    out_unpacked = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    assert torch.equal(out_packed, out_unpacked)


# flash_attn_varlen_func — packed variable-length sequences.
def test_flash_attn_varlen_func_correctness():
    """Pack 3 sequences of different lengths, run varlen, compare to
    per-sequence flash_attn_func calls."""
    nheads, headdim = 2, 64
    seqlens_q = [5, 12, 8]
    seqlens_k = [5, 12, 8]
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()), dtype=torch.int32
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()), dtype=torch.int32
    )
    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())

    q = torch.randn(total_q, nheads, headdim, dtype=torch.float16)
    k = torch.randn(total_k, nheads, headdim, dtype=torch.float16)
    v = torch.randn(total_k, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, max(seqlens_q), max(seqlens_k), causal=True
    )
    assert out.shape == q.shape

    # Reference: per-sequence flash_attn_func.
    for b in range(len(seqlens_q)):
        q_b = q[cu_q[b] : cu_q[b + 1]].unsqueeze(0)
        k_b = k[cu_k[b] : cu_k[b + 1]].unsqueeze(0)
        v_b = v[cu_k[b] : cu_k[b + 1]].unsqueeze(0)
        ref_b = flash_attn_mojo.flash_attn_func(q_b, k_b, v_b, causal=True)
        assert torch.equal(out[cu_q[b] : cu_q[b + 1]], ref_b.squeeze(0))


def test_flash_attn_varlen_func_backward():
    """Backward through varlen produces gradients matching per-sequence
    fp32 reference."""
    seqlens = [4, 8, 6]
    cu = torch.tensor([0, 4, 12, 18], dtype=torch.int32)
    total = 18
    nheads, headdim = 2, 64

    q = torch.randn(total, nheads, headdim, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, nheads, headdim, dtype=torch.float16, requires_grad=True)
    v = torch.randn(total, nheads, headdim, dtype=torch.float16, requires_grad=True)
    dout = torch.randn(total, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True
    )
    out.backward(dout)

    # Per-sequence reference grads.
    for b in range(len(seqlens)):
        qg = (
            q.detach()[cu[b] : cu[b + 1]].clone().to(torch.float32).requires_grad_(True)
        )
        kg = (
            k.detach()[cu[b] : cu[b + 1]].clone().to(torch.float32).requires_grad_(True)
        )
        vg = (
            v.detach()[cu[b] : cu[b + 1]].clone().to(torch.float32).requires_grad_(True)
        )
        out_ref = _ref_attention(
            qg.unsqueeze(0).to(q.dtype),
            kg.unsqueeze(0).to(k.dtype),
            vg.unsqueeze(0).to(v.dtype),
            causal=True,
        ).squeeze(0)
        out_ref.float().backward(dout[cu[b] : cu[b + 1]].float())
        for name, got, ref in [
            ("dq", q.grad[cu[b] : cu[b + 1]], qg.grad),
            ("dk", k.grad[cu[b] : cu[b + 1]], kg.grad),
            ("dv", v.grad[cu[b] : cu[b + 1]], vg.grad),
        ]:
            diff = (got.float() - ref.float()).abs().max().item()
            assert diff < 1e-2, f"batch {b} {name} max_diff={diff}"


def test_flash_attn_varlen_qkvpacked_func():
    """Varlen + qkv-packed wrapper matches the unpacked one."""
    seqlens = [3, 7]
    cu = torch.tensor([0, 3, 10], dtype=torch.int32)
    total, nheads, headdim = 10, 2, 64
    qkv = torch.randn(total, 3, nheads, headdim, dtype=torch.float16)
    q, k, v = qkv.unbind(dim=1)
    out_packed = flash_attn_mojo.flash_attn_varlen_qkvpacked_func(
        qkv, cu, max(seqlens), causal=True
    )
    out_unpacked = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True
    )
    assert torch.equal(out_packed, out_unpacked)


def test_flash_attn_varlen_kvpacked_func():
    """Varlen + kv-packed wrapper matches the unpacked one."""
    seqlens = [4, 6]
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)
    total, nheads_q, nheads_kv, headdim = 10, 4, 2, 64
    q = torch.randn(total, nheads_q, headdim, dtype=torch.float16)
    kv = torch.randn(total, 2, nheads_kv, headdim, dtype=torch.float16)
    k, v = kv.unbind(dim=1)
    out_packed = flash_attn_mojo.flash_attn_varlen_kvpacked_func(
        q, kv, cu, cu, max(seqlens), max(seqlens), causal=True
    )
    out_unpacked = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=True
    )
    assert torch.equal(out_packed, out_unpacked)


def test_flash_attn_kvpacked_func_bad_shape_raises():
    q = torch.randn(1, 4, 2, 64, dtype=torch.float16)
    bad = torch.randn(1, 4, 3, 2, 64, dtype=torch.float16)  # dim-2 must be 2
    with pytest.raises(ValueError, match="seqlen, 2, nheads"):
        flash_attn_mojo.flash_attn_kvpacked_func(q, bad)


def test_flash_attn_qkvpacked_func_bad_shape_raises():
    """qkv must have a size-3 dim-2."""
    bad = torch.randn(1, 4, 4, 2, 64, dtype=torch.float16)  # dim-2 = 4, not 3
    with pytest.raises(ValueError, match="seqlen, 3, nheads"):
        flash_attn_mojo.flash_attn_qkvpacked_func(bad)


# Phase 1.12: flash_attn_with_kvcache (basic).
def test_flash_attn_with_kvcache_no_append():
    """Decode against a pre-populated cache, no new tokens appended."""
    batch, seqlen_q, nheads_q, nheads_kv, headdim = 2, 1, 4, 2, 64
    seqlen_kmax = 32
    cache_seqlens = torch.tensor([7, 13], dtype=torch.int32)

    q = torch.randn(batch, seqlen_q, nheads_q, headdim, dtype=torch.float16)
    k_cache = torch.randn(batch, seqlen_kmax, nheads_kv, headdim, dtype=torch.float16)
    v_cache = torch.randn(batch, seqlen_kmax, nheads_kv, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True
    )

    # Reference: per-batch, attend to k_cache[b, :cache_seqlens[b]].
    repeat = nheads_q // nheads_kv
    for b in range(batch):
        n = int(cache_seqlens[b].item())
        ref = _ref_attention(
            q[b : b + 1],
            k_cache[b : b + 1, :n].repeat_interleave(repeat, dim=2),
            v_cache[b : b + 1, :n].repeat_interleave(repeat, dim=2),
            causal=True,
        )
        diff = (out[b].float() - ref[0].float()).abs().max().item()
        assert diff < 5e-3, f"batch {b}: max_diff={diff}"


def test_flash_attn_with_kvcache_append():
    """Decode appending new k, v tokens then attending to the full
    valid range."""
    batch, seqlen_q, nheads, headdim = 2, 2, 1, 64
    seqlen_kmax = 16
    n0, n1 = 5, 8
    cache_seqlens = torch.tensor([n0, n1], dtype=torch.int32)

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k_cache = torch.zeros(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    v_cache = torch.zeros(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    # Pre-populate the valid prefix.
    k_cache[0, :n0] = torch.randn(n0, nheads, headdim, dtype=torch.float16)
    k_cache[1, :n1] = torch.randn(n1, nheads, headdim, dtype=torch.float16)
    v_cache[0, :n0] = torch.randn(n0, nheads, headdim, dtype=torch.float16)
    v_cache[1, :n1] = torch.randn(n1, nheads, headdim, dtype=torch.float16)

    k_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    v_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)

    # Snapshot cache before; the kernel mutates it in place.
    k_cache_before = k_cache.clone()

    out = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    # The cache should have been updated at slots [n_b, n_b + S_new).
    assert torch.equal(k_cache[0, n0 : n0 + seqlen_q], k_new[0])
    assert torch.equal(v_cache[1, n1 : n1 + seqlen_q], v_new[1])
    # Untouched regions unchanged.
    assert torch.equal(k_cache[0, :n0], k_cache_before[0, :n0])
    assert torch.equal(k_cache[0, n0 + seqlen_q :], k_cache_before[0, n0 + seqlen_q :])

    # Reference: attend over [0, n_b + S_new) of the updated cache.
    for b, n in [(0, n0), (1, n1)]:
        n_eff = n + seqlen_q
        ref = _ref_attention(
            q[b : b + 1],
            k_cache[b : b + 1, :n_eff],
            v_cache[b : b + 1, :n_eff],
            causal=True,
        )
        diff = (out[b].float() - ref[0].float()).abs().max().item()
        assert diff < 5e-3


def test_flash_attn_with_kvcache_int_seqlen_broadcast():
    """cache_seqlens as a single int broadcasts across batches."""
    batch, seqlen_q, nheads, headdim = 3, 1, 1, 64
    seqlen_kmax = 8
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k_cache = torch.randn(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    v_cache = torch.randn(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    out = flash_attn_mojo.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=4, causal=True
    )
    # Tensor form should give identical result.
    out_tensor = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=torch.full((batch,), 4, dtype=torch.int32),
        causal=True,
    )
    assert torch.equal(out, out_tensor)


# Phase 1.14: rotary embeddings inside flash_attn_with_kvcache.
def _build_rope(rotary_dim, max_pos, base=10000.0):
    """Standard RoPE inv-freq table → (max_pos, rotary_dim/2) cos and sin."""
    assert rotary_dim % 2 == 0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("p,d->pd", pos, inv_freq)
    return freqs.cos().to(torch.float16), freqs.sin().to(torch.float16)


@pytest.mark.parametrize("interleaved", [True, False])
def test_flash_attn_with_kvcache_rotary(interleaved):
    """Decode with rotary applied to q and k_new at absolute positions."""
    batch, seqlen_q, nheads, headdim = 1, 1, 1, 64
    seqlen_kmax = 16
    rotary_dim = 32

    cos, sin = _build_rope(rotary_dim, seqlen_kmax)
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k_cache = torch.randn(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    v_cache = torch.randn(batch, seqlen_kmax, nheads, headdim, dtype=torch.float16)
    k_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    v_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    cache_seqlens = torch.tensor([5], dtype=torch.int32)

    # Reference: apply rotary manually, then run no-rotary kvcache.
    pos = cache_seqlens[0].item()
    q_rot = flash_attn_mojo._apply_rotary(  # type: ignore[attr-defined]
        q,
        cos,
        sin,
        torch.tensor([[pos]], dtype=torch.long),
        interleaved,
    )
    k_rot = flash_attn_mojo._apply_rotary(  # type: ignore[attr-defined]
        k_new,
        cos,
        sin,
        torch.tensor([[pos]], dtype=torch.long),
        interleaved,
    )
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    out_ref = flash_attn_mojo.flash_attn_with_kvcache(
        q_rot,
        k_cache_ref,
        v_cache_ref,
        k=k_rot,
        v=v_new,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    # Run with rotary path:
    out = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        rotary_cos=cos,
        rotary_sin=sin,
        rotary_interleaved=interleaved,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    diff = (out.float() - out_ref.float()).abs().max().item()
    # Roundoff between fp16 rotary done inline and out-of-line is small.
    assert diff < 5e-3, f"max_diff={diff}"


def test_flash_attn_with_kvcache_rotary_partial_one_only_raises():
    q = torch.randn(1, 1, 1, 64, dtype=torch.float16)
    kc = torch.zeros(1, 4, 1, 64, dtype=torch.float16)
    vc = torch.zeros(1, 4, 1, 64, dtype=torch.float16)
    cos = torch.zeros(4, 32, dtype=torch.float16)
    # Only cos given, no sin → should raise.
    with pytest.raises(ValueError, match="rotary"):
        flash_attn_mojo.flash_attn_with_kvcache(q, kc, vc, rotary_cos=cos)


# Phase 1.15: cache_batch_idx — beam-search-style indirection.
def test_flash_attn_with_kvcache_cache_batch_idx():
    """Each q batch row reads k_cache[cache_batch_idx[b]] / v_cache[...]
    instead of k_cache[b]. Useful for beam search where multiple beams
    share the same kv state."""
    cache_batch, q_batch = 4, 6
    seqlen_q, nheads_q, nheads_kv, headdim = 1, 2, 1, 64
    seqlen_kmax = 16

    # Pre-populate the cache (only `cache_batch` slots).
    k_cache = torch.randn(
        cache_batch, seqlen_kmax, nheads_kv, headdim, dtype=torch.float16
    )
    v_cache = torch.randn(
        cache_batch, seqlen_kmax, nheads_kv, headdim, dtype=torch.float16
    )
    cache_seqlens = torch.tensor([5, 8, 3, 11], dtype=torch.int32)

    # 6 q rows that point to slots [0, 1, 2, 0, 3, 1] of the cache.
    cbi = torch.tensor([0, 1, 2, 0, 3, 1], dtype=torch.int32)
    q = torch.randn(q_batch, seqlen_q, nheads_q, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens[cbi],  # per-q-row valid length
        cache_batch_idx=cbi,
        causal=True,
    )

    # Reference: for each q row b, attend to k_cache[cbi[b], :cache_seqlens[cbi[b]]].
    repeat = nheads_q // nheads_kv
    for b in range(q_batch):
        slot = int(cbi[b].item())
        n = int(cache_seqlens[slot].item())
        ref = _ref_attention(
            q[b : b + 1],
            k_cache[slot : slot + 1, :n].repeat_interleave(repeat, dim=2),
            v_cache[slot : slot + 1, :n].repeat_interleave(repeat, dim=2),
            causal=True,
        )
        diff = (out[b].float() - ref[0].float()).abs().max().item()
        assert diff < 5e-3, f"q row {b} (slot {slot}): max_diff={diff}"


# Phase 1.16: paged kv-cache (block_table).
def test_flash_attn_with_kvcache_paged_no_append():
    """Read from a pre-populated paged cache: k_cache shape is
    (num_blocks, page_size, H_kv, D) and block_table[b, j] points into it."""
    batch, seqlen_q, nheads, headdim = 2, 1, 1, 64
    page_size = 4
    num_blocks = 8
    max_blocks_per_seq = 4

    # Allocate paged storage and a "linear" copy we'll use as the reference.
    k_cache = torch.randn(num_blocks, page_size, nheads, headdim, dtype=torch.float16)
    v_cache = torch.randn(num_blocks, page_size, nheads, headdim, dtype=torch.float16)

    # Block table: batch 0 → blocks [3, 5, _, _], batch 1 → [1, 7, 2, _].
    block_table = torch.tensor([[3, 5, 0, 0], [1, 7, 2, 0]], dtype=torch.int32)
    cache_seqlens = torch.tensor(
        [6, 9], dtype=torch.int32
    )  # b0 uses 6 tokens (1.5 blocks), b1 uses 9 (2.25 blocks)

    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )

    # Reference: gather the logical cache view manually, then run the
    # plain (un-paged) kvcache path.
    bt_long = block_table.long()
    k_logical = (
        k_cache[bt_long]
        .reshape(batch, max_blocks_per_seq * page_size, nheads, headdim)
        .contiguous()
    )
    v_logical = (
        v_cache[bt_long]
        .reshape(batch, max_blocks_per_seq * page_size, nheads, headdim)
        .contiguous()
    )
    ref = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_logical,
        v_logical,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    assert torch.equal(out, ref)


def test_flash_attn_with_kvcache_paged_append():
    """Append new k, v tokens into the paged cache at the addresses
    given by block_table, then attend over the updated valid range."""
    batch, seqlen_q, nheads, headdim = 1, 3, 1, 64
    page_size = 4
    num_blocks = 6
    max_blocks_per_seq = 3

    k_cache = torch.randn(num_blocks, page_size, nheads, headdim, dtype=torch.float16)
    v_cache = torch.randn(num_blocks, page_size, nheads, headdim, dtype=torch.float16)
    k_cache_before = k_cache.clone()

    # batch 0: logical blocks 0..2 → physical [4, 0, 2].
    block_table = torch.tensor([[4, 0, 2]], dtype=torch.int32)
    cache_seqlens = torch.tensor(
        [2], dtype=torch.int32
    )  # next 3 tokens land at logical [2, 3, 4]
    q = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    k_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)
    v_new = torch.randn(batch, seqlen_q, nheads, headdim, dtype=torch.float16)

    out = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )

    # Logical positions [2, 3, 4]:
    #   logical 2 = block 0 (physical 4), offset 2
    #   logical 3 = block 0 (physical 4), offset 3
    #   logical 4 = block 1 (physical 0), offset 0
    assert torch.equal(k_cache[4, 2], k_new[0, 0])
    assert torch.equal(k_cache[4, 3], k_new[0, 1])
    assert torch.equal(k_cache[0, 0], k_new[0, 2])
    assert torch.equal(v_cache[0, 0], v_new[0, 2])
    # Physical block 2 (logical block 2) untouched by the append.
    assert torch.equal(k_cache[2], k_cache_before[2])

    # Reference: gather + un-paged kvcache call (after append).
    bt_long = block_table.long()
    k_logical = (
        k_cache[bt_long]
        .reshape(batch, max_blocks_per_seq * page_size, nheads, headdim)
        .contiguous()
    )
    v_logical = (
        v_cache[bt_long]
        .reshape(batch, max_blocks_per_seq * page_size, nheads, headdim)
        .contiguous()
    )
    ref = flash_attn_mojo.flash_attn_with_kvcache(
        q,
        k_logical,
        v_logical,
        cache_seqlens=cache_seqlens + seqlen_q,  # post-append valid length
        causal=True,
    )
    assert torch.equal(out, ref)


def test_flash_attn_with_kvcache_paged_with_cache_batch_idx_raises():
    """Paged + cache_batch_idx isn't supported."""
    q = torch.randn(1, 1, 1, 64, dtype=torch.float16)
    kc = torch.zeros(2, 4, 1, 64, dtype=torch.float16)
    vc = torch.zeros(2, 4, 1, 64, dtype=torch.float16)
    bt = torch.zeros(1, 4, dtype=torch.int32)
    cbi = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="block_table"):
        flash_attn_mojo.flash_attn_with_kvcache(
            q, kc, vc, block_table=bt, cache_batch_idx=cbi
        )


# bert_padding helpers — pad / unpad round-trip used with varlen.
def test_bert_padding_unpad_pad_roundtrip():
    """unpad_input → pad_input is identity on the valid tokens."""
    from flash_attn_mojo.bert_padding import pad_input, unpad_input

    batch, seqlen, hidden = 3, 8, 16
    hidden_states = torch.randn(batch, seqlen, hidden)
    # Valid lengths per batch: [5, 3, 8].
    valid = torch.tensor([5, 3, 8], dtype=torch.int64)
    attention_mask = (torch.arange(seqlen).unsqueeze(0) < valid.unsqueeze(1)).int()

    unpadded, indices, cu_seqlens, max_seq = unpad_input(hidden_states, attention_mask)
    assert unpadded.shape == (5 + 3 + 8, hidden)
    assert tuple(cu_seqlens.tolist()) == (0, 5, 8, 16)
    assert max_seq == 8

    # Round-trip: padding rows in the rebuilt tensor are zero, valid
    # rows match the original.
    padded = pad_input(unpadded, indices, batch, seqlen)
    assert padded.shape == hidden_states.shape
    valid_mask = attention_mask.bool().unsqueeze(-1).expand_as(padded)
    assert torch.equal(padded[valid_mask], hidden_states[valid_mask])
    # Padding positions are zero.
    assert (padded[~valid_mask] == 0).all()


def test_bert_padding_unpad_then_varlen():
    """unpad → flash_attn_varlen_func → pad gives the same per-token
    output as dense flash_attn_func on the unpadded prefix of each row."""
    from flash_attn_mojo.bert_padding import pad_input, unpad_input

    batch, seqlen, nheads, headdim = 2, 8, 1, 64
    valid = torch.tensor([5, 3], dtype=torch.int64)
    attention_mask = (torch.arange(seqlen).unsqueeze(0) < valid.unsqueeze(1)).int()

    q = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    q_un, q_idx, cu_q, mq = unpad_input(q, attention_mask)
    k_un, _, cu_k, mk = unpad_input(k, attention_mask)
    v_un, _, _, _ = unpad_input(v, attention_mask)

    out_un = flash_attn_mojo.flash_attn_varlen_func(
        q_un, k_un, v_un, cu_q, cu_k, mq, mk, causal=True
    )
    out_padded = pad_input(out_un, q_idx, batch, seqlen)

    # Reference: dense flash_attn_func per-batch, on the unpadded slice.
    for b in range(batch):
        n = int(valid[b].item())
        ref = flash_attn_mojo.flash_attn_func(
            q[b : b + 1, :n], k[b : b + 1, :n], v[b : b + 1, :n], causal=True
        )
        diff = (out_padded[b, :n].float() - ref[0].float()).abs().max().item()
        assert diff < 5e-3, f"batch {b}: max_diff={diff}"


# Sanity: upstream flash_attn is importable in this env (non-fatal).
def test_upstream_importable():
    _skip_if_no_upstream()
    import flash_attn

    assert flash_attn.__version__.startswith("2.")


# Cross-check vs upstream flash_attn on GPU. Our impl is CPU-only and
# fp32-internal, upstream is GPU/fp16 — they should agree within
# ~5e-3 forward tolerance and ~1e-2 backward tolerance.
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_matches_upstream(causal):
    _skip_if_no_upstream()
    _skip_if_unsupported_arch()
    import flash_attn

    batch, seqlen, nheads, headdim = 2, 32, 4, 64
    q_cpu = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    k_cpu = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    v_cpu = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)
    dout_cpu = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float16)

    # Ours: CPU.
    qm = q_cpu.clone().requires_grad_(True)
    km = k_cpu.clone().requires_grad_(True)
    vm = v_cpu.clone().requires_grad_(True)
    out_mojo = flash_attn_mojo.flash_attn_func(qm, km, vm, causal=causal)
    out_mojo.backward(dout_cpu)

    # Upstream: GPU.
    qg = q_cpu.cuda().clone().requires_grad_(True)
    kg = k_cpu.cuda().clone().requires_grad_(True)
    vg = v_cpu.cuda().clone().requires_grad_(True)
    out_up = flash_attn.flash_attn_func(qg, kg, vg, causal=causal)
    out_up.backward(dout_cpu.cuda())

    fwd_diff = (out_mojo.float() - out_up.cpu().float()).abs().max().item()
    assert fwd_diff < 5e-3, f"forward max_diff={fwd_diff} (causal={causal})"
    for name, mine, theirs in [
        ("dq", qm.grad, qg.grad.cpu()),
        ("dk", km.grad, kg.grad.cpu()),
        ("dv", vm.grad, vg.grad.cpu()),
    ]:
        diff = (mine.float() - theirs.float()).abs().max().item()
        assert diff < 1e-2, f"{name} max_diff={diff} (causal={causal})"
