"""Backward-path tests for the pure-Mojo native extension.

Covers `causal_conv1d_mojo.causal_conv1d_fn` followed by `.backward()`:
the standard pytorch-reference dx/dw/db match, shape/dtype invariants,
the final_states gradient tail, initial_states gradients, seq_idx
segmented backward, the width sweep, and zero-sized tensors. The
forward-only tests live in `test_fwd.py`; the single-step update tests
live in `test_update.py`.

Each test runs on every available device + every supported dtype. CPU is
always present; CUDA is exercised only if a GPU is detected.
"""

import pytest
import torch

import causal_conv1d_mojo

from _helpers import (
    _DW_TOL,
    _DX_TOL,
    _make_bias,
    _max_diff,
    _ref_grads,
    _ref_grads_with_seq_idx,
)


# ===---------- width sweep (2 / 3 / 4) ----------=== #


def test_width_backward(device, dtype, width, bias_present):
    B, D, L = 2, 32, 128
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, width, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype]
    assert _max_diff(weight.grad, dw_ref) < _DW_TOL[dtype]
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < _DW_TOL[dtype]


# ===---------- backward / autograd ----------=== #


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_backward_matches_pytorch_ref(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, activation)

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    )
    assert _max_diff(weight.grad, dw_ref) < _DW_TOL[dtype], (
        f"dw max_diff={_max_diff(weight.grad, dw_ref)} (sums over B*L={B * L} terms)"
    )
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < _DW_TOL[dtype], (
            f"db max_diff={_max_diff(bias.grad, db_ref)} (sums over B*L={B * L} terms)"
        )
    else:
        assert bias is None and db_ref is None


def test_backward_shapes_and_dtypes(device, dtype, activation, bias_present):
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    if bias_present:
        assert bias.grad.shape == bias.shape and bias.grad.dtype == bias.dtype


# ===---------- final_states backward ----------=== #


def test_final_states_backward(device, dtype, width, bias_present):
    """Gradient w.r.t. final_states is added to the matching tail of dx.

    final_states[b, c, i] = x[b, c, seqlen - (W-1) + i] for the
    seqlen >= W-1 case, so dfinal_states[b, c, i] flows directly back
    into dx[b, c, seqlen - (W-1) + i] in addition to the conv-path dx
    contribution.
    """
    B, D, L, W = 2, 16, 64, width
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)
    dfs = torch.randn(B, D, W - 1, dtype=dtype, device=device)

    out, fs = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu", return_final_states=True
    )
    torch.autograd.backward([out, fs], [dout, dfs])

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    dx_ref = dx_ref.clone()
    dx_ref[..., -(W - 1) :] += dfs.to(dx_ref.dtype)

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    )
    assert _max_diff(weight.grad, dw_ref) < _DW_TOL[dtype]
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < _DW_TOL[dtype]


# ===---------- initial_states backward ----------=== #


def test_initial_states_backward(device, dtype, width, bias_present):
    """Backward through initial_states: dx, dw, dbias, and dinitial_states
    all match the cat([initial_states, x]) reference. The kernel reads
    initial_states for the silu' recomputation in chunk 0 / tidx 0 and
    accumulates the boundary dweight contribution; dinitial_states is
    derived from dpre[0..W-2] with the anti-causal weight kernel.
    """
    from causal_conv1d_mojo import causal_conv1d_ref

    B, D, L = 2, 16, 64
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, width, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    init = torch.randn(B, D, width - 1, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, initial_states=init, activation="silu"
    )
    out.backward(dout)

    # Reference: cat([init, x], -1) -> standard causal_conv1d_ref ->
    # autograd. Slice the resulting dx into dinit + dx parts.
    x_ref = x.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    init_ref = init.detach().clone().requires_grad_()
    out_ref = causal_conv1d_ref(
        torch.cat([init_ref, x_ref], dim=-1),
        w_ref,
        bias=b_ref,
        initial_states=None,
        activation="silu",
    )[..., width - 1 :]
    out_ref.backward(dout)

    assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, x_ref.grad)}"
    )
    assert _max_diff(weight.grad, w_ref.grad) < _DW_TOL[dtype], (
        f"dw max_diff={_max_diff(weight.grad, w_ref.grad)}"
    )
    if bias_present:
        assert _max_diff(bias.grad, b_ref.grad) < _DW_TOL[dtype], (
            f"db max_diff={_max_diff(bias.grad, b_ref.grad)}"
        )
    # dinitial_states correctness — main feature. Tighter tolerance than
    # dweight since the sum is over much fewer terms (W-1, not B*L).
    assert _max_diff(init.grad, init_ref.grad) < _DX_TOL[dtype], (
        f"dinit max_diff={_max_diff(init.grad, init_ref.grad)}"
    )


# ===---------- seq_idx backward ----------=== #


@pytest.mark.parametrize(
    "seq_idx_pattern", ["single", "two_equal", "varied", "with_padding"]
)
def test_seq_idx_backward(device, dtype, seq_idx_pattern, activation, bias_present):
    """Backward through seq_idx: dx/dw/db match the segmented reference.

    For each seq_idx run, only positions in that run contributed to
    each other in the forward; padding positions produced zero output
    so their dpre is zero. The backward must reproduce that segmented
    flow.
    """
    B, D, L, W = 2, 16, 64, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    if seq_idx_pattern == "single":
        seq_idx = torch.zeros(B, L, dtype=torch.int32, device=device)
    elif seq_idx_pattern == "two_equal":
        seq_idx = torch.cat(
            [
                torch.zeros(B, L // 2, dtype=torch.int32, device=device),
                torch.ones(B, L - L // 2, dtype=torch.int32, device=device),
            ],
            dim=1,
        )
    elif seq_idx_pattern == "varied":
        seq_idx = torch.cat(
            [
                torch.zeros(B, 10, dtype=torch.int32, device=device),
                torch.ones(B, 25, dtype=torch.int32, device=device),
                torch.full((B, L - 35), 2, dtype=torch.int32, device=device),
            ],
            dim=1,
        )
    else:  # with_padding
        seq_idx = torch.cat(
            [
                torch.zeros(B, 16, dtype=torch.int32, device=device),
                torch.full((B, 16), -1, dtype=torch.int32, device=device),
                torch.ones(B, L - 32, dtype=torch.int32, device=device),
            ],
            dim=1,
        )

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, seq_idx=seq_idx, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads_with_seq_idx(
        x, weight, bias, seq_idx, dout, activation
    )

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}, pattern={seq_idx_pattern}"
    )
    assert _max_diff(weight.grad, dw_ref) < _DW_TOL[dtype], (
        f"dw max_diff={_max_diff(weight.grad, dw_ref)}, pattern={seq_idx_pattern}"
    )
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < _DW_TOL[dtype], (
            f"db max_diff={_max_diff(bias.grad, db_ref)}, pattern={seq_idx_pattern}"
        )


# ===---------- zero-sized tensors ----------=== #


@pytest.mark.parametrize("shape", [(0, 64, 128), (2, 0, 128), (2, 64, 0), (0, 0, 0)])
def test_zero_sized_backward(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    if bias_present:
        assert bias.grad.shape == bias.shape and bias.grad.dtype == bias.dtype
