"""Tests for the pure-Mojo native path (no MAX framework).

The native extension currently only specializes fp16 / width=4 /
has_bias=True / activation="silu" / no initial_states / no return_final_states.
"""

import pytest
import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native path is CUDA-only"
)


def _expected(x, weight, bias):
    return causal_conv1d_ref(x, weight, bias=bias, activation="silu")


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_contiguous(shape):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")

    diff = _max_diff(out, _expected(x, weight, bias))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_x_seq_stride_not_one():
    """x is (B, D, L) but came from a transpose so stride(2) != 1."""
    B, D, L = 2, 64, 128
    W = 4
    # Allocate as (B, L, D) then transpose to expose (B, D, L) with non-unit
    # innermost stride. After transpose: stride(2) == D, stride(1) == 1.
    x_view = torch.randn(B, L, D, dtype=torch.float16, device="cuda").transpose(1, 2)
    assert x_view.shape == (B, D, L)
    assert not x_view.is_contiguous()
    assert x_view.stride(2) != 1

    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_view, weight, bias=bias, activation="silu"
    )

    diff = _max_diff(out, _expected(x_view, weight, bias))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_x_sliced():
    """x is a slice of a larger tensor (contiguous stride=1 on last dim, but
    leading strides are larger than the slice's shape would imply if it
    were contiguous)."""
    B, D, L = 2, 64, 128
    W = 4
    big_x = torch.randn(B, D, L * 2, dtype=torch.float16, device="cuda")
    x_slice = big_x[:, :, :L]
    assert x_slice.shape == (B, D, L)
    assert not x_slice.is_contiguous()
    assert x_slice.stride(2) == 1
    assert x_slice.stride(1) == L * 2  # gap between rows

    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_slice, weight, bias=bias, activation="silu"
    )

    diff = _max_diff(out, _expected(x_slice, weight, bias))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_weight():
    """weight is (D, W) but stride(1) != 1 (e.g., from transpose)."""
    B, D, L = 2, 64, 128
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight_view = torch.randn(W, D, dtype=torch.float16, device="cuda").transpose(0, 1)
    assert weight_view.shape == (D, W)
    assert not weight_view.is_contiguous()
    assert weight_view.stride(1) != 1

    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight_view, bias=bias, activation="silu"
    )

    diff = _max_diff(out, _expected(x, weight_view, bias))
    assert diff < 2e-2, f"max_diff={diff}"


# ===---------- backward / autograd ----------=== #


def _ref_grads(x, weight, bias, dout):
    """Reference: pytorch impl of causal_conv1d_fwd, backward via autograd."""
    x_g = x.detach().requires_grad_()
    w_g = weight.detach().requires_grad_()
    b_g = bias.detach().requires_grad_()
    D, W = w_g.shape
    L = x_g.shape[-1]
    pre = F.conv1d(x_g, w_g.unsqueeze(1), b_g, padding=W - 1, groups=D)[..., :L]
    out = F.silu(pre)
    out.backward(dout)
    return x_g.grad, w_g.grad, b_g.grad


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_backward_matches_pytorch_ref(shape):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda", requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=torch.float16, device="cuda", requires_grad=True)
    dout = torch.randn(B, D, L, dtype=torch.float16, device="cuda")

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout)

    # fp16 grads accumulate over (B, L); allow looser tol than forward.
    assert _max_diff(x.grad, dx_ref) < 1e-1, f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    assert _max_diff(weight.grad, dw_ref) < 1.0, (
        f"dw max_diff={_max_diff(weight.grad, dw_ref)} (sums over B*L={B * L} terms)"
    )
    assert _max_diff(bias.grad, db_ref) < 1.0, (
        f"db max_diff={_max_diff(bias.grad, db_ref)} (sums over B*L={B * L} terms)"
    )


def test_backward_shapes_and_dtypes():
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda", requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=torch.float16, device="cuda", requires_grad=True)
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out.backward(dout)

    assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    assert bias.grad.shape == bias.shape and bias.grad.dtype == bias.dtype
