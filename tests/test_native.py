"""Tests for the pure-Mojo native path (no MAX framework).

The native extension currently only specializes fp16 / width=4 /
has_bias=True / activation="silu" / no initial_states / no return_final_states.

Each test runs on every available device. CPU is always present; CUDA is
exercised only if a GPU is detected.
"""

import pytest
import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


# Devices to run every test against. CPU is always available; CUDA is
# parametrised in but skipped per-test if the box has no GPU. fp16 on CPU
# is supported on PyTorch 2.x — the native CPU kernel computes everything
# in fp32 internally and casts back at the boundary.
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.fixture(params=_DEVICES)
def device(request):
    return request.param


# Activations the public API accepts. silu and swish are the same op; None
# is the bias-only forward (no activation). Tests run all three.
@pytest.fixture(params=[None, "silu", "swish"])
def activation(request):
    return request.param


# `bias=None` is the bias-free forward (`out = silu(conv1d(x, w))`). The
# kernel's `has_bias` comptime parameter selects the path.
@pytest.fixture(params=[True, False], ids=["with_bias", "no_bias"])
def bias_present(request):
    return request.param


def _make_bias(D, *, dtype, device, present, requires_grad=False):
    if not present:
        return None
    return torch.randn(D, dtype=dtype, device=device, requires_grad=requires_grad)


def _expected(x, weight, bias, activation):
    return causal_conv1d_ref(x, weight, bias=bias, activation=activation)


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_contiguous(device, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device)
    weight = torch.randn(D, W, dtype=torch.float16, device=device)
    bias = _make_bias(D, dtype=torch.float16, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x, weight, bias, activation))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_x_seq_stride_not_one(device, activation, bias_present):
    """x is (B, D, L) but came from a transpose so stride(2) != 1."""
    B, D, L = 2, 64, 128
    W = 4
    # Allocate as (B, L, D) then transpose to expose (B, D, L) with non-unit
    # innermost stride. After transpose: stride(2) == D, stride(1) == 1.
    x_view = torch.randn(B, L, D, dtype=torch.float16, device=device).transpose(1, 2)
    assert x_view.shape == (B, D, L)
    assert not x_view.is_contiguous()
    assert x_view.stride(2) != 1

    weight = torch.randn(D, W, dtype=torch.float16, device=device)
    bias = _make_bias(D, dtype=torch.float16, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_view, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x_view, weight, bias, activation))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_x_sliced(device, activation, bias_present):
    """x is a slice of a larger tensor (contiguous stride=1 on last dim, but
    leading strides are larger than the slice's shape would imply if it
    were contiguous)."""
    B, D, L = 2, 64, 128
    W = 4
    big_x = torch.randn(B, D, L * 2, dtype=torch.float16, device=device)
    x_slice = big_x[:, :, :L]
    assert x_slice.shape == (B, D, L)
    assert not x_slice.is_contiguous()
    assert x_slice.stride(2) == 1
    assert x_slice.stride(1) == L * 2  # gap between rows

    weight = torch.randn(D, W, dtype=torch.float16, device=device)
    bias = _make_bias(D, dtype=torch.float16, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_slice, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x_slice, weight, bias, activation))
    assert diff < 2e-2, f"max_diff={diff}"


def test_noncontiguous_weight(device, activation, bias_present):
    """weight is (D, W) but stride(1) != 1 (e.g., from transpose)."""
    B, D, L = 2, 64, 128
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device)
    weight_view = torch.randn(W, D, dtype=torch.float16, device=device).transpose(0, 1)
    assert weight_view.shape == (D, W)
    assert not weight_view.is_contiguous()
    assert weight_view.stride(1) != 1

    bias = _make_bias(D, dtype=torch.float16, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight_view, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x, weight_view, bias, activation))
    assert diff < 2e-2, f"max_diff={diff}"


# ===---------- backward / autograd ----------=== #


def _ref_grads(x, weight, bias, dout, activation):
    """Reference: pytorch impl of causal_conv1d_fwd, backward via autograd."""
    x_g = x.detach().requires_grad_()
    w_g = weight.detach().requires_grad_()
    b_g = bias.detach().requires_grad_() if bias is not None else None
    D, W = w_g.shape
    L = x_g.shape[-1]
    pre = F.conv1d(x_g, w_g.unsqueeze(1), b_g, padding=W - 1, groups=D)[..., :L]
    out = F.silu(pre) if activation in ("silu", "swish") else pre
    out.backward(dout)
    return x_g.grad, w_g.grad, (b_g.grad if b_g is not None else None)


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_backward_matches_pytorch_ref(device, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=torch.float16, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=torch.float16, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, activation)

    # fp16 grads accumulate over (B, L); allow looser tol than forward.
    assert _max_diff(x.grad, dx_ref) < 1e-1, f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    assert _max_diff(weight.grad, dw_ref) < 1.0, (
        f"dw max_diff={_max_diff(weight.grad, dw_ref)} (sums over B*L={B * L} terms)"
    )
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < 1.0, (
            f"db max_diff={_max_diff(bias.grad, db_ref)} (sums over B*L={B * L} terms)"
        )
    else:
        assert bias is None and db_ref is None


def test_backward_shapes_and_dtypes(device, activation, bias_present):
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=torch.float16, device=device, present=bias_present, requires_grad=True
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


# ===---------- zero-sized tensors ----------=== #
# Each (B, D, L) configuration with at least one zero dimension. A
# well-behaved op should accept these and return correctly-shaped
# (empty) outputs / gradients without raising. On the GPU side
# `enqueue_function` rejects any `grid_dim == 0`, so the launchers
# need an explicit early-out.


@pytest.mark.parametrize("shape", [(0, 64, 128), (2, 0, 128), (2, 64, 0), (0, 0, 0)])
def test_zero_sized_forward(device, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device)
    weight = torch.randn(D, W, dtype=torch.float16, device=device)
    bias = _make_bias(D, dtype=torch.float16, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.numel() == 0


@pytest.mark.parametrize("shape", [(0, 64, 128), (2, 0, 128), (2, 64, 0), (0, 0, 0)])
def test_zero_sized_backward(device, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=torch.float16, device=device, present=bias_present, requires_grad=True
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
