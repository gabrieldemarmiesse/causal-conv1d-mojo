"""Tests for the pure-Mojo native path (no MAX framework).

The native extension currently only specializes fp16 / width=4 /
has_bias=True / activation="silu" / no initial_states / no return_final_states.
"""
import pytest
import torch

from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native path is CUDA-only")


@pytest.fixture(scope="module")
def native_mod():
    from causal_conv1d_mojo._native import causal_conv1d_native
    return causal_conv1d_native


def _call(native_mod, x, weight, bias, out):
    native_mod.causal_conv1d_fwd_fp16_w4_silu_bias(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        x.shape[0], x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        torch.cuda.current_stream().cuda_stream,
    )


def _expected(x, weight, bias):
    return causal_conv1d_ref(x, weight, bias=bias, activation="silu")


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_contiguous(native_mod, shape):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)

    _call(native_mod, x, weight, bias, out)
    torch.cuda.synchronize()

    diff = _max_diff(out, _expected(x, weight, bias))
    assert diff < 1e-2, f"max_diff={diff}"


def test_noncontiguous_x_seq_stride_not_one(native_mod):
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
    out = torch.empty(B, D, L, dtype=torch.float16, device="cuda")

    _call(native_mod, x_view, weight, bias, out)
    torch.cuda.synchronize()

    diff = _max_diff(out, _expected(x_view, weight, bias))
    assert diff < 1e-2, f"max_diff={diff}"


def test_noncontiguous_x_sliced(native_mod):
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
    out = torch.empty(B, D, L, dtype=torch.float16, device="cuda")

    _call(native_mod, x_slice, weight, bias, out)
    torch.cuda.synchronize()

    diff = _max_diff(out, _expected(x_slice, weight, bias))
    assert diff < 1e-2, f"max_diff={diff}"


def test_noncontiguous_weight(native_mod):
    """weight is (D, W) but stride(1) != 1 (e.g., from transpose)."""
    B, D, L = 2, 64, 128
    W = 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight_view = torch.randn(W, D, dtype=torch.float16, device="cuda").transpose(0, 1)
    assert weight_view.shape == (D, W)
    assert not weight_view.is_contiguous()
    assert weight_view.stride(1) != 1

    bias = torch.randn(D, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)

    _call(native_mod, x, weight_view, bias, out)
    torch.cuda.synchronize()

    diff = _max_diff(out, _expected(x, weight_view, bias))
    assert diff < 1e-2, f"max_diff={diff}"
