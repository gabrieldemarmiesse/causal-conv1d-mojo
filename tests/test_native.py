"""Tests for the pure-Mojo native path (no MAX framework).

The native extension supports fp16 / bf16 / fp32, width in {2, 3, 4},
optional bias, optional silu/swish activation, optional seq_idx (fwd
only), optional initial_states (fwd only), and return_final_states.

Each test runs on every available device + every supported dtype. CPU is
always present; CUDA is exercised only if a GPU is detected.
"""

import pytest
import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


# Devices to run every test against. CPU is always available; CUDA is
# parametrised in but skipped per-test if the box has no GPU. fp16/bf16
# on CPU are supported on PyTorch 2.x — the native CPU kernel computes
# everything in fp32 internally and casts back at the boundary.
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


# Dtypes supported by both the GPU and CPU paths. bf16 has only 7
# mantissa bits (vs fp16's 10), so reduction error on the backward pass
# is the loosest of the three; fp32 is the tightest.
@pytest.fixture(
    params=[torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def dtype(request):
    return request.param


# Tolerances are dominated by accumulator precision in the inner conv
# (forward) and by the size of the (B, L) reduction (dweight, dbias).
# bf16 has half the mantissa of fp16 — both forward and the per-element
# backward (dx) get noticeably looser; the reduction tolerances scale
# with B*L regardless of dtype but are roughly the same across fp16/bf16
# in practice (the fp32 accumulators absorb most of it).
_FWD_TOL = {torch.float16: 2e-2, torch.bfloat16: 2e-1, torch.float32: 1e-4}
_DX_TOL = {torch.float16: 1e-1, torch.bfloat16: 5e-1, torch.float32: 1e-3}
# dweight/dbias are sums over B*L terms — same fp32 accumulator on all
# paths, so the only delta is the cast back to the input dtype at the
# boundary. fp32 keeps the full accumulator so the tolerance collapses.
_DW_TOL = {torch.float16: 1.0, torch.bfloat16: 2.0, torch.float32: 1e-2}


def _make_bias(D, *, dtype, device, present, requires_grad=False):
    if not present:
        return None
    return torch.randn(D, dtype=dtype, device=device, requires_grad=requires_grad)


def _expected(x, weight, bias, activation):
    return causal_conv1d_ref(x, weight, bias=bias, activation=activation)


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_contiguous(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x, weight, bias, activation))
    assert diff < _FWD_TOL[dtype], f"max_diff={diff}"


def test_noncontiguous_x_seq_stride_not_one(device, dtype, activation, bias_present):
    """x is (B, D, L) but came from a transpose so stride(2) != 1."""
    B, D, L = 2, 64, 128
    W = 4
    # Allocate as (B, L, D) then transpose to expose (B, D, L) with non-unit
    # innermost stride. After transpose: stride(2) == D, stride(1) == 1.
    x_view = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    assert x_view.shape == (B, D, L)
    assert not x_view.is_contiguous()
    assert x_view.stride(2) != 1

    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_view, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x_view, weight, bias, activation))
    assert diff < _FWD_TOL[dtype], f"max_diff={diff}"


def test_noncontiguous_x_sliced(device, dtype, activation, bias_present):
    """x is a slice of a larger tensor (contiguous stride=1 on last dim, but
    leading strides are larger than the slice's shape would imply if it
    were contiguous)."""
    B, D, L = 2, 64, 128
    W = 4
    big_x = torch.randn(B, D, L * 2, dtype=dtype, device=device)
    x_slice = big_x[:, :, :L]
    assert x_slice.shape == (B, D, L)
    assert not x_slice.is_contiguous()
    assert x_slice.stride(2) == 1
    assert x_slice.stride(1) == L * 2  # gap between rows

    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x_slice, weight, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x_slice, weight, bias, activation))
    assert diff < _FWD_TOL[dtype], f"max_diff={diff}"


def test_noncontiguous_weight(device, dtype, activation, bias_present):
    """weight is (D, W) but stride(1) != 1 (e.g., from transpose)."""
    B, D, L = 2, 64, 128
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight_view = torch.randn(W, D, dtype=dtype, device=device).transpose(0, 1)
    assert weight_view.shape == (D, W)
    assert not weight_view.is_contiguous()
    assert weight_view.stride(1) != 1

    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight_view, bias=bias, activation=activation
    )

    diff = _max_diff(out, _expected(x, weight_view, bias, activation))
    assert diff < _FWD_TOL[dtype], f"max_diff={diff}"


# ===---------- width sweep (2 / 3 / 4) ----------=== #
# The bulk of the suite exercises width=4 (Mamba's setting). These
# tests exist to guard the cheaper widths. Forward, backward, and the
# seq_idx path are each hit at all three widths.


@pytest.fixture(params=[2, 3, 4], ids=["w2", "w3", "w4"])
def width(request):
    return request.param


def test_width_forward(device, dtype, width, activation, bias_present):
    B, D, L = 2, 32, 128
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    diff = _max_diff(out, _expected(x, weight, bias, activation))
    assert diff < _FWD_TOL[dtype], f"width={width}, max_diff={diff}"


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


def test_width_seq_idx_forward(device, dtype, width):
    """Packed-sequence forward at every width — guards that the seq_idx
    mask honours the W-1 lookback correctly for width != 4."""
    B, D, L = 1, 16, 64
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    seq_idx = torch.cat(
        [
            torch.zeros(B, L // 2, dtype=torch.int32, device=device),
            torch.ones(B, L - L // 2, dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, seq_idx=seq_idx, activation=None
    )
    expected = _ref_with_seq_idx(x, weight, None, seq_idx, None)
    assert _max_diff(out, expected) < _FWD_TOL[dtype]


def test_width_invalid_raises():
    """Widths outside {2, 3, 4} are rejected by the validator before
    touching the kernel."""
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 5, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="width in"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight)


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


# ===---------- return_final_states / final_states_out ----------=== #


def _expected_final_states(x, width):
    """Reference: last `width-1` columns of x with left zero-pad if
    seqlen < width-1. Same as `F.pad(x, (W-1-L, 0))[..., -W+1:]` from
    upstream's `causal_conv1d_ref`.
    """
    seqlen = x.shape[-1]
    pad_left = max(0, (width - 1) - seqlen)
    if pad_left > 0:
        return F.pad(x, (pad_left, 0))
    return x[..., -(width - 1) :].contiguous()


@pytest.mark.parametrize("seqlen", [2, 3, 4, 16, 128])
def test_return_final_states_forward(device, dtype, seqlen, bias_present):
    """final_states matches `F.pad(x, (W-1-L, 0))[-W+1:]`.

    Covers the seqlen < W-1 (pad-left), seqlen == W-1, and the common
    seqlen >> W-1 cases. activation is fixed at silu since it doesn't
    affect final_states.
    """
    B, D, W = 2, 16, 4
    x = torch.randn(B, D, seqlen, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out, final_states = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu", return_final_states=True
    )

    # out should still match the no-final-states reference.
    diff = _max_diff(out, _expected(x, weight, bias, "silu"))
    assert diff < _FWD_TOL[dtype], f"out max_diff={diff}"

    expected_fs = _expected_final_states(x, W)
    assert final_states.shape == (B, D, W - 1)
    assert final_states.dtype == x.dtype
    assert final_states.device == x.device
    fs_diff = _max_diff(final_states, expected_fs)
    # final_states is just a copy/pad — should be exact, not just close.
    assert fs_diff == 0.0, f"final_states max_diff={fs_diff}"


def test_final_states_out_user_provided(device, dtype):
    """User-allocated `final_states_out` is filled in-place (so the user's
    reference reflects the kernel's output)."""
    B, D, L, W = 2, 16, 32, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)

    fs_out = torch.empty(B, D, W - 1, dtype=dtype, device=device)
    fs_storage = fs_out.data_ptr()

    out, fs_returned = causal_conv1d_mojo.causal_conv1d_fn(
        x,
        weight,
        bias=None,
        activation=None,
        return_final_states=True,
        final_states_out=fs_out,
    )

    # Returned and user-provided tensors share storage (autograd may
    # wrap the returned tensor in a new Python object, but the data
    # buffer is the same one we passed in).
    assert fs_returned.data_ptr() == fs_storage
    assert _max_diff(fs_out, _expected_final_states(x, W)) == 0.0


def test_final_states_out_validation():
    """Passing `final_states_out` without `return_final_states=True` is
    a user error."""
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float16)
    fs = torch.empty(1, 8, 3, dtype=torch.float16)

    with pytest.raises(ValueError, match="return_final_states=True"):
        causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, return_final_states=False, final_states_out=fs
        )


def test_final_states_backward(device, dtype, bias_present):
    """Gradient w.r.t. final_states is added to the matching tail of dx.

    final_states[b, c, i] = x[b, c, seqlen - (W-1) + i] for the
    seqlen >= W-1 case, so dfinal_states[b, c, i] flows directly back
    into dx[b, c, seqlen - (W-1) + i] in addition to the conv-path dx
    contribution.
    """
    B, D, L, W = 2, 16, 64, 4
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
    # Sum both grads simultaneously.
    torch.autograd.backward([out, fs], [dout, dfs])

    # Reference: combine standard backward + the dfinal_states tail.
    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    # final_states is just `x[:, :, -(W-1):]` (seqlen >= W-1 here), so
    # its gradient lands in the matching slice of dx_ref.
    dx_ref = dx_ref.clone()
    dx_ref[..., -(W - 1) :] += dfs.to(dx_ref.dtype)

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    )
    assert _max_diff(weight.grad, dw_ref) < _DW_TOL[dtype]
    if bias_present:
        assert _max_diff(bias.grad, db_ref) < _DW_TOL[dtype]


def test_final_states_short_seqlen(device, dtype):
    """seqlen < W-1: final_states is left zero-padded."""
    B, D, L, W = 2, 16, 2, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)

    _, fs = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=None, activation=None, return_final_states=True
    )

    # First (W-1-L) cols should be zero, last L cols should equal x.
    pad = (W - 1) - L
    assert torch.all(fs[..., :pad] == 0)
    assert _max_diff(fs[..., pad:], x) == 0.0


# ===---------- initial_states (chunked stateful execution) ----------=== #


def _expected_with_initial_states(x, weight, bias, initial_states, activation):
    """Reference: pre-pend initial_states to x along seqlen, run conv with
    padding=0, slice back. Mirrors upstream's `causal_conv1d_ref` branch.
    """
    seqlen = x.shape[-1]
    D, W = weight.shape
    x_full = torch.cat([initial_states, x], dim=-1)
    pre = F.conv1d(x_full, weight.unsqueeze(1), bias, padding=0, groups=D)[..., :seqlen]
    return F.silu(pre) if activation in ("silu", "swish") else pre


def test_initial_states_forward(device, dtype, width, activation, bias_present):
    """initial_states matches the cat([init, x]) reference."""
    B, D, L = 2, 16, 64
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)
    initial_states = torch.randn(B, D, width - 1, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        activation=activation,
    )
    expected = _expected_with_initial_states(
        x, weight, bias, initial_states, activation
    )
    diff = _max_diff(out, expected)
    assert diff < _FWD_TOL[dtype], f"width={width}, max_diff={diff}"


def test_initial_states_chunked_roundtrip(device, dtype, bias_present):
    """Threading `final_states_out[i]` → `initial_states[i+1]` reproduces a
    full-sequence forward. This is the canonical use case for both APIs.
    """
    B, D, L, W = 1, 16, 96, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    # Reference: one shot.
    full_out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu"
    )

    # Chunked: split x along seqlen, run kernel per chunk, thread state.
    chunks = [x[..., :32], x[..., 32:64], x[..., 64:]]
    init = None
    chunked_outs = []
    for c in chunks:
        out_c, init = causal_conv1d_mojo.causal_conv1d_fn(
            c,
            weight,
            bias=bias,
            initial_states=init,
            return_final_states=True,
            activation="silu",
        )
        chunked_outs.append(out_c)
    chunked_out = torch.cat(chunked_outs, dim=-1)

    # Should be bit-identical to the one-shot call (no roundoff diff;
    # both paths sum the same fp32 accumulators).
    diff = _max_diff(full_out, chunked_out)
    assert diff < _FWD_TOL[dtype], f"chunked vs full max_diff={diff}"


def test_initial_states_backward_raises(device):
    """Backward through initial_states is not implemented yet."""
    B, D, L, W = 1, 8, 32, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device=device, requires_grad=True)
    init = torch.randn(B, D, W - 1, dtype=torch.float16, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, initial_states=init, activation="silu"
    )
    with pytest.raises(NotImplementedError, match="backward through initial_states"):
        out.sum().backward()


def test_initial_states_mutual_exclusion_with_seq_idx():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float16)
    seq_idx = torch.zeros(1, 16, dtype=torch.int32)
    init = torch.randn(1, 8, 3, dtype=torch.float16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, seq_idx=seq_idx, initial_states=init
        )


def test_initial_states_shape_validation():
    x = torch.randn(2, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float16)
    bad = torch.randn(2, 8, 4, dtype=torch.float16)  # should be (2, 8, 3)
    with pytest.raises(ValueError, match="initial_states shape"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight, initial_states=bad)


# ===---------- seq_idx (packed sequences) ----------=== #


def _ref_with_seq_idx(x, weight, bias, seq_idx, activation):
    """Reference for packed-sequence forward: each contiguous run of
    equal seq_idx values is treated as an independent sequence (the
    conv shouldn't read across boundaries). Padding (seq_idx < 0)
    rows output 0.
    """
    B, D, L = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        ids = seq_idx[b].cpu().numpy()
        # Walk runs of equal id; for each run [start, end), do an
        # independent causal conv over that slice.
        start = 0
        while start < L:
            end = start + 1
            while end < L and ids[end] == ids[start]:
                end += 1
            run_id = int(ids[start])
            if run_id < 0:
                # padding — leave out as zero
                start = end
                continue
            seg = x[b : b + 1, :, start:end]
            seg_out = causal_conv1d_ref(seg, weight, bias=bias, activation=activation)
            out[b : b + 1, :, start:end] = seg_out
            start = end
    return out


@pytest.mark.parametrize(
    "seq_idx_pattern",
    [
        # one sequence covering the whole row (sanity: matches no-seq-idx case).
        "single",
        # two equal-length packed sequences.
        "two_equal",
        # three sequences of varied lengths.
        "varied",
        # a padding region in the middle (negative ids → output 0).
        "with_padding",
    ],
)
def test_seq_idx_forward(device, dtype, seq_idx_pattern, activation, bias_present):
    B, D, L, W = 2, 16, 64, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

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
        # 0..15 -> seq 0, 16..31 -> padding (-1), 32..63 -> seq 1
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

    expected = _ref_with_seq_idx(x, weight, bias, seq_idx, activation)
    diff = _max_diff(out, expected)
    assert diff < _FWD_TOL[dtype], f"max_diff={diff}, pattern={seq_idx_pattern}"


def test_seq_idx_backward_raises(device):
    """Backward through seq_idx is not implemented yet; the autograd
    Function raises NotImplementedError if anyone tries to backprop."""
    B, D, L, W = 1, 8, 32, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=torch.float16, device=device, requires_grad=True)
    seq_idx = torch.zeros(B, L, dtype=torch.int32, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, seq_idx=seq_idx, activation="silu"
    )
    with pytest.raises(NotImplementedError, match="backward through seq_idx"):
        out.sum().backward()


def test_seq_idx_inference_no_grad(device, dtype):
    """Inference under no_grad with requires_grad=True inputs must
    not invoke backward, so seq_idx works fine."""
    B, D, L, W = 1, 16, 32, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    seq_idx = torch.zeros(B, L, dtype=torch.int32, device=device)

    with torch.no_grad():
        out = causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, seq_idx=seq_idx, activation=None
        )
    expected = _ref_with_seq_idx(x, weight, None, seq_idx, None)
    assert _max_diff(out, expected) < _FWD_TOL[dtype]


def test_seq_idx_mutual_exclusion_with_final_states():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float16)
    seq_idx = torch.zeros(1, 16, dtype=torch.int32)
    with pytest.raises(ValueError, match="mutually exclusive"):
        causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, seq_idx=seq_idx, return_final_states=True
        )


def test_seq_idx_dtype_validation():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float16)
    bad = torch.zeros(1, 16, dtype=torch.int64)
    with pytest.raises(ValueError, match="seq_idx.dtype must be int32"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight, seq_idx=bad)


# ===---------- zero-sized tensors ----------=== #
# Each (B, D, L) configuration with at least one zero dimension. A
# well-behaved op should accept these and return correctly-shaped
# (empty) outputs / gradients without raising. On the GPU side
# `enqueue_function` rejects any `grid_dim == 0`, so the launchers
# need an explicit early-out.


@pytest.mark.parametrize("shape", [(0, 64, 128), (2, 0, 128), (2, 64, 0), (0, 0, 0)])
def test_zero_sized_forward(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.numel() == 0


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
