"""Forward-path tests for the pure-Mojo native extension.

Covers `causal_conv1d_mojo.causal_conv1d_fn` in its no-backward
configurations (or where the test only inspects forward outputs):
contiguous + non-contiguous inputs, the width sweep, final_states,
initial_states, seq_idx packed sequences, validation errors, and
zero-sized tensors. The matching backward-path tests live in
`test_bwd.py`; the single-step update tests live in `test_update.py`.

Each test runs on every available device + every supported dtype. CPU is
always present; CUDA is exercised only if a GPU is detected.
"""

import pytest
import torch

import causal_conv1d_mojo

from _helpers import (
    _FWD_TOL,
    _expected,
    _expected_final_states,
    _expected_with_initial_states,
    _make_bias,
    _max_diff,
    _ref_with_seq_idx,
)


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
    assert x_slice.stride(1) == L * 2

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


# ===---------- return_final_states / final_states_out ----------=== #


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

    diff = _max_diff(out, _expected(x, weight, bias, "silu"))
    assert diff < _FWD_TOL[dtype], f"out max_diff={diff}"

    expected_fs = _expected_final_states(x, W)
    assert final_states.shape == (B, D, W - 1)
    assert final_states.dtype == x.dtype
    assert final_states.device == x.device
    fs_diff = _max_diff(final_states, expected_fs)
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


def test_final_states_short_seqlen(device, dtype):
    """seqlen < W-1: final_states is left zero-padded."""
    B, D, L, W = 2, 16, 2, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)

    _, fs = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=None, activation=None, return_final_states=True
    )

    pad = (W - 1) - L
    assert torch.all(fs[..., :pad] == 0)
    assert _max_diff(fs[..., pad:], x) == 0.0


# ===---------- initial_states (chunked stateful execution) ----------=== #


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

    full_out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu"
    )

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

    diff = _max_diff(full_out, chunked_out)
    assert diff < _FWD_TOL[dtype], f"chunked vs full max_diff={diff}"


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
    bad = torch.randn(2, 8, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="initial_states shape"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight, initial_states=bad)


# ===---------- seq_idx (packed sequences) ----------=== #


@pytest.mark.parametrize(
    "seq_idx_pattern",
    [
        "single",
        "two_equal",
        "varied",
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


# ===---------- widths 5..9 (newly supported beyond upstream's 2..4) ----------=== #
#
# Upstream causal-conv1d only ships widths 2..4. Since we JIT-compile
# per (dtype × width × ...) leaf on first use, supporting wider kernels
# is "free" up to the smem-halo limit kWidth - 1 ≤ kNElts:
#   * kNElts = 8 for fp16/bf16 → widths up to 9
#   * kNElts = 4 for fp32 → widths up to 5
# Backward additionally requires seqlen % 1024 == 0 for widths > 5
# because the bwd's narrow-tail path drops to kNElts=4. These tests
# cover the *forward* path at the new widths on fp16/bf16/fp32, plus a
# width=5 backward on each dtype (5 is the max that works for fp32 +
# the smallest "new" width).


@pytest.mark.parametrize("W", [5, 6, 7, 8, 9])
def test_fwd_wide_widths_fp16(device, W):
    """fp16 forward at widths 5..9 — newly supported."""
    B, D, L = 2, 64, 128
    dtype = torch.float16
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, activation="silu")
    ref = causal_conv1d_mojo.causal_conv1d_ref(x, weight, activation="silu")
    assert _max_diff(out, ref) < _FWD_TOL[dtype]


def test_fwd_width5_fp32(device):
    """fp32 forward at the new width=5 (fp32 max — width 6+ would
    require a wider halo than kNElts=4 provides)."""
    B, D, L = 2, 64, 128
    x = torch.randn(B, D, L, dtype=torch.float32, device=device)
    weight = torch.randn(D, 5, dtype=torch.float32, device=device)
    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, activation="silu")
    ref = causal_conv1d_mojo.causal_conv1d_ref(x, weight, activation="silu")
    assert _max_diff(out, ref) < _FWD_TOL[torch.float32]


def test_fwd_width6_fp32_rejected():
    """fp32 + width=6 should error cleanly (kNElts=4 can't hold the halo)."""
    x = torch.randn(2, 64, 128, dtype=torch.float32)
    weight = torch.randn(64, 6, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="width must be in 2..5"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight)


def test_fwd_width10_fp16_rejected():
    """width=10 exceeds even the fp16 limit — should error cleanly."""
    x = torch.randn(2, 64, 128, dtype=torch.float16)
    weight = torch.randn(64, 10, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="width must be in 2..9"):
        causal_conv1d_mojo.causal_conv1d_fn(x, weight)


# ===---------- torch.compile integration ----------=== #
#
# The kernel-dispatch boundary is wrapped in `torch.library.custom_op`
# so dynamo can capture the call as an atomic FX node. Without that
# wrapping, `torch.compile(fullgraph=True)` would crash trying to
# trace into the JIT-compile path (which does filesystem I/O).


def test_torch_compile_fullgraph_cuda():
    """torch.compile(fullgraph=True) must trace cleanly, no graph break."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    B, D, L = 2, 64, 128
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    w = torch.randn(D, 4, dtype=torch.float16, device="cuda")
    b = torch.randn(D, dtype=torch.float16, device="cuda")

    @torch.compile(fullgraph=True)
    def f(x, w, b):
        return causal_conv1d_mojo.causal_conv1d_fn(x, w, b, activation="silu")

    out_compiled = f(x, w, b)
    out_eager = causal_conv1d_mojo.causal_conv1d_fn(x, w, b, activation="silu")
    # Custom op call is bit-identical to the eager path (same kernel,
    # same allocations) — exact equality, not just close.
    assert torch.equal(out_compiled, out_eager)


def test_torch_compile_autograd_cuda():
    """Backward through a compiled forward must work."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    B, D, L = 2, 64, 128
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda", requires_grad=True)
    w = torch.randn(D, 4, dtype=torch.float16, device="cuda", requires_grad=True)
    b = torch.randn(D, dtype=torch.float16, device="cuda", requires_grad=True)

    @torch.compile()
    def f(x, w, b):
        return causal_conv1d_mojo.causal_conv1d_fn(x, w, b, activation="silu").sum()

    loss = f(x, w, b)
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert b.grad is not None and b.grad.shape == b.shape
