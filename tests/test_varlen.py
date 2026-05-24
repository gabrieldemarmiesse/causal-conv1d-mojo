"""Tests for `causal_conv1d_varlen_states` — the packed-batch state
extractor.

Upstream's version uses a Triton kernel; ours is pure PyTorch (the op
is pure data movement, so PyTorch's strided copy is already good).
Both ship under the same name and the same `_ref` alias, so a single
correctness suite covers both.
"""

import pytest
import torch

import causal_conv1d_mojo


def _make_packed(cu_seqlens: list[int], dim: int, *, dtype, device):
    """Build a (total_tokens, dim) packed-batch tensor where each
    sequence's tokens contain easy-to-verify monotonically-increasing
    values."""
    total = cu_seqlens[-1]
    x = torch.arange(total * dim, dtype=dtype, device=device).reshape(total, dim)
    return x, torch.tensor(cu_seqlens, dtype=torch.int32, device=device)


def test_varlen_states_basic_cpu():
    """Three sequences of varying length, dim=2, state_len=3. Sequence
    lengths 5, 2, 4 → expect last 3 / all 2 (zero-padded) / last 3 of
    each respectively."""
    x, cu = _make_packed([0, 5, 7, 11], dim=2, dtype=torch.float32, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=3)
    assert out.shape == (3, 2, 3)
    assert out.dtype == torch.float32

    # Seq 0: tokens 0..4, last 3 are tokens 2,3,4 → values rows 2,3,4 of x.
    expected_0 = x[2:5].T
    assert torch.equal(out[0], expected_0)

    # Seq 1: tokens 5,6 (only 2 tokens, state_len=3 → left-zero-pad).
    assert torch.equal(out[1, :, 0], torch.zeros(2))
    assert torch.equal(out[1, :, 1:], x[5:7].T)

    # Seq 2: tokens 7..10, last 3 are 8,9,10.
    expected_2 = x[8:11].T
    assert torch.equal(out[2], expected_2)


def test_varlen_states_zero_length_sequence():
    """Empty middle sequence — state should be entirely zero-padded."""
    x, cu = _make_packed([0, 3, 3, 5], dim=4, dtype=torch.float32, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=2)
    assert out.shape == (3, 4, 2)
    assert torch.equal(out[1], torch.zeros(4, 2))


def test_varlen_states_state_longer_than_all_sequences():
    """state_len > every sequence — every output should be left-padded."""
    x, cu = _make_packed([0, 2, 4], dim=3, dtype=torch.float16, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=8)
    assert out.shape == (2, 3, 8)
    # 6 zero columns on the left, then 2 columns from x.
    assert torch.equal(out[0, :, :6], torch.zeros(3, 6, dtype=torch.float16))
    assert torch.equal(out[0, :, 6:], x[0:2].T)
    assert torch.equal(out[1, :, :6], torch.zeros(3, 6, dtype=torch.float16))
    assert torch.equal(out[1, :, 6:], x[2:4].T)


def test_varlen_states_ref_alias():
    """`_ref` is the same function — exported for upstream compat."""
    assert (
        causal_conv1d_mojo.causal_conv1d_varlen_states
        is causal_conv1d_mojo.causal_conv1d_varlen_states_ref
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_varlen_states_matches_upstream_cuda():
    """Spot-check we match upstream's Triton kernel on CUDA — same
    semantics, same output dtype + layout."""
    pytest.importorskip("causal_conv1d.causal_conv1d_varlen")
    from causal_conv1d.causal_conv1d_varlen import (
        causal_conv1d_varlen_states as upstream_fn,
    )

    torch.manual_seed(0)
    cu = torch.tensor([0, 17, 17, 32, 80], dtype=torch.int32, device="cuda")
    x = torch.randn(80, 64, dtype=torch.float16, device="cuda")
    state_len = 10

    ours = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len)
    theirs = upstream_fn(x, cu, state_len)

    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    assert torch.equal(ours, theirs)
