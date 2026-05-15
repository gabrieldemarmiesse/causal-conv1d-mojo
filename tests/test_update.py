"""Tests for `causal_conv1d_mojo.causal_conv1d_update` (single-step /
KV-cache decode).

Forward-path tests for the full sequence live in `test_fwd.py`;
backward-path tests live in `test_bwd.py`. There is no backward for
`causal_conv1d_update` (it mutates the rolling state in place during
decoding — no autograd).

Each test runs on every available device + every supported dtype.
"""

import pytest
import torch

import causal_conv1d_mojo
from causal_conv1d_mojo import causal_conv1d_update_ref

from _helpers import _FWD_TOL, _make_bias, _max_diff


def test_update_single_token(device, dtype, width, activation, bias_present):
    """Decode one token at a time; out and state mutation match the
    pytorch reference."""
    B, D = 2, 16
    state_len = width - 1  # tightest state — common in Mamba decode
    x = torch.randn(B, D, dtype=dtype, device=device)  # (B, D), no seqlen
    weight = torch.randn(D, width, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)
    state = torch.randn(B, D, state_len, dtype=dtype, device=device)

    state_ours = state.clone()
    state_ref = state.clone()

    out_ours = causal_conv1d_mojo.causal_conv1d_update(
        x, state_ours, weight, bias=bias, activation=activation
    )
    out_ref = causal_conv1d_update_ref(
        x, state_ref, weight, bias=bias, activation=activation
    )

    assert _max_diff(out_ours, out_ref) < _FWD_TOL[dtype], (
        f"width={width}, out diff={_max_diff(out_ours, out_ref)}"
    )
    assert _max_diff(state_ours, state_ref) < _FWD_TOL[dtype], (
        f"state mutation differs: diff={_max_diff(state_ours, state_ref)}"
    )


def test_update_short_burst(device, dtype, width):
    """Multiple new tokens per call (seqlen > 1), state_len > width-1.
    Exercises the shift loop properly."""
    B, D, L = 1, 8, 3
    state_len = width + 4  # state holds more history than strictly needed
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    state = torch.randn(B, D, state_len, dtype=dtype, device=device)

    state_ours = state.clone()
    state_ref = state.clone()

    out_ours = causal_conv1d_mojo.causal_conv1d_update(
        x, state_ours, weight, activation="silu"
    )
    out_ref = causal_conv1d_update_ref(x, state_ref, weight, activation="silu")

    assert _max_diff(out_ours, out_ref) < _FWD_TOL[dtype]
    assert _max_diff(state_ours, state_ref) < _FWD_TOL[dtype]


def test_update_decode_sequence_matches_full_forward(device, dtype, bias_present):
    """Roll the kernel one-token-at-a-time over a full sequence and the
    concatenated outputs should match a single-shot causal_conv1d_fn over
    the same input. Both paths use fp32 internally and round identically
    per element, so fp16/bf16 are bit-for-bit identical; fp32 can show a
    single-ulp FMA-reorder delta (~5e-7) on cuda but never more."""
    B, D, L, W = 2, 16, 24, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)

    full_out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu"
    )

    # Decode loop: state starts as zeros (no history before t=0).
    state = torch.zeros(B, D, W - 1, dtype=dtype, device=device)
    decoded = []
    for t in range(L):
        out_t = causal_conv1d_mojo.causal_conv1d_update(
            x[:, :, t : t + 1], state, weight, bias=bias, activation="silu"
        )
        decoded.append(out_t)
    decoded_out = torch.cat(decoded, dim=-1)

    tol = 1e-6 if dtype == torch.float32 else 0.0
    assert _max_diff(full_out, decoded_out) <= tol


def test_update_circular_buffer_matches_ref(
    device, dtype, width, activation, bias_present
):
    """cache_seqlens != None: state is a circular buffer with the per-batch
    write head at cache_seqlens[b] mod state_len. Compare against
    upstream's ref impl which exercises the same semantics."""
    B, D, state_len = 2, 16, width + 5  # state larger than W-1 to exercise wrap
    x = torch.randn(B, D, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    bias = _make_bias(D, dtype=dtype, device=device, present=bias_present)
    state = torch.randn(B, D, state_len, dtype=dtype, device=device)
    # Mix of write heads: b=0 starts mid-buffer, b=1 starts past state_len
    # to exercise the modulo. cache_seqlens[b] >= state_len is intentional.
    cache_seqlens = torch.tensor(
        [state_len // 2, state_len + 3], dtype=torch.int32, device=device
    )

    state_ours = state.clone()
    state_ref = state.clone()

    out_ours = causal_conv1d_mojo.causal_conv1d_update(
        x,
        state_ours,
        weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )
    out_ref = causal_conv1d_update_ref(
        x,
        state_ref,
        weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )

    assert _max_diff(out_ours, out_ref) < _FWD_TOL[dtype], (
        f"out diff={_max_diff(out_ours, out_ref)}"
    )
    assert _max_diff(state_ours, state_ref) < _FWD_TOL[dtype], (
        f"state mutation differs: diff={_max_diff(state_ours, state_ref)}"
    )


def test_update_circular_decode_loop(device, dtype):
    """Repeated circular-buffer decode must match a one-shot causal_conv1d_fn
    over the same input — the cache_seqlens write head advances with each
    call and the conv reads the right history each time."""
    B, D, L, W = 1, 8, 16, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)

    full_out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, activation="silu")

    # Circular buffer with state_len exactly W-1 -- write head wraps every
    # call. Start at 0; advance by 1 each call.
    state_len = W - 1
    state = torch.zeros(B, D, state_len, dtype=dtype, device=device)
    decoded = []
    for t in range(L):
        cs = torch.tensor([t], dtype=torch.int32, device=device)
        out_t = causal_conv1d_mojo.causal_conv1d_update(
            x[:, :, t : t + 1],
            state,
            weight,
            activation="silu",
            cache_seqlens=cs,
        )
        decoded.append(out_t)
    decoded_out = torch.cat(decoded, dim=-1)

    assert _max_diff(full_out, decoded_out) < _FWD_TOL[dtype]


def test_update_conv_state_indices(device, dtype):
    """Per-batch state-row indirection: conv_state.shape[0] is a pool, and
    conv_state_indices[b] picks which slot serves batch element b."""
    pool_size = 5
    B, D, W = 3, 16, 4
    state_len = W - 1
    x = torch.randn(B, D, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)

    pool = torch.randn(pool_size, D, state_len, dtype=dtype, device=device)
    indices = torch.tensor([3, 0, 4], dtype=torch.int32, device=device)

    pool_ours = pool.clone()

    out_ours = causal_conv1d_mojo.causal_conv1d_update(
        x,
        pool_ours,
        weight,
        activation="silu",
        conv_state_indices=indices,
    )

    pool_ref = pool.clone()
    state_gathered = pool_ref[indices.to(torch.int64)]  # (B, D, state_len)
    out_ref = causal_conv1d_update_ref(x, state_gathered, weight, activation="silu")
    pool_ref[indices.to(torch.int64)] = state_gathered

    assert _max_diff(out_ours, out_ref) < _FWD_TOL[dtype]
    # Pool slots not addressed by `indices` must be untouched.
    untouched_mask = torch.ones(pool_size, dtype=torch.bool, device=device)
    untouched_mask[indices.to(torch.int64)] = False
    assert _max_diff(pool_ours[untouched_mask], pool[untouched_mask]) == 0.0, (
        "untouched pool slots were modified"
    )
    assert _max_diff(pool_ours, pool_ref) < _FWD_TOL[dtype]


def test_update_padding_token(device, dtype):
    """conv_state_indices[b] < 0 marks a padding token: output zeros, state
    untouched."""
    pool_size = 4
    B, D, W = 3, 8, 4
    state_len = W - 1
    x = torch.randn(B, D, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    pool = torch.randn(pool_size, D, state_len, dtype=dtype, device=device)
    # b=1 is a padding token; b=0 and b=2 are real (slots 0 and 2).
    indices = torch.tensor([0, -1, 2], dtype=torch.int32, device=device)

    pool_before = pool.clone()
    out = causal_conv1d_mojo.causal_conv1d_update(
        x, pool, weight, conv_state_indices=indices
    )

    # b=1 row of the output must be all zeros.
    assert torch.all(out[1] == 0), "padding token row not zeroed"
    # Pool: only slots 0 and 2 should have changed.
    untouched = (pool_before != pool).any(dim=(1, 2))
    assert untouched[0].item() and untouched[2].item()
    assert not untouched[1].item() and not untouched[3].item()


def test_update_state_too_small_raises():
    x = torch.randn(1, 8, 1, dtype=torch.float16)
    state = torch.zeros(1, 8, 2, dtype=torch.float16)  # < width - 1 = 3
    weight = torch.randn(8, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="must be >="):
        causal_conv1d_mojo.causal_conv1d_update(x, state, weight)
