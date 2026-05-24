"""Variable-length packed-batch helpers — `causal_conv1d_varlen_states`.

Mirrors upstream's
`causal_conv1d.causal_conv1d_varlen.causal_conv1d_varlen_states`.

Given a packed batch of variable-length sequences laid out as a single
`(total_tokens, dim)` tensor with cumulative sequence lengths in
`cu_seqlens`, extract the last `state_len` tokens of each sequence
into a `(batch, dim, state_len)` tensor — the conv-state input format.
Shorter-than-`state_len` sequences are zero-padded on the left.

This is pure data movement (gather + zero fill); no conv, no autograd
support, and no Mojo kernel — the bottleneck is global-memory
bandwidth, and PyTorch's strided copy is already close to peak for
this access pattern. If you have a workload where this op shows up
non-trivially in profiles, open an issue and we'll port it to Mojo.

The implementation is identical to upstream's
`causal_conv1d_varlen_states_ref` (the same package's pure-PyTorch
reference) — we just expose it under both the optimized and `_ref`
names so call sites that select between them by config keep working.
"""

from __future__ import annotations

import torch


def causal_conv1d_varlen_states(
    x: torch.Tensor, cu_seqlens: torch.Tensor, state_len: int
) -> torch.Tensor:
    """Extract the trailing `state_len` tokens of each packed sequence.

    Args:
        x: (total_tokens, dim) packed batch of token activations.
        cu_seqlens: (batch + 1,) cumulative sequence lengths, starting
            at 0. Must be sorted non-decreasing.
        state_len: number of trailing tokens per sequence to copy into
            the output state. Sequences shorter than `state_len` get
            left zero-padded.

    Returns:
        states: (batch, dim, state_len), `dtype` and `device` matching
        `x`.
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    # Allocate (batch, state_len, dim) then transpose to (batch, dim,
    # state_len) so the trailing-dim stride is 1 in the source-axis
    # direction (matching upstream's output layout).
    states = torch.zeros(
        batch, state_len, dim, dtype=x.dtype, device=x.device
    ).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        n = end_idx - start_idx
        if n > 0:
            states[i, :, -n:] = x[start_idx:end_idx].T
    return states


# `_ref` alias — identical implementation. Upstream ships both names
# (optimized Triton kernel + pure-PyTorch reference); since our impl
# is already pure PyTorch, the two names point at the same function.
causal_conv1d_varlen_states_ref = causal_conv1d_varlen_states
