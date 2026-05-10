"""Pad / unpad utilities for use with varlen attention.

Mirrors ``flash_attn.bert_padding`` from upstream. The varlen funcs in
this package consume a flat ``(total_seq, ...)`` tensor plus
``cu_seqlens`` start-offsets — these helpers convert to/from the
padded ``(batch, max_seqlen, ...)`` layout that most encoders produce.
"""

from __future__ import annotations

import torch


__all__ = [
    "unpad_input",
    "pad_input",
    "index_first_axis",
]


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows of ``x`` along its first axis. Equivalent to
    ``x[indices]`` but spelled out so callers can match the upstream
    name (and so we can later swap in a fused gather)."""
    return x[indices]


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Strip the padding from a (batch, seqlen, ...) tensor.

    Args:
        hidden_states: ``(batch, seqlen, ...)`` — any trailing dims OK.
        attention_mask: ``(batch, seqlen)`` int / bool — 1 means valid,
            0 means pad.

    Returns: ``(unpadded, indices, cu_seqlens, max_seqlen_in_batch)`` —
        unpadded: ``(total_valid, ...)`` flattened valid tokens.
        indices: ``(total_valid,)`` int — flat-position of each valid
            token in the original ``(batch * seqlen, ...)`` layout.
            Useful for ``pad_input`` to scatter back.
        cu_seqlens: ``(batch + 1,)`` int32 — cumulative valid lengths.
        max_seqlen_in_batch: ``int`` — max valid length across the batch.
    """
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be 2-D (batch, seqlen); got "
            f"{tuple(attention_mask.shape)}"
        )
    batch, seqlen = attention_mask.shape
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    max_seqlen_in_batch = int(seqlens_in_batch.max().item()) if batch > 0 else 0
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=hidden_states.device)
    cu_seqlens[1:] = seqlens_in_batch.cumsum(dim=0)
    # Indices of valid tokens in the flat (batch * seqlen, ...) layout.
    flat_mask = attention_mask.reshape(-1).bool()
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    flat_hidden = hidden_states.reshape(batch * seqlen, *hidden_states.shape[2:])
    unpadded = flat_hidden[indices]
    return unpadded, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Inverse of ``unpad_input``: scatter ``(total_valid, ...)`` rows
    back into ``(batch, seqlen, ...)`` at the slots given by ``indices``.

    Padding rows are filled with zeros (matches upstream)."""
    out = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])
