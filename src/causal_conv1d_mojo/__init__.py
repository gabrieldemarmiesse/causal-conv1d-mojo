"""causal_conv1d, fused into a single Mojo GPU kernel and called via a
direct Python <-> Mojo CPython extension (no MAX framework).

The extension is built from `_native/causal_conv1d_native.mojo`. Run
`pixi run build-native` once after edits.

Current specialization grid (single combo):
    dtype = fp16
    width = 4
    has_bias = True
    activation = "silu"
    initial_states = None
    return_final_states = False

Anything outside that raises NotImplementedError.
"""
from __future__ import annotations

import torch

# `mojo.importer` registers a Python import hook so that
#   from causal_conv1d_mojo._native import causal_conv1d_native
# triggers a one-time `mojo build --emit shared-lib` of the matching
# .mojo source on first import, caching the resulting .so under
# __mojocache__/. No manual `pixi run build-native` needed.
import mojo.importer  # noqa: F401  (registers the import hook)

from causal_conv1d_mojo._native import causal_conv1d_native as _native_mod


__version__ = "1.6.1"


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if seq_idx is not None:
        raise NotImplementedError("seq_idx is not supported")
    if initial_states is not None:
        raise NotImplementedError("initial_states is not supported")
    if return_final_states or final_states_out is not None:
        raise NotImplementedError("return_final_states is not supported")
    if activation not in ("silu", "swish"):
        raise NotImplementedError(
            "only activation in {'silu', 'swish'} is supported"
        )
    if bias is None:
        raise NotImplementedError("bias is required")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise NotImplementedError("only fp16 is supported")
    if not x.is_cuda:
        raise NotImplementedError("only CUDA tensors are supported")
    if weight.shape[1] != 4:
        raise NotImplementedError(f"only width=4 is supported (got {weight.shape[1]})")

    out = torch.empty_like(x)
    _native_mod.causal_conv1d_fwd_fp16_w4_silu_bias(
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        torch.cuda.current_stream().cuda_stream,
    )
    return out
