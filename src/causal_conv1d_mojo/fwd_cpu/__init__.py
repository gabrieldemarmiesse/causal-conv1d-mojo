"""CPU forward subpackage: lazy per-variant JIT + Python wrapper."""

from __future__ import annotations

import torch  # noqa: F401  — needed for beartype to resolve `torch.Tensor` annotations

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo.fwd_cpu._jit import call_fwd_cpu


def native_fwd_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    apply_silu: bool,
) -> None:
    config = (
        _DTYPE_CODE[x.dtype],
        weight.shape[1],  # width
        bias is not None,  # has_bias
        seq_idx is not None,  # has_seq_idx
        initial_states is not None,  # has_initial_states
        bool(apply_silu),
    )
    runtime_args = (
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
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
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
    )
    call_fwd_cpu(config, runtime_args)
