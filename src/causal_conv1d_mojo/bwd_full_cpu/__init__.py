"""CPU fused backward subpackage: lazy per-variant JIT + Python wrapper."""

from __future__ import annotations

import torch  # noqa: F401  — needed for beartype to resolve `torch.Tensor` annotations

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo.bwd_full_cpu._jit import call_bwd_full_cpu


def native_bwd_full_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight_acc: torch.Tensor,
    dbias_acc: torch.Tensor | None,
    dinitial_states: torch.Tensor | None,
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
        dout.data_ptr(),
        dx.data_ptr(),
        dweight_acc.data_ptr(),
        _ptr(dbias_acc),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
        _ptr(dinitial_states),
        dinitial_states.stride(0) if dinitial_states is not None else 0,
        dinitial_states.stride(1) if dinitial_states is not None else 0,
        dinitial_states.stride(2) if dinitial_states is not None else 0,
    )
    call_bwd_full_cpu(config, runtime_args)
