"""CPU single-step update subpackage: lazy per-variant JIT + Python wrapper."""

from __future__ import annotations

import torch  # noqa: F401  — needed for beartype to resolve `torch.Tensor` annotations

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo.update_cpu._jit import call_update_cpu


def native_update_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    out: torch.Tensor,
    apply_silu: bool,
) -> None:
    config = (
        _DTYPE_CODE[x.dtype],
        weight.shape[1],  # width
        bias is not None,  # has_bias
        bool(apply_silu),
        state_indices is not None,  # has_state_indices
        cache_seqlens is not None,  # is_circular
    )
    runtime_args = (
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        conv_state.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        conv_state.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        _ptr(state_indices),
        _ptr(cache_seqlens),
    )
    call_update_cpu(config, runtime_args)
