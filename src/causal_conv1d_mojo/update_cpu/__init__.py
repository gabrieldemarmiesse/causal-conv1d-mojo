"""CPU single-step update subpackage: lazy per-variant JIT + Python wrapper."""

from __future__ import annotations

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo.update_cpu._jit import call_update_cpu


def native_update_cpu(
    x, weight, bias, conv_state, state_indices, cache_seqlens, out, apply_silu
):
    call_update_cpu(
        (
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
            int(bias is not None),
            int(apply_silu),
            _DTYPE_CODE[x.dtype],
            weight.shape[1],
            int(state_indices is not None),
            _ptr(state_indices),
            int(cache_seqlens is not None),
            _ptr(cache_seqlens),
        )
    )
