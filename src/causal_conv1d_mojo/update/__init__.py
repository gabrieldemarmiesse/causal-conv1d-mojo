"""GPU single-step update subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr


def native_update(
    x, weight, bias, conv_state, state_indices, cache_seqlens, out, apply_silu
):
    # 29-tuple expected by the JIT-generated variant entry point.
    # Each unique runtime config lazily compiles its own single-variant
    # `.so` on first use, then caches it under
    # `$XDG_CACHE_HOME/causal_conv1d_mojo/update/`.
    from causal_conv1d_mojo.update._jit import call_update

    call_update(
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
            torch.cuda.current_stream().cuda_stream,
            weight.shape[1],
            int(state_indices is not None),
            _ptr(state_indices),
            int(cache_seqlens is not None),
            _ptr(cache_seqlens),
        )
    )
