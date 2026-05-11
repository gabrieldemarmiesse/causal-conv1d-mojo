"""CPU forward subpackage: kernel + dispatcher + Python wrapper."""

from __future__ import annotations

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr


def native_fwd_cpu(x, weight, bias, seq_idx, initial_states, out, apply_silu):
    from causal_conv1d_mojo.fwd_cpu import dispatch

    dispatch.causal_conv1d_fwd_cpu(
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
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        int(seq_idx is not None),
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        weight.shape[1],
        int(initial_states is not None),
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
    )
