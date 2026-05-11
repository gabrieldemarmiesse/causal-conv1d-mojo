"""GPU fused backward subpackage: kernel + dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr


def native_bwd_full(
    x,
    weight,
    bias,
    dout,
    seq_idx,
    initial_states,
    dx,
    dweight_acc,
    dbias_acc,
    dinitial_states,
    apply_silu,
):
    from causal_conv1d_mojo.bwd_full import dispatch

    dispatch.causal_conv1d_bwd_full(
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
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        torch.cuda.current_stream().cuda_stream,
        weight.shape[1],
        int(seq_idx is not None),
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        int(initial_states is not None),
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
        _ptr(dinitial_states),
        dinitial_states.stride(0) if dinitial_states is not None else 0,
        dinitial_states.stride(1) if dinitial_states is not None else 0,
        dinitial_states.stride(2) if dinitial_states is not None else 0,
    )
