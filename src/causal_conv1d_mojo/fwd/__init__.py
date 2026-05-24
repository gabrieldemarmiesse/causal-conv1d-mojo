"""GPU forward subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo._mps import gpu_address, gpu_address_or_zero


def native_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    apply_silu: bool,
) -> None:
    # The 29-tuple expected by the JIT-generated variant entry point.
    # Each unique runtime config (dtype × width × has_bias × has_seq_idx ×
    # has_initial_states × apply_silu × contig_inner × aligned_seq) lazily
    # compiles its own single-variant `.so` on first use, then caches it
    # under `$XDG_CACHE_HOME/causal_conv1d_mojo/fwd/` for future processes.
    from causal_conv1d_mojo.fwd._jit import call_fwd

    call_fwd(
        (
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
            torch.cuda.current_stream().cuda_stream,
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
            1,  # use_external_stream: CUDA path wraps torch's stream
        )
    )


def native_fwd_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    apply_silu: bool,
) -> None:
    """Mac/MPS path — identical to `native_fwd` except we (a) flush
    torch's MPS command queue before launch so any pending torch
    writes to these tensors land before our kernel reads them, (b)
    pass each tensor's Metal `gpuAddress` instead of its CUDA-style
    `data_ptr()` (torch's MPS data_ptr is the `id<MTLBuffer>` Obj-C
    pointer, not a GPU VA — Mojo's kernel can't dereference that;
    see ``_mps.py`` for the conversion), and (c) signal "no external
    stream" with ``stream_handle_addr=0`` so the Mojo launcher
    enqueues on the `DeviceContext`'s default stream (Metal has no
    CUDA-style streams). The kernel itself is the same JIT'd
    variant used by the CUDA path.
    """
    from causal_conv1d_mojo.fwd._jit import call_fwd

    torch.mps.synchronize()
    call_fwd(
        (
            gpu_address(x),
            gpu_address(weight),
            gpu_address_or_zero(bias),
            gpu_address(out),
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
            0,  # stream_handle_addr — Metal has no streams; enqueue on ctx
            int(seq_idx is not None),
            gpu_address_or_zero(seq_idx),
            seq_idx.stride(0) if seq_idx is not None else 0,
            seq_idx.stride(1) if seq_idx is not None else 0,
            weight.shape[1],
            int(initial_states is not None),
            gpu_address_or_zero(initial_states),
            initial_states.stride(0) if initial_states is not None else 0,
            initial_states.stride(1) if initial_states is not None else 0,
            initial_states.stride(2) if initial_states is not None else 0,
            0,  # use_external_stream: Metal path enqueues on ctx
        )
    )
