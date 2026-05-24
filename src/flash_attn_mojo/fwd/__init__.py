"""GPU forward subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE


def native_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None:
    """JIT-compile (if needed) and dispatch a single GPU forward call.

    q, k, v, out: (batch, seqlen, nheads, head_dim) tensors, contiguous
        in the head_dim (last) axis. dtype must match across all four;
        currently only fp16 + head_dim=64 is supported by the kernel.
    softmax_scale: scalar applied to Q·Kᵀ before softmax (typically
        `1 / sqrt(head_dim)`).
    """
    from flash_attn_mojo.fwd._jit import call_fwd

    batch, seqlen, nheads, head_dim = q.shape

    call_fwd(
        (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[q.dtype],
            head_dim,
            1,  # use_external_stream: CUDA path wraps torch's stream
        )
    )
