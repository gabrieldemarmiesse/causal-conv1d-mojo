"""GPU forward subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr


def native_fwd(x, weight, bias, seq_idx, initial_states, out, apply_silu):
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
        )
    )
