"""GPU backward subpackage: kernel + JIT dispatcher + Python wrapper.

STATUS: scaffold only. `native_bwd` raises `NotImplementedError`
until the actual Mojo kernel is implemented.
"""

from __future__ import annotations

import torch  # noqa: F401  — needed for beartype to resolve annotations


def native_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
) -> None:
    """JIT-compile + dispatch the GPU backward kernel.

    Placeholder until `bwd/kernel.mojo` is implemented.
    """
    raise NotImplementedError(
        "flash_attn_mojo.bwd.native_bwd: kernel not implemented yet"
    )
