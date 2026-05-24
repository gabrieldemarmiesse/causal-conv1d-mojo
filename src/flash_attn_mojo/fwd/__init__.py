"""GPU forward subpackage: kernel + JIT dispatcher + Python wrapper.

STATUS: scaffold only. `native_fwd` raises `NotImplementedError`
until the actual Mojo kernel is implemented.
"""

from __future__ import annotations

import torch  # noqa: F401  — needed for beartype to resolve annotations


def native_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
) -> None:
    """JIT-compile + dispatch the GPU forward kernel.

    Placeholder until `fwd/kernel.mojo` is implemented.
    """
    raise NotImplementedError(
        "flash_attn_mojo.fwd.native_fwd: kernel not implemented yet"
    )
