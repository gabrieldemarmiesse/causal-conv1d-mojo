"""DLPack-based jax <-> torch bridge.

The Mojo kernels are dispatched from `_fn.py` / `_update.py` via torch
tensors (data_ptr, strides, current cuda stream). To let callers pass
`jax.Array`s through the same public API, we convert them to torch
views via DLPack at the entry point, run the existing torch path, and
convert the result(s) back. DLPack handover is zero-copy: the torch
tensor and the jax array share the same device buffer.

`is_jax_array` does *not* import jax — only checks `sys.modules` — so
this module is free to import unconditionally from the public API
modules without paying a startup cost when jax isn't installed.
"""

from __future__ import annotations

import sys


def is_jax_array(obj) -> bool:
    """Return True iff `obj` is a `jax.Array` (without forcing a jax import).

    If jax has never been imported in this process, `obj` cannot be a
    `jax.Array` — short-circuit to avoid pulling in jax just to answer
    the question for torch-only callers.
    """
    if "jax" not in sys.modules:
        return False
    import jax

    return isinstance(obj, jax.Array)


def any_jax(*args) -> bool:
    """True iff any of `args` is a `jax.Array`."""
    return any(is_jax_array(a) for a in args)


def jax_to_torch(arr):
    """Zero-copy `jax.Array` -> `torch.Tensor` via DLPack.

    The returned tensor shares storage with the jax array's underlying
    PJRT buffer. Mutations through the torch view are visible to jax
    (subject to jax's normal liveness rules — the original `arr` must
    stay alive while we operate on the view).
    """
    import torch

    return torch.utils.dlpack.from_dlpack(arr)


def torch_to_jax(t):
    """Zero-copy `torch.Tensor` -> `jax.Array` via DLPack.

    Used at the end of the bridge to convert results (`out`, `dx`,
    `dweight`, …) back to jax arrays so callers stay in the jax world.
    """
    import jax.numpy as jnp

    return jnp.from_dlpack(t)
