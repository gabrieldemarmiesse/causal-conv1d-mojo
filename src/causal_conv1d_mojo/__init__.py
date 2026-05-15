"""causal_conv1d, fused into Mojo kernels and called via direct
Python <-> Mojo CPython extensions (no MAX framework).

Layout: each of the six Python entry points lives in its own
subpackage (`fwd/`, `bwd_full/`, `fwd_cpu/`, `bwd_full_cpu/`,
`update/`, `update_cpu/`).

GPU subpackages (`fwd`, `bwd_full`, `update`) use JIT-on-first-use:
each runtime config compiles its own single-variant `.so` via
`mojo build` at call time and caches the result under
`$XDG_CACHE_HOME/causal_conv1d_mojo/<subpkg>/`. See
`<subpkg>/_jit.py`.

CPU subpackages still use the original AOT model: `from
causal_conv1d_mojo.<subpkg>_cpu import dispatch` triggers a one-time
`mojo build` of the matching `dispatch.mojo` on first import, via
`mojo.importer`'s sys.meta_path hook.
"""

from __future__ import annotations

# Registers the import hook used by the CPU subpackages'
# `from <subpkg>_cpu import dispatch` lazy import.
import mojo.importer  # noqa: F401

from causal_conv1d_mojo._fn import causal_conv1d_fn
from causal_conv1d_mojo._update import causal_conv1d_update
from causal_conv1d_mojo.reference import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)


__version__ = "1.6.1"

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "causal_conv1d_ref",
    "causal_conv1d_update_ref",
]
