"""causal_conv1d, fused into Mojo kernels and called via direct
Python <-> Mojo CPython extensions (no MAX framework).

Layout: each of the six Python entry points lives in its own
subpackage (`fwd/`, `bwd_full/`, `fwd_cpu/`, `bwd_full_cpu/`,
`update/`, `update_cpu/`). Every subpackage bundles its Mojo kernel,
Mojo dispatcher, and Python wrapper. First-time use of one of the
public APIs lazily imports — and therefore lazily compiles via
`mojo.importer` — only the subpackages it needs, instead of paying
for all six dispatch trees upfront.
"""

from __future__ import annotations

from pathlib import Path as _Path

# `mojo.importer` registers a Python import hook so that
#   from causal_conv1d_mojo.<subpkg> import dispatch
# triggers a one-time `mojo build --emit shared-lib` of the matching
# .mojo source on first import, caching the resulting .so under
# `<subpkg>/__mojocache__/`. No manual build step needed.
import mojo.importer  # noqa: F401  (registers the import hook)

# Each dispatcher's compile_function is wired with
# dump_asm=StaticString("./ptx/<name>_%.ptx"); ensure the directory
# exists in CWD before any kernel triggers a compile, since
# StaticString paths are written as-is without parent-dir creation.
_Path("./ptx").mkdir(exist_ok=True)

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
