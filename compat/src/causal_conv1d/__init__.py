"""Drop-in `import causal_conv1d` shim for `causal-conv1d-mojo`.

This package exists only to make `causal_conv1d_mojo` callable as
`causal_conv1d` for code originally written against the upstream
Tri Dao `causal-conv1d` library. The whole implementation lives in
the `causal_conv1d_mojo` package; this file just re-exports the
public API and refuses to load if upstream is also installed (the
two packages compete for the same top-level import name, and pip
won't prevent the install).
"""

from __future__ import annotations

from pathlib import Path as _Path

# Upstream `causal-conv1d` ships sibling files (`causal_conv1d_interface.py`
# etc.) inside its own `causal_conv1d/` directory. Pip lets both packages
# install into the same `site-packages/causal_conv1d/` directory; whichever
# was installed last wins the `__init__.py` file fight, but the other
# package's submodules are left behind, producing a Frankenstein
# namespace. If our `__init__.py` is the one Python loaded but upstream's
# `causal_conv1d_interface.py` is sitting next to it, fail loudly instead
# of silently picking a half-installed winner.
_pkg_dir = _Path(__file__).resolve().parent
if (_pkg_dir / "causal_conv1d_interface.py").is_file():
    raise ImportError(
        "Both `causal-conv1d` (upstream Tri Dao) and "
        "`causal-conv1d-mojo-compatibility` are installed. They conflict "
        "on the `causal_conv1d` import name. Run "
        "`pip uninstall causal-conv1d` to keep this shim, or uninstall "
        "this package if you want the upstream implementation."
    )

from causal_conv1d_mojo import (  # noqa: E402
    __version__,
    causal_conv1d_fn,
    causal_conv1d_ref,
    causal_conv1d_update,
    causal_conv1d_update_ref,
)

__all__ = [
    "__version__",
    "causal_conv1d_fn",
    "causal_conv1d_ref",
    "causal_conv1d_update",
    "causal_conv1d_update_ref",
]
