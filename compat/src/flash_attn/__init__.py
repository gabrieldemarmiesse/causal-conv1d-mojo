"""Drop-in `import flash_attn` shim for `flash-attn-mojo`.

This package exists only to make `flash_attn_mojo` callable as
`flash_attn` for code originally written against the upstream Tri Dao
`flash-attn` library. The whole implementation lives in the
`flash_attn_mojo` package; this file just re-exports the public API
and refuses to load if upstream is also installed.
"""

from __future__ import annotations

from pathlib import Path as _Path

# Upstream `flash-attn` ships sibling files (`flash_attn_interface.py`
# etc.) inside its own `flash_attn/` directory. Pip lets both packages
# install into the same `site-packages/flash_attn/` directory; whichever
# was installed last wins the `__init__.py` file fight, but the other
# package's submodules are left behind, producing a Frankenstein
# namespace. If our `__init__.py` is the one Python loaded but
# upstream's `flash_attn_interface.py` is sitting next to it, fail
# loudly instead of silently picking a half-installed winner.
_pkg_dir = _Path(__file__).resolve().parent
if (_pkg_dir / "flash_attn_interface.py").is_file():
    raise ImportError(
        "Both `flash-attn` (upstream Tri Dao) and "
        "`flash-attn-mojo-compatibility` are installed. They conflict "
        "on the `flash_attn` import name. Run "
        "`pip uninstall flash-attn` to keep this shim, or uninstall "
        "this package if you want the upstream implementation."
    )

from flash_attn_mojo import (  # noqa: E402
    __version__,
    flash_attn_func,
    flash_attn_ref,
)

__all__ = [
    "__version__",
    "flash_attn_func",
    "flash_attn_ref",
]
