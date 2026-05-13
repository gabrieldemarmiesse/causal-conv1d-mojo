"""Shared JIT-on-first-use infrastructure for the GPU subpackages.

Each subpackage (`fwd`, `bwd_full`, `update`) has its own `_jit.py`
that owns the bits that genuinely differ between them — how to extract
the comptime config tuple from the Python-side args, how to name a
variant for that config, and how to template the variant `.mojo`
source — and delegates the rest (codegen → mojo build → cache → load)
to `compile_and_load_variant` below.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable

from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str) -> Path:
    """Per-subpackage variant cache root (e.g. `~/.cache/causal_conv1d_mojo/fwd/`)."""
    return _CACHE_HOME / "causal_conv1d_mojo" / subpkg


def compile_and_load_variant(
    *,
    subpkg: str,
    source_dir: Path,
    shared_files: Iterable[str],
    mod_name: str,
    variant_source: str,
    entry_point_name: str,
):
    """Materialise, compile (if needed), load, and return the variant fn.

    Layout under `cache_dir_for(subpkg) / mod_name / `:
        <mod_name>.mojo               generated from `variant_source`
        kernel.mojo / common.mojo …   symlinks into `source_dir`
        <mod_name>.hash-<src>.so      compiled output, content-addressed

    The `.hash-<src>.so` suffix is over all `.mojo` files in the variant
    dir (resolving symlinks for content), so editing `kernel.mojo`,
    `common.mojo`, `launch.mojo`, or the generated `variant.mojo`
    template all invalidate the cached `.so` automatically.

    Args:
        subpkg: subpackage tag, e.g. ``"fwd"``. Controls the cache root.
        source_dir: directory holding the shared `kernel.mojo` /
            `common.mojo` / `launch.mojo` to symlink in.
        shared_files: filenames under `source_dir` to symlink into the
            variant dir so `mojo build` can resolve `from kernel import …`
            etc.
        mod_name: stable, human-readable name for this variant (used as
            the directory name, the Python module name passed to
            `ExtensionFileLoader`, and the `PyInit_<mod_name>` symbol
            inside the generated `.so`).
        variant_source: full text of the generated `<mod_name>.mojo`.
        entry_point_name: name of the function the generated module
            exposes (e.g. ``"causal_conv1d_fwd_variant"``).
    """
    variant_dir = cache_dir_for(subpkg) / mod_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    for fname in shared_files:
        link = variant_dir / fname
        target = source_dir / fname
        if link.is_symlink():
            try:
                if Path(os.readlink(link)) == target:
                    continue
            except OSError:
                pass
            link.unlink()
        elif link.exists():
            link.unlink()
        link.symlink_to(target)

    variant_mojo = variant_dir / f"{mod_name}.mojo"
    if not variant_mojo.exists() or variant_mojo.read_text() != variant_source:
        variant_mojo.write_text(variant_source)

    src_hash = _hash_mojo_dir(variant_dir)
    so_path = variant_dir / f"{mod_name}.hash-{src_hash}.so"

    if not so_path.is_file():
        for old in variant_dir.glob(f"{mod_name}.hash-*.so"):
            old.unlink()
        print(
            f"[causal_conv1d_mojo] JIT-compiling {subpkg} variant "
            f"{mod_name} — cached for future runs.",
            file=sys.stderr,
        )
        try:
            subprocess_run_mojo(
                [
                    "build",
                    str(variant_mojo),
                    "--emit",
                    "shared-lib",
                    "-o",
                    str(so_path),
                ],
                capture_output=True,
                check=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"JIT compilation of {subpkg} variant {mod_name} failed: {e}"
            ) from e

    loader = importlib.machinery.ExtensionFileLoader(mod_name, str(so_path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return getattr(module, entry_point_name)


def _hash_mojo_dir(d: Path) -> str:
    """Content-hash the dir's `.mojo` files for cache-invalidation."""
    hasher = hashlib.sha256()
    for f in sorted(d.glob("*.mojo")):
        hasher.update(f.name.encode())
        hasher.update(f.read_bytes())
    return hasher.hexdigest()[:16]
