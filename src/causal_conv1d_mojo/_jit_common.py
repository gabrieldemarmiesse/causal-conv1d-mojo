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
import time
from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str) -> Path:
    """Per-subpackage variant cache root (e.g. `~/.cache/causal_conv1d_mojo/fwd/`)."""
    return _CACHE_HOME / "causal_conv1d_mojo" / subpkg


def compile_and_load_variant(
    *,
    subpkg: str,
    include_dirs: Iterable[Path],
    mod_name: str,
    variant_source: str,
    entry_point_name: str,
):
    """Materialise, compile (if needed), load, and return the variant fn.

    Layout under `cache_dir_for(subpkg) / mod_name / `:
        <mod_name>.mojo               generated from `variant_source`
        <mod_name>.hash-<src>.so      compiled output, content-addressed

    `mojo build` resolves `from kernel import …` etc. via the ``-I``
    flag — one per entry in ``include_dirs`` — so the shared source
    files stay in the package tree and we don't have to symlink them
    into each variant dir.

    The `.hash-<src>.so` suffix is over the generated `variant.mojo`
    plus every `.mojo` file in each include dir, so editing any of
    them invalidates the cached `.so` automatically.

    Args:
        subpkg: subpackage tag, e.g. ``"fwd"``. Controls the cache root.
        include_dirs: directories passed to ``mojo build -I``. The
            first entry should be the subpackage source dir (holding
            ``kernel.mojo`` / ``common.mojo`` / ``launch.mojo``); the
            package root can be added to pick up shared files like
            ``_ctx.mojo``.
        mod_name: stable, human-readable name for this variant (used as
            the directory name, the Python module name passed to
            `ExtensionFileLoader`, and the `PyInit_<mod_name>` symbol
            inside the generated `.so`).
        variant_source: full text of the generated `<mod_name>.mojo`.
        entry_point_name: name of the function the generated module
            exposes (e.g. ``"causal_conv1d_fwd_variant"``).
    """
    include_dirs = [Path(d) for d in include_dirs]
    variant_dir = cache_dir_for(subpkg) / mod_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_mojo = variant_dir / f"{mod_name}.mojo"
    if not variant_mojo.exists() or variant_mojo.read_text() != variant_source:
        variant_mojo.write_text(variant_source)

    src_hash = _hash_sources(variant_mojo, include_dirs)
    so_path = variant_dir / f"{mod_name}.hash-{src_hash}.so"

    if not so_path.is_file():
        for old in variant_dir.glob(f"{mod_name}.hash-*.so"):
            old.unlink()
        print(
            f"[causal_conv1d_mojo] JIT-compiling {subpkg} variant "
            f"{mod_name} — cached for future runs.",
            file=sys.stderr,
            end="",
        )
        cmd = ["build", str(variant_mojo)]
        for d in include_dirs:
            cmd += ["-I", str(d)]
        cmd += ["--emit", "shared-lib", "-o", str(so_path)]
        try:
            t1 = time.perf_counter()
            subprocess_run_mojo(cmd, capture_output=True, check=True)
            t2 = time.perf_counter()
            print(f" done in {(t2 - t1) * 1000:8.1f} ms", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(
                f"JIT compilation of {subpkg} variant {mod_name} failed: {e}"
            ) from e

    loader = importlib.machinery.ExtensionFileLoader(mod_name, str(so_path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # Stash the module in sys.modules so callers that need extra entry
    # points (e.g. a one-time setup helper alongside the per-call kernel
    # launcher) can `importlib.import_module(mod_name)` to reach them.
    sys.modules[mod_name] = module
    return getattr(module, entry_point_name)


def _hash_sources(variant_mojo: Path, include_dirs: Iterable[Path]) -> str:
    """Content-hash the variant + every `.mojo` in each include dir."""
    hasher = hashlib.sha256()
    hasher.update(variant_mojo.name.encode())
    hasher.update(variant_mojo.read_bytes())
    for d in include_dirs:
        for f in sorted(Path(d).glob("*.mojo")):
            hasher.update(str(f).encode())
            hasher.update(f.read_bytes())
    return hasher.hexdigest()[:16]
