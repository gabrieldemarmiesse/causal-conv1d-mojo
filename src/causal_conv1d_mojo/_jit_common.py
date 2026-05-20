"""Shared JIT-on-first-use infrastructure for the GPU + CPU subpackages.

Each subpackage has its own `_jit.py` that owns the bits that genuinely
differ between them — how to extract the comptime config tuple from
the Python-side args and how to name a variant for that config — and
delegates the rest (mojo build → cache → load) to `compile_and_load`
below.

The Mojo side of each variant is a *single static* `variant.mojo` file
that reads its comptime parameters via `std.sys.get_defined_*`. The
Python wrapper materialises those parameters as `-D KEY=VALUE` args to
`mojo build`, so the generated `.so` is keyed by the full config — no
f-string codegen, no per-variant `.mojo` scribble on disk.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping
import time
from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str) -> Path:
    """Per-subpackage variant cache root (e.g. `~/.cache/causal_conv1d_mojo/fwd/`)."""
    return _CACHE_HOME / "causal_conv1d_mojo" / subpkg


def compile_and_load(
    *,
    subpkg: str,
    source_file: Path,
    include_dirs: Iterable[Path] = (),
    defines: Mapping[str, str] = {},
    mod_name: str,
):
    """Compile a static `variant.mojo` with `-D` + `-I` and return the loaded module.

    `mojo build` resolves `from kernel import …` etc. via the ``-I``
    flag — one per entry in ``include_dirs`` — and gets the variant's
    comptime parameters via ``-D KEY=VALUE`` (read inside the `.mojo`
    file with `std.sys.get_defined_*`).

    The compiled `.so` is cached at
    ``cache_dir_for(subpkg) / <mod_name>.hash-<h>.so`` and is
    content-addressed over the source file, every `.mojo` in each
    include dir, and the `defines` mapping — so editing any of those
    invalidates the cached `.so` automatically.

    The Mojo source must export ``PyInit_variant`` (every variant
    shares the same Python module name internally; we disambiguate
    on the Python side via the unique ``mod_name`` cache key).

    Args:
        subpkg: subpackage tag, e.g. ``"fwd"``. Controls the cache root.
        source_file: path to the static `variant.mojo` to compile.
        include_dirs: directories passed to ``mojo build -I``. The
            source file's directory doesn't need to be repeated.
        defines: comptime parameters; each ``(k, v)`` becomes
            ``-D KEY=VALUE`` and is read inside the source via
            ``std.sys.get_defined_*``.
        mod_name: stable, human-readable identifier for this variant
            (used as the cache key and the Python-side module name).

    Returns:
        The loaded Python extension module. Get entry-point functions
        via ``getattr(module, name)``.
    """
    include_dirs = [Path(d) for d in include_dirs]
    cache_dir = cache_dir_for(subpkg)
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_hash = _hash_sources(source_file, include_dirs, defines)
    so_path = cache_dir / f"{mod_name}.hash-{src_hash}.so"

    if not so_path.is_file():
        for old in cache_dir.glob(f"{mod_name}.hash-*.so"):
            old.unlink()
        print(
            f"[causal_conv1d_mojo] compiling {subpkg} variant {mod_name} — "
            f"cached for future runs.",
            file=sys.stderr,
            end="",
        )
        cmd = ["build", str(source_file)]
        for d in include_dirs:
            cmd += ["-I", str(d)]
        for k, v in defines.items():
            cmd += ["-D", f"{k}={v}"]
        cmd += ["--emit", "shared-lib", "-o", str(so_path)]
        try:
            t1 = time.perf_counter()
            subprocess_run_mojo(cmd, capture_output=True, check=True)
            t2 = time.perf_counter()
            print(f" done in {(t2 - t1) * 1000:8.1f} ms", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(
                f"Compilation of {subpkg} variant {mod_name} failed: {e}"
            ) from e

    # The .so exports `PyInit_variant` (every variant.mojo uses the same
    # symbol), so we always load with module name "variant"; the unique
    # `mod_name` is just a Python-side disambiguator for sys.modules.
    loader = importlib.machinery.ExtensionFileLoader("variant", str(so_path))
    spec = importlib.util.spec_from_loader("variant", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[mod_name] = module
    return module


def _hash_sources(
    source_file: Path,
    include_dirs: Iterable[Path],
    defines: Mapping[str, str],
) -> str:
    """Content-hash the source file + every `.mojo` in each include dir + defines."""
    hasher = hashlib.sha256()
    hasher.update(source_file.name.encode())
    hasher.update(source_file.read_bytes())
    for d in include_dirs:
        for f in sorted(Path(d).glob("*.mojo")):
            hasher.update(str(f).encode())
            hasher.update(f.read_bytes())
    for k in sorted(defines):
        hasher.update(f"{k}={defines[k]}\n".encode())
    return hasher.hexdigest()[:16]
