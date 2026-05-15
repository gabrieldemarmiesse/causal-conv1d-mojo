"""Measure cold-compile time for each of the 6 Mojo subpackages.

`mojo.importer` builds each `<subpkg>/dispatch.mojo` on first import
and caches the resulting `.so` under `<subpkg>/__mojocache__/`. This
script clears those per-package caches and times the resulting
rebuilds (one mojo build per subpackage).

The compiler also keeps a *global* IR cache at
`~/.modular/.mojo_cache/` (currently ~17 GB on this box) that holds
stdlib + MAX-kernel-library IR shared across all Mojo projects on
the machine. We deliberately do NOT clear it: rebuilding the stdlib
from scratch would dominate the timing and isn't what changes when
you edit code in this repo. Pass `--clear-global` to nuke it too if
you want a true cold-start number.

Run with `pixi run -e bench python benchmarks/bench_compile_time.py`.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

SUBPKGS = ("fwd", "bwd_full", "update", "fwd_cpu", "bwd_full_cpu", "update_cpu")
ROOT = Path(__file__).resolve().parent.parent / "src" / "causal_conv1d_mojo"
GLOBAL_CACHE = Path.home() / ".modular" / ".mojo_cache"


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def clear_pkg_cache(subpkg: str) -> None:
    cache_dir = ROOT / subpkg / "__mojocache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    # Older `mojo build` invocations dropped a libdispatch.so alongside
    # the source; the importer doesn't use it but it's stale state.
    stray = ROOT / subpkg / "libdispatch.so"
    if stray.exists():
        stray.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clear-global",
        action="store_true",
        help=f"Also delete {GLOBAL_CACHE} (true cold-start; ~17 GB rebuild).",
    )
    args = ap.parse_args()

    print(f"per-package caches under: {ROOT}/<subpkg>/__mojocache__/")
    print(f"global mojo cache:        {GLOBAL_CACHE} ({_dir_size_mb(GLOBAL_CACHE):.0f} MB)")
    print()

    for sub in SUBPKGS:
        clear_pkg_cache(sub)
    if args.clear_global and GLOBAL_CACHE.exists():
        print(f"clearing global cache ({_dir_size_mb(GLOBAL_CACHE):.0f} MB)...")
        shutil.rmtree(GLOBAL_CACHE)

    # Register the importer hook BEFORE timing — the first compile
    # shouldn't pay for mojo.importer's own module load.
    import causal_conv1d_mojo  # noqa: F401

    print(f"{'subpkg':<16} {'compile (s)':>12} {'.so size (MB)':>14}")
    print("-" * 44)
    total = 0.0
    for sub in SUBPKGS:
        t0 = time.perf_counter()
        __import__(f"causal_conv1d_mojo.{sub}.dispatch")
        dt = time.perf_counter() - t0
        total += dt
        so_mb = _dir_size_mb(ROOT / sub / "__mojocache__")
        print(f"{sub:<16} {dt:>12.2f} {so_mb:>14.1f}")
    print("-" * 44)
    print(f"{'TOTAL':<16} {total:>12.2f}")


if __name__ == "__main__":
    main()
