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

Cache layout (post-PR):

    $XDG_CACHE_HOME/causal_conv1d_mojo/<subpkg>/<backend>[/<arch>]/<mod>.hash-<h>.so

`<backend>` is one of {cpu, cuda, rocm, metal}. GPU backends carry an
extra `<arch>` segment (e.g. `sm89`, `gfx942`, `macos15`) because the
compiled artefact is locked to that target — cubin is per-sm, HSACO is
per-gfx. `<h>` mixes the file contents *and* an env signature (Python
ABI, mojo version, modular SDK path, ptxas version on CUDA, this
file's own hash) so any env shift that could change codegen / ABI
busts the cache instead of silently loading a stale `.so`. The full
list of signals is in `_env_signature` below.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import importlib.util
import os
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable, Mapping

from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str, backend: str, backend_arch: str = "") -> Path:
    """Per-subpackage per-backend variant cache root.

    CPU: `~/.cache/causal_conv1d_mojo/<subpkg>/cpu/`.
    GPU: `~/.cache/causal_conv1d_mojo/<subpkg>/<backend>/<arch>/`.

    Separating by backend keeps the cache human-inspectable and lets
    one machine hold artefacts for multiple GPUs side by side.
    """
    if backend == "cpu":
        return _CACHE_HOME / "causal_conv1d_mojo" / subpkg / "cpu"
    return _CACHE_HOME / "causal_conv1d_mojo" / subpkg / backend / backend_arch


def compile_and_load(
    *,
    subpkg: str,
    source_file: Path,
    include_dirs: Iterable[Path] = (),
    defines: Mapping[str, str] = {},
    mod_name: str,
    backend: str,
    backend_arch: str = "",
):
    """Compile a static `variant.mojo` with `-D` + `-I` and return the loaded module.

    `mojo build` resolves `from kernel import …` etc. via the ``-I``
    flag — one per entry in ``include_dirs`` — and gets the variant's
    comptime parameters via ``-D KEY=VALUE`` (read inside the `.mojo`
    file with `std.sys.get_defined_*`).

    The compiled `.so` is cached at
    ``cache_dir_for(subpkg, backend, backend_arch) / <mod_name>.hash-<h>.so``
    and is content-addressed over the source file, every `.mojo` in
    each include dir, the `defines` mapping, *and* the env signature
    returned by `_env_signature(backend)` — so anything that could
    change codegen / ABI / linking invalidates the cache automatically.

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
        backend: ``"cpu"``, ``"cuda"``, ``"rocm"``, or ``"metal"``.
            Selects the cache subdir and the env signals folded into
            the hash. GPU callers should obtain this via
            ``detect_gpu_backend()``.
        backend_arch: target arch for the GPU backend (e.g. ``"sm89"``,
            ``"gfx942"``, ``"macos15"``). Ignored when ``backend="cpu"``.

    Returns:
        The loaded Python extension module. Get entry-point functions
        via ``getattr(module, name)``.
    """
    include_dirs = [Path(d) for d in include_dirs]
    cache_dir = cache_dir_for(subpkg, backend, backend_arch)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env_sig = _env_signature(backend)
    src_hash = _hash_sources(source_file, include_dirs, defines, env_sig)
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
        except subprocess.CalledProcessError as e:
            # `capture_output=True` swallows the compiler diagnostics; surface
            # them so the user sees *which* line of which `.mojo` failed
            # instead of a bare "exit status 1".
            print(" FAILED", file=sys.stderr)
            stdout = _decode(e.stdout)
            stderr = _decode(e.stderr)
            details = []
            if stdout.strip():
                details.append(f"--- mojo stdout ---\n{stdout.rstrip()}")
            if stderr.strip():
                details.append(f"--- mojo stderr ---\n{stderr.rstrip()}")
            details_str = ("\n\n" + "\n\n".join(details)) if details else ""
            raise RuntimeError(
                f"Compilation of {subpkg} variant {mod_name} failed "
                f"(exit {e.returncode}). Command: {' '.join(cmd)}"
                f"{details_str}"
            ) from e
        except Exception as e:
            print(" FAILED", file=sys.stderr)
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


@functools.cache
def detect_gpu_backend() -> tuple[str, str]:
    """Probe torch to determine the active GPU backend + arch identifier.

    Returns ``(backend, arch)``:
        - ``("cuda", "smXY")`` for NVIDIA (cubin is sm-locked).
        - ``("rocm", "gfxNNN")`` for AMD (HSACO is gfx-locked; we
          strip the ``:sramecc+:xnack-`` suffix that ``gcnArchName``
          appends because those modes don't affect the cached binary
          for our kernels).
        - ``("metal", "macosNN")`` for Apple. Metal AIR is largely
          forward-compatible across chip generations within a macOS
          major; we key on the macOS major version because system
          updates have invalidated cached AIR in the past.

    Cached for the process — switching GPUs mid-process isn't
    supported anyway. Raises ``RuntimeError`` if no GPU backend is
    available; the GPU subpackage shouldn't be in that codepath.
    """
    import torch  # noqa: PLC0415  — torch is heavy; defer until needed

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            arch = torch.cuda.get_device_properties(0).gcnArchName
            arch = arch.split(":")[0]  # drop sramecc/xnack mode suffix
            return ("rocm", arch)
        major, minor = torch.cuda.get_device_capability(0)
        return ("cuda", f"sm{major}{minor}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        macos_major = (platform.mac_ver()[0] or "0").split(".")[0]
        return ("metal", f"macos{macos_major}")
    raise RuntimeError(
        "no GPU backend available — install torch with cuda / rocm / mps support"
    )


# --------------------------------------------------------------------------
# Env signature: anything that affects codegen / ABI / linking but isn't
# captured by source contents + defines.
# --------------------------------------------------------------------------


def _env_signature(backend: str) -> dict[str, str]:
    """Stable signals that should bust the cache when they change.

    Common to every backend:
        * **soabi**: Python C-extension ABI tag (e.g. ``cpython-313-x86_64-linux-gnu``).
          Captures Python minor version + arch + OS in one field. A
          ``.so`` built against cpython 3.13 won't safely load under
          cpython 3.14.
        * **mojo_version**: ``mojo --version`` output. Includes the
          build hash, so nightly bumps invalidate.
        * **modular_root**: path to the modular SDK install. Mojo
          bakes this into the `.so`'s ``RUNPATH`` so the loader can
          find ``libKGENCompilerRTShared.so`` — if the path moves
          (different venv, reinstalled wheel) the cached `.so` is
          un-loadable.
        * **jit_common_hash**: this file's own contents. Defensive
          against future changes to the `mojo build` invocation
          (extra flags etc.) that wouldn't otherwise show up in the
          source/defines hash.

    CUDA-only:
        * **ptxas**: which ptxas produced the cubin embedded in the
          `.so`. See ``_ptxas_signature``. AMD/Metal use bundled
          assemblers shipped with the modular SDK, so they're already
          subsumed by ``mojo_version``.
    """
    sig = {
        "soabi": sysconfig.get_config_var("SOABI") or "",
        "mojo_version": _mojo_version(),
        "modular_root": _modular_root(),
        "jit_common_hash": _self_hash(),
    }
    if backend == "cuda":
        sig["ptxas"] = _ptxas_signature()
    return sig


@functools.cache
def _mojo_version() -> str:
    r = subprocess_run_mojo(["--version"], capture_output=True, check=True)
    out = r.stdout
    if isinstance(out, bytes):
        out = out.decode("utf-8", errors="replace")
    return out.strip()


@functools.cache
def _modular_root() -> str:
    """Path to the modular SDK install — baked into the `.so` RUNPATH."""
    from mojo._package_root import get_package_root  # noqa: PLC0415

    root = get_package_root()
    return str(root)


@functools.cache
def _self_hash() -> str:
    """Hash of this file's contents — bust the cache on jit infra changes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@functools.cache
def _ptxas_signature() -> str:
    """Identify the ptxas that mojo will hand its PTX to.

    Mirrors the env-var-vs-vendored-vs-bundled logic in
    ``causal_conv1d_mojo/__init__.py`` so we don't accidentally treat
    "user-overridden external ptxas" the same as "we set it to the
    cu12 vendored one".

    Three cases:
        * **bundled**: ``MODULAR_NVPTX_COMPILER_PATH`` unset → mojo
          uses the ptxas shipped with the modular SDK. Subsumed by
          ``mojo_version`` already.
        * **cu12**: env var points at the vendored
          ``nvidia-cuda-nvcc-cu12`` wheel's ptxas (what
          ``__init__.py`` sets by default). Key on the pip package
          version — no subprocess needed.
        * **external**: env var points elsewhere (system CUDA toolkit
          etc.). Hash the path plus the binary's ``--version`` output;
          a system upgrade can replace the binary in place without
          the path changing.
    """
    env_path = os.environ.get("MODULAR_NVPTX_COMPILER_PATH", "")
    if not env_path:
        return "bundled"
    out = subprocess.run(
        [env_path, "--version"],
        capture_output=True,
        check=True,
        text=True,
    )
    return f"external:{env_path}:{out.stdout.strip()}"


def _decode(buf) -> str:
    """`bytes | str | None` → `str`. subprocess returns either depending
    on whether `text=True` was passed; we always pass `capture_output=True`
    without `text=True`, so we typically get bytes."""
    if buf is None:
        return ""
    if isinstance(buf, bytes):
        return buf.decode("utf-8", errors="replace")
    return buf


def _hash_sources(
    source_file: Path,
    include_dirs: Iterable[Path],
    defines: Mapping[str, str],
    env_sig: Mapping[str, str],
) -> str:
    """Content-hash the source file + every `.mojo` in each include dir
    + defines + env signature."""
    hasher = hashlib.sha256()
    hasher.update(source_file.name.encode())
    hasher.update(source_file.read_bytes())
    for d in include_dirs:
        for f in sorted(Path(d).glob("*.mojo")):
            hasher.update(str(f).encode())
            hasher.update(f.read_bytes())
    for k in sorted(defines):
        hasher.update(f"{k}={defines[k]}\n".encode())
    for k in sorted(env_sig):
        hasher.update(f"env:{k}={env_sig[k]}\n".encode())
    return hasher.hexdigest()[:16]
