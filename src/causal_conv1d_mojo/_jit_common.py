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
from types import ModuleType
from collections.abc import Iterable, Mapping

from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str, backend: str, backend_arch: str = "") -> Path:
    """Per-subpackage per-backend per-CPU variant cache root.

    CPU subpkg: `<root>/<subpkg>/cpu/<cpu_tag>/`.
    GPU subpkg: `<root>/<subpkg>/<backend>/<arch>/<cpu_tag>/`.

    The trailing `<cpu_tag>` segment isolates artefacts built on
    different host CPUs. Mojo's CPU codegen defaults to `-march=native`
    (verified by inspecting the generated `.so`: AVX2 instructions on a
    Haswell-era box, AVX-512 on Sapphire Rapids), so a `.so` built on
    one CPU will SIGILL on a host with fewer ISA extensions. This
    matters in shared-cache scenarios — HPC clusters with one
    `~/.cache` mounted across heterogeneous compute nodes are the
    classic footgun. Even GPU-subpackage `.so`s contain host-side
    glue code, so the per-CPU split applies to *every* backend, not
    just CPU.
    """
    cpu_tag = _cpu_microarch_tag()
    if backend == "cpu":
        return _CACHE_HOME / "causal_conv1d_mojo" / subpkg / "cpu" / cpu_tag
    return (
        _CACHE_HOME / "causal_conv1d_mojo" / subpkg / backend / backend_arch / cpu_tag
    )


def compile_and_load(
    *,
    subpkg: str,
    source_file: Path,
    include_dirs: Iterable[Path] = (),
    defines: Mapping[str, str] = {},
    mod_name: str,
    backend: str,
    backend_arch: str = "",
) -> ModuleType:
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
        # Production safety net: when `CAUSAL_CONV1D_USE_CACHE_ONLY` is
        # set in the environment, refuse to JIT-compile and fail loudly
        # instead. Use case: shipping a container that includes a
        # pre-warmed cache directory. Any cache miss in production —
        # e.g. an unexpected input shape that maps to a variant the
        # warmup run didn't exercise — would otherwise silently incur
        # ~1.2 s of JIT compile in the request hot path; this flag
        # converts that into a noisy error so the deployment process
        # can be fixed (extend the warmup, or remove the flag).
        if os.environ.get("CAUSAL_CONV1D_USE_CACHE_ONLY"):
            env_sig_lines = "\n".join(f"    {k} = {v!r}" for k, v in env_sig.items())
            raise RuntimeError(
                f"causal_conv1d_mojo: cache miss for {subpkg} variant "
                f"{mod_name!r}, but CAUSAL_CONV1D_USE_CACHE_ONLY is set. "
                f"Expected `.so` at {so_path!s}.\n"
                f"\n"
                f"This usually means the production host's env signature "
                f"differs from the one the cache was warmed on, or the "
                f"input shape maps to a variant the warmup didn't "
                f"exercise. Current env signature:\n"
                f"{env_sig_lines}\n"
                f"\n"
                f"To fix: rerun the warmup on a host matching the above "
                f"signals exactly, or unset CAUSAL_CONV1D_USE_CACHE_ONLY "
                f"to allow on-demand JIT compilation."
            )
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
        # Pass the accelerator target explicitly so the compiler doesn't
        # auto-detect the GPU via the HIP/CUDA runtime.  Auto-detection can
        # return arch strings with suffixes (e.g. "gfx942:sramecc-:xnack-")
        # that Mojo's normalizer doesn't strip, causing a compile-time
        # constraint failure even for supported architectures (e.g. MI300A).
        if backend in ("cuda", "rocm") and backend_arch:
            cmd += ["--target-accelerator", backend_arch]
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
            raw = torch.cuda.get_device_properties(0).gcnArchName
            arch = raw.split(":")[0]  # drop sramecc/xnack mode suffix
            if not arch.startswith("gfx") or arch == "gfx":
                # Empty or malformed gfxNNN → silently keying on "gfx"
                # would risk sharing a cache slot across truly-different
                # ROCm targets. Fail loudly.
                raise RuntimeError(
                    f"ROCm device reports an unrecognised gcnArchName "
                    f"({raw!r}); refusing to share a cache slot across "
                    f"unknown AMD GPU targets. Open an issue with the "
                    f"output of `rocminfo | head -50`."
                )
            # MI300A and MI300X both report gfx942 from gcnArchName, but the
            # Mojo stdlib exposes them as distinct targets ("gfx942" → MI300X,
            # "mi300a" → MI300A).  Distinguish via the device name so the
            # cache doesn't conflate the two and the compiler gets the right
            # target.
            if arch == "gfx942":
                name = torch.cuda.get_device_name(0)
                if "MI300A" in name:
                    arch = "mi300a"
            return ("rocm", arch)
        major, minor = torch.cuda.get_device_capability(0)
        return ("cuda", f"sm{major}{minor}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        macos_major = (platform.mac_ver()[0] or "").split(".")[0]
        if not macos_major:
            # `platform.mac_ver()` empty on a Mac is exotic enough that
            # silently bucketing every such host together is wrong. Fail.
            raise RuntimeError(
                "Could not detect macOS major version via "
                "`platform.mac_ver()`. The JIT cache keys on this to "
                "avoid sharing Metal AIR across OS upgrades that have "
                "historically invalidated it. Open an issue with the "
                "output of `sw_vers`."
            )
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
        * **cpu_brand**: full host-CPU brand string (e.g.
          ``Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz``). Mojo's
          ``-march=native`` codegen bakes host-CPU SIMD instructions
          into the `.so` (AVX2/AVX-512 on x86, NEON/SVE on ARM, etc.).
          Running a `.so` built on one CPU on a host with fewer ISA
          extensions SIGILLs. The cache directory path also embeds a
          short tag derived from this string, so identical-CPU hits
          stay clustered; the full brand string going into the hash
          guards against the tag colliding across truly-different
          CPUs.
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
        "cpu_brand": _cpu_brand(),
        "jit_common_hash": _self_hash(),
    }
    if backend == "cuda":
        sig["ptxas"] = _ptxas_signature()
    return sig


@functools.cache
def _cpu_brand() -> str:
    """Full host-CPU brand string — distinguishes ISA generations.

    Mojo's CPU codegen defaults to `-march=native`, so the produced
    `.so` contains instructions specific to the build host's CPU
    (AVX2 on Haswell, AVX-512 on Sapphire Rapids, NEON+SVE on ARM
    server parts). The brand string identifies the CPU generation
    closely enough that two hosts sharing it can safely share a
    cached `.so`.

    Sources, in order of preference:
        Linux  — ``/proc/cpuinfo`` "model name" line (or "Hardware" /
                 "Processor" on ARM, where some kernels omit model name).
        macOS  — ``sysctl -n machdep.cpu.brand_string``.
        Other  — ``platform.processor()``.

    Raises ``RuntimeError`` if no source produces a non-empty brand —
    silently sharing the cache across unknown CPUs would let one user's
    AVX-512 binary SIGILL on another user's AVX2-only host. Better to
    fail loudly so the user can either fix the detection or open an
    issue for their platform.
    """
    sysname = platform.system()
    if sysname == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            text = ""
        for key in ("model name", "Hardware", "Processor"):
            for line in text.splitlines():
                if line.startswith(key) and ":" in line:
                    brand = line.split(":", 1)[1].strip()
                    if brand:
                        return brand
    elif sysname == "Darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                check=True,
                text=True,
            )
            brand = r.stdout.strip()
            if brand:
                return brand
        except Exception:
            pass
    fallback = platform.processor().strip()
    if fallback:
        return fallback
    raise RuntimeError(
        "Could not detect a host-CPU brand string. The JIT cache keys "
        "on the CPU model because mojo's `-march=native` codegen bakes "
        "host SIMD into each .so; silently sharing the cache across "
        "unknown CPUs risks SIGILL on hosts with fewer ISA extensions. "
        "Open an issue at "
        "https://github.com/gabrieldemarmiesse/causal-conv1d-mojo/issues "
        "with `uname -a` and (Linux) `cat /proc/cpuinfo | head -30` so "
        "we can teach the detector your platform."
    )


@functools.cache
def _cpu_microarch_tag() -> str:
    """Short filesystem-safe identifier derived from `_cpu_brand()`.

    Used as a directory segment in the cache path. We want it short
    (path components shouldn't bloat) and stable, but readable enough
    that ``ls ~/.cache/causal_conv1d_mojo/<sub>/cpu/`` is informative.
    Format: ``<sanitized-prefix>__<8-hex-of-full-brand>``. The hex
    suffix guarantees no collisions across truly-different CPUs even
    when the prefix happens to match (e.g. two Xeon Golds with the
    same first 24 characters).
    """
    brand = _cpu_brand()  # raises on undetectable CPUs
    # Sanitize: keep alphanumerics, replace everything else with `_`.
    # Then collapse runs of `_`, strip leading/trailing `_`, truncate.
    prefix = "".join(c if c.isalnum() else "_" for c in brand.lower())
    while "__" in prefix:
        prefix = prefix.replace("__", "_")
    prefix = prefix.strip("_")[:24]
    digest = hashlib.sha256(brand.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{prefix}__{digest}"


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


def _decode(buf: bytes | str | None) -> str:
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
