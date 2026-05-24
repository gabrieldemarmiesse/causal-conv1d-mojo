"""JIT-on-first-use dispatcher for flash_attn fwd.

Each unique runtime config (dtype × head_dim × use_external_stream)
compiles the static ``fwd/variant.mojo`` once via ``mojo build -D
KEY=VALUE …`` and caches the resulting ``.so`` on disk. Mirrors
causal-conv1d-mojo's `fwd/_jit.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from flash_attn_mojo._jit_common import compile_and_load, detect_gpu_backend

_FWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _FWD_DIR.parent
_VARIANT_MOJO = _FWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
# `get_defined_dtype` in std.sys parses these via `DType._from_str`.
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_fwd(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call.

    ``args`` is the 22-tuple of runtime values built by
    ``fwd/__init__.py::native_fwd``.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    # Tack ctx_handle on as the 23rd positional arg — the variant
    # entry point destructures `args[22]` for it.
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    # See `fwd/__init__.py::native_fwd` for the runtime tuple layout.
    # Indices 21, 22, 23 are the compile-time gating fields appended
    # after the kernel's actual positional args (which end at 20 with
    # the CUDA stream handle).
    dtype_code = args[21]
    head_dim = args[22]
    use_external_stream = bool(args[23])
    return (dtype_code, head_dim, use_external_stream)


def _mod_name(config: tuple) -> str:
    """Readable, deterministic identifier for a config."""
    (dt, hd, ues) = config
    return f"{_DTYPE_NAME[dt]}_hd{hd}_extstr{int(ues)}"


def _defines(config: tuple) -> dict[str, str]:
    """Materialise the config as `-D KEY=VALUE` pairs for `mojo build`."""
    (dt, hd, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "HEAD_DIM": str(hd),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="fwd",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.flash_attn_fwd_variant
    acquire = module.flash_attn_fwd_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
