"""JIT-on-first-use dispatcher for causal_conv1d update.

Each unique runtime config (dtype × width × has_bias × apply_silu ×
has_state_indices × is_circular × use_external_stream) compiles the
static ``update/variant.mojo`` once via ``mojo build -D KEY=VALUE …``
and caches the resulting ``.so`` on disk.

Performance note (AMD-specific): The Mojo `DeviceContext()` constructor
calls `hipStreamCreate` under the hood, and the matching `__del__`
calls `hipStreamDestroy`. At decode shapes the update kernel is only
~3 us of GPU work, so a per-call stream churn (1 ms+ on the CPU side)
dwarfs everything else. Each variant exposes a
`causal_conv1d_update_acquire_ctx` entry point; the first call obtains
a process-lifetime DeviceContext handle, and subsequent dispatches
pass it in to `launch_update` — no new hipStream is created per call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_UPDATE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_DIR.parent
_VARIANT_MOJO = _UPDATE_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_update(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single update call."""
    config = _config_from_args(args)
    variant_fn, ctx_handle = _get_variant_fn(config)
    # Tack ctx_handle on as the 31st positional arg — the variant
    # entry point destructures `args[30]` for it.
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    return (
        args[22],  # dtype_code
        args[24],  # width
        bool(args[20]),  # has_bias
        bool(args[21]),  # apply_silu
        bool(args[25]),  # has_state_indices
        bool(args[27]),  # is_circular
        # See fwd/_jit.py for why this is comptime, not runtime branch.
        bool(args[29]),  # use_external_stream (1 for CUDA, 0 for Metal)
    )


def _mod_name(config: tuple) -> str:
    (dt, w, hb, silu, hi, circ, ues) = config
    return (
        f"{_DTYPE_NAME[dt]}_w{w}"
        f"_hb{int(hb)}_silu{int(silu)}_hi{int(hi)}_circ{int(circ)}"
        f"_extstr{int(ues)}"
    )


def _defines(config: tuple) -> dict[str, str]:
    (dt, w, hb, silu, hi, circ, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "APPLY_SILU": b(silu),
        "HAS_STATE_INDICES": b(hi),
        "IS_CIRCULAR": b(circ),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple):
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="update",
        source_file=_VARIANT_MOJO,
        include_dirs=(_UPDATE_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_update_variant
    acquire = module.causal_conv1d_update_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
