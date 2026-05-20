"""JIT-on-first-use dispatcher for the CPU update."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load

_UPDATE_CPU_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_CPU_DIR.parent
_VARIANT_MOJO = _UPDATE_CPU_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_update_cpu(args: tuple) -> None:
    variant_fn = _get_variant_fn(_config_from_args(args))
    variant_fn(*args)


def _config_from_args(args: tuple) -> tuple:
    return (
        args[22],  # dtype_code
        args[23],  # width
        bool(args[20]),  # has_bias
        bool(args[21]),  # apply_silu
        bool(args[24]),  # has_state_indices
        bool(args[26]),  # is_circular
    )


def _mod_name(config: tuple) -> str:
    (dt, w, hb, silu, hi, circ) = config
    return (
        f"{_DTYPE_NAME[dt]}_w{w}"
        f"_hb{int(hb)}_silu{int(silu)}_hi{int(hi)}_circ{int(circ)}"
    )


def _defines(config: tuple) -> dict[str, str]:
    (dt, w, hb, silu, hi, circ) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "APPLY_SILU": b(silu),
        "HAS_STATE_INDICES": b(hi),
        "IS_CIRCULAR": b(circ),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple):
    module = compile_and_load(
        subpkg="update_cpu",
        source_file=_VARIANT_MOJO,
        include_dirs=(_UPDATE_CPU_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=_mod_name(config),
    )
    return module.causal_conv1d_update_cpu_variant
