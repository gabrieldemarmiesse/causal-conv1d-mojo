"""JIT-on-first-use dispatcher for the CPU fused backward."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load

_BWD_FULL_CPU_DIR = Path(__file__).resolve().parent
_PKG_DIR = _BWD_FULL_CPU_DIR.parent
_VARIANT_MOJO = _BWD_FULL_CPU_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_bwd_full_cpu(config: tuple, runtime_args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single CPU bwd_full call."""
    variant_fn = _get_variant_fn(config)
    variant_fn(*runtime_args)


def _mod_name(config: tuple) -> str:
    (dt, w, hb, hs, hi, silu) = config
    return f"{_DTYPE_NAME[dt]}_w{w}_hb{int(hb)}_hs{int(hs)}_hi{int(hi)}_silu{int(silu)}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, w, hb, hs, hi, silu) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "HAS_SEQ_IDX": b(hs),
        "HAS_INITIAL_STATES": b(hi),
        "APPLY_SILU": b(silu),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple):
    module = compile_and_load(
        subpkg="bwd_full_cpu",
        source_file=_VARIANT_MOJO,
        include_dirs=(_BWD_FULL_CPU_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=_mod_name(config),
        backend="cpu",
    )
    return module.causal_conv1d_bwd_full_cpu_variant
