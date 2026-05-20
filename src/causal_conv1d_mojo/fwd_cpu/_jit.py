"""JIT-on-first-use dispatcher for the CPU forward.

Mirrors `fwd/_jit.py`: each unique config (dtype × width × has_bias ×
has_seq_idx × has_initial_states × apply_silu) compiles the static
``fwd_cpu/variant.mojo`` once via ``mojo build -D KEY=VALUE …`` and
caches the resulting ``.so`` on disk. Replaces the old AOT comptime-
sweep dispatcher, which took ~30 s on first import even when the
caller only needed one variant.

`config` and `runtime_args` are deliberately split: comptime values
are baked into the `.so` via `-D`, so the variant `.mojo` only ever
sees runtime values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load

_FWD_CPU_DIR = Path(__file__).resolve().parent
_PKG_DIR = _FWD_CPU_DIR.parent
_VARIANT_MOJO = _FWD_CPU_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_fwd_cpu(config: tuple, runtime_args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single CPU fwd call.

    Args:
        config: 6-tuple
            (dtype_code, width, has_bias, has_seq_idx,
             has_initial_states, apply_silu).
        runtime_args: positional args forwarded to the variant `.mojo`.
    """
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
        subpkg="fwd_cpu",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_CPU_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=_mod_name(config),
    )
    return module.causal_conv1d_fwd_cpu_variant
