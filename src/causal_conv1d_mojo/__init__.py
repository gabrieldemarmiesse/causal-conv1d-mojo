"""causal_conv1d, fused into Mojo kernels and called via direct
Python <-> Mojo CPython extensions (no MAX framework).

Layout: each of the six Python entry points lives in its own
subpackage (`fwd/`, `bwd_full/`, `update/` for GPU; `fwd_cpu/`,
`bwd_full_cpu/`, `update_cpu/` for the CPU fallback). All six use
the same JIT-on-first-use model: each runtime config compiles its
own single-variant `.so` via `mojo build` at call time and caches
the result under
`$XDG_CACHE_HOME/causal_conv1d_mojo/<subpkg>/<backend>[/<arch>]/`.
See `<subpkg>/_jit.py` and `_jit_common.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

# MAX 26.3+ links against CUDA-13's internal libnvptxcompiler and refuses to
# load on NVIDIA driver <580 (CUDA <13). Pointing MAX at an external CUDA-12
# ptxas via MODULAR_NVPTX_COMPILER_PATH disables the driver-version guard.
if "MODULAR_NVPTX_COMPILER_PATH" not in os.environ:
    try:
        import nvidia.cuda_nvcc  # type: ignore[import-not-found]

        _ptxas = Path(nvidia.cuda_nvcc.__file__).parent / "bin" / "ptxas"
        if _ptxas.is_file():
            os.environ["MODULAR_NVPTX_COMPILER_PATH"] = str(_ptxas)
    except ImportError:
        pass

# Registers `mojo.importer`'s sys.meta_path hook. We no longer rely
# on it for the CPU subpackages (they go through the same JIT path
# as the GPU subpackages now), but other mojo-package internals
# import this lazily — keep it imported here so the first call
# doesn't pay the hook-registration cost.
import mojo.importer  # noqa: F401, E402

from causal_conv1d_mojo._fn import causal_conv1d_fn
from causal_conv1d_mojo._update import causal_conv1d_update
from causal_conv1d_mojo.causal_conv1d_varlen import (
    causal_conv1d_varlen_states,
    causal_conv1d_varlen_states_ref,
)
from causal_conv1d_mojo.reference import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)


__version__ = "1.6.1"

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_ref",
    "causal_conv1d_update",
    "causal_conv1d_update_ref",
    "causal_conv1d_varlen_states",
    "causal_conv1d_varlen_states_ref",
]
