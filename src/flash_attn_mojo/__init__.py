"""flash-attn, fused into Mojo kernels and called via direct
Python <-> Mojo CPython extensions (no MAX framework).

Layout: each of the public entry points lives in its own subpackage
(`fwd/`, `bwd/`) for the actual conv kernels, with thin Python
wrappers at the package root that handle autograd plumbing.

GPU subpackages use JIT-on-first-use: each runtime config compiles
its own single-variant `.so` via `mojo build` at call time and
caches the result under
`$XDG_CACHE_HOME/flash_attn_mojo/<subpkg>/<backend>/<gpu_arch>/<cpu_tag>/`.
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

# Other mojo-package internals import this lazily — keep it imported
# here so the first call doesn't pay the hook-registration cost.
import mojo.importer  # noqa: F401, E402

from flash_attn_mojo._fn import flash_attn_func  # noqa: E402
from flash_attn_mojo.reference import flash_attn_ref  # noqa: E402

__version__ = "0.0.1"

__all__ = [
    "flash_attn_func",
    "flash_attn_ref",
]
