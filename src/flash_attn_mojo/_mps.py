"""Apple Silicon (MPS) interop helpers.

The Mojo GPU kernels run on Apple Metal via `DeviceContext`. A Mojo
`DeviceBuffer`'s internal `_device_ptr` is the Metal-3.1 `gpuAddress`
of the underlying `MTLBuffer` (a 64-bit GPU virtual address). On
NVIDIA, torch's `tensor.data_ptr()` is already a CUDA device VA, so
plumbing it straight through to the kernel works. On Apple, however,
torch's `tensor.data_ptr()` is the `id<MTLBuffer>` Obj-C object
pointer (verified: `object_getClassName` -> `AGXG16GFamilyBuffer`),
NOT the GPU VA — Mojo can't dereference it.

We extract the GPU VA via the same selector Mojo uses internally:

  [storage.MTLBuffer gpuAddress] + (tensor.data_ptr() - storage.data_ptr())

The base `gpuAddress` is per-`MTLBuffer`; the tensor's storage offset
(stored by torch as a byte delta on top of the buffer obj pointer)
gives the per-tensor location. Result: a 64-bit GPU VA that Mojo's
JIT-compiled Metal kernel can use exactly like a CUDA pointer, with
no copy through host memory.

Synchronization: torch's MPS runs on its own command queue; Mojo's
`DeviceContext` runs on its own. The `MTLDevice` is shared, so memory
is consistent once both queues have flushed. The Python wrappers in
`_fn.py` / `_update.py` call `torch.mps.synchronize()` before
launching the Mojo kernel and rely on `ctx.synchronize()` inside the
JIT'd Mojo entry point to flush our side before returning.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _objc() -> ctypes.CDLL:
    """Lazy-load libobjc and pre-bind the C ABI types for the few
    selectors we use. `objc_msgSend.argtypes` MUST be set on Apple
    Silicon — without it the ABI defaults are wrong for variadic
    msgSend and you'll get a segfault on entry.
    """
    libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.objc_msgSend.restype = ctypes.c_uint64
    libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return libobjc


@lru_cache(maxsize=1)
def _sel_gpu_address() -> int:
    libobjc = _objc()
    return libobjc.sel_registerName(b"gpuAddress")


def gpu_address(t: torch.Tensor) -> int:
    """Return the Metal GPU virtual address of `t`'s first element.

    Handles non-zero storage offsets (sliced views, etc.) by adding
    the byte delta between `tensor.data_ptr()` and the storage's
    `data_ptr()` to the buffer's base `gpuAddress`. The kernel reads
    elements via `base + b*stride_b + c*stride_c + l*stride_l`, with
    strides already in elements, so this base + offset is enough.
    """
    storage = t.untyped_storage()
    buf_obj = storage.data_ptr()
    if buf_obj == 0:
        return 0
    libobjc = _objc()
    base_gpu = libobjc.objc_msgSend(buf_obj, _sel_gpu_address())
    offset_bytes = t.data_ptr() - buf_obj
    return base_gpu + offset_bytes


def gpu_address_or_zero(t: torch.Tensor | None) -> int:
    """Same as `gpu_address` but returns 0 for `None` — matches the
    `_ptr()` helper in `_dtype.py` used by the kernel argument-builders.
    """
    if t is None:
        return 0
    return gpu_address(t)
