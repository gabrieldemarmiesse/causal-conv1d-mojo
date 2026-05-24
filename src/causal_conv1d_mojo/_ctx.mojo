"""Shared GPU DeviceContext helper for the JIT subpackages.

Picked up by each variant via the package-root entry in the
`include_dirs=...` tuple passed to `_jit_common.compile_and_load`
(see each GPU subpackage's `_jit.py`). Lives at the package root
rather than inside any one subpackage so the three GPU launchers
(`fwd`, `bwd_full`, `update`) can share a single definition.

On AMD `var ctx = DeviceContext()` per call ends up issuing
`hipStreamCreate` + matching `hipStreamDestroy` each launch. Those
calls bleed into `torch.profiler`'s `self_device_time_total` for the
surrounding kernel and add measurable per-call overhead. The Python
side calls `acquire_ctx_handle` once per variant on first use, caches
the returned address, and threads it through every subsequent launch;
the launcher wraps it via the non-owning DeviceContext constructor so
no fresh hipStream is created per call.
"""

from std.gpu.host import DeviceContext


def acquire_ctx_handle() raises -> Int:
    """Create a DeviceContext, retain its handle, and leak the wrapper.

    The Python side calls this once per variant and caches the returned
    integer. The handle stays alive for the duration of the process —
    the matching release happens at process exit (or never).

    Returns the address of the underlying C++ DeviceContext as an Int.
    """
    var ctx = DeviceContext()
    # Retain so the handle survives this function's __del__; the caller
    # now owns the extra refcount.
    ctx._retain()
    var raw_ptr = ctx._handle.value()
    return Int(raw_ptr)
