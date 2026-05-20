"""JIT-on-first-use dispatcher for causal_conv1d fwd.

Each unique runtime config (dtype × width × has_bias × has_seq_idx ×
has_initial_states × apply_silu × contig_inner × aligned_seq) lazily
generates a tiny single-variant ``.mojo`` file at call time, compiles
it via ``mojo build``, and caches the resulting ``.so`` on disk.
Subsequent calls (this process or any future one) hit the cache and
just dispatch.

The codegen + compile + load + cache plumbing lives in
``causal_conv1d_mojo._jit_common``. This module just owns the bits
that are fwd-specific: how to read the config out of the Python-side
args, how to name a variant, and how to template the variant source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load_variant

_FWD_DIR = Path(__file__).resolve().parent
_CTX_MOJO = _FWD_DIR.parent / "_ctx.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_EXPR = {0: "DType.float16", 1: "DType.bfloat16", 2: "DType.float32"}

# Per-thread element count: 8 for fp16/bf16, 4 for fp32. Mirrors
# `kNEltsFwd` in fwd/common.mojo.
_KN_ELTS = {0: 8, 1: 8, 2: 4}
# Block size (`kNThreads` in fwd/common.mojo).
_KNTHREADS = 128


def call_fwd(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call.

    ``args`` is the 29-tuple of runtime values built by
    ``fwd/__init__.py::native_fwd``.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    # Tack ctx_handle on as the 30th positional arg — the variant
    # entry point destructures `args[29]` for it. Avoids the per-call
    # hipStreamCreate/Destroy churn from `var ctx = DeviceContext()`.
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[17]
    width = args[23]
    has_bias = bool(args[15])
    apply_silu = bool(args[16])
    has_seq_idx = bool(args[19])
    has_initial_states = bool(args[24])
    seqlen = args[6]
    contig_inner = args[9] == 1 and args[11] == 1 and args[14] == 1
    aligned_seq = (seqlen % (_KNTHREADS * _KN_ELTS[dtype_code])) == 0
    # `vec_aligned` is the weaker "seqlen % kNElts == 0" — true for any
    # power-of-two seqlen on fp16/bf16 (kNElts=8) and fp32 (kNElts=4).
    # When this holds, every thread's kNElts slice either fits entirely
    # inside [0, seqlen) or starts past it, so the partial-chunk scalar
    # fallback path in the kernel becomes statically dead. Mirrors
    # upstream's `kIsVecLoad` BOOL_SWITCH (gated on the same condition).
    vec_aligned = (seqlen % _KN_ELTS[dtype_code]) == 0
    # `use_external_stream` is a comptime gate: True for CUDA/HIP (wrap
    # torch's CUstream/hipStream and enqueue on it), False for Metal
    # (no CUDA-style streams; enqueue on the DeviceContext directly +
    # sync after). Passed as comptime so the variant only codegens one
    # branch — a runtime `if` here costs ~30 μs/call on NVIDIA, even
    # when the branch is perfectly predictable. `args[29]` is set by
    # the Python wrappers (`native_fwd` passes 1, `native_fwd_mps`
    # passes 0). Can't be derived from `stream_handle_addr` itself
    # because torch's default CUDA stream has cuda_stream == 0.
    use_external_stream = bool(args[29])
    return (
        dtype_code,
        width,
        has_bias,
        has_seq_idx,
        has_initial_states,
        apply_silu,
        contig_inner,
        aligned_seq,
        vec_aligned,
        use_external_stream,
    )


def _mod_name(config: tuple) -> str:
    """Readable, deterministic identifier for a config.

    Used as the cache directory name, the Python module name, and the
    PyInit symbol suffix in the generated `.so`. Reading it should be
    enough to reproduce the config by hand.
    """
    (dt, w, hb, hs, hi, silu, c, a, va, ues) = config
    return (
        f"{_DTYPE_NAME[dt]}_w{w}"
        f"_hb{int(hb)}_hs{int(hs)}_hi{int(hi)}_silu{int(silu)}"
        f"_contig{int(c)}_aligned{int(a)}_vec{int(va)}_extstr{int(ues)}"
    )


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple):
    import sys

    mod_name = _mod_name(config)
    fn = compile_and_load_variant(
        subpkg="fwd",
        source_dir=_FWD_DIR,
        shared_files=("kernel.mojo", "common.mojo", "launch.mojo", _CTX_MOJO),
        mod_name=mod_name,
        variant_source=_generate_variant_source(mod_name, config),
        entry_point_name="causal_conv1d_fwd_variant",
    )
    # The shared loader stashes the loaded variant module in
    # sys.modules so we can grab the one-shot ctx-handle helper without
    # re-importing the .so.
    module = sys.modules[mod_name]
    acquire = getattr(module, "causal_conv1d_fwd_acquire_ctx")
    ctx_handle = int(acquire(()))
    return fn, ctx_handle


def _generate_variant_source(mod_name: str, config: tuple) -> str:
    (
        dtype_code,
        width,
        has_bias,
        has_seq_idx,
        has_initial_states,
        apply_silu,
        contig_inner,
        aligned_seq,
        vec_aligned,
        use_external_stream,
    ) = config
    return f'''\
"""JIT-generated variant for causal_conv1d_fwd (config-frozen).

Generated by causal_conv1d_mojo.fwd._jit; do not hand-edit.
Module: {mod_name}
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from launch import launch_fwd
from _ctx import acquire_ctx_handle


def causal_conv1d_fwd_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext.

    Called once per variant from the Python side; the returned address
    is reused for every subsequent `causal_conv1d_fwd_variant` call.
    """
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def causal_conv1d_fwd_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr: Int = Int(py=args[0])
    var w_addr: Int = Int(py=args[1])
    var b_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var batch_int: Int = Int(py=args[4])
    var dim_int: Int = Int(py=args[5])
    var seqlen_int: Int = Int(py=args[6])
    var x_b_stride: Int = Int(py=args[7])
    var x_c_stride: Int = Int(py=args[8])
    var x_l_stride: Int = Int(py=args[9])
    var w_c_stride: Int = Int(py=args[10])
    var w_w_stride: Int = Int(py=args[11])
    var o_b_stride: Int = Int(py=args[12])
    var o_c_stride: Int = Int(py=args[13])
    var o_l_stride: Int = Int(py=args[14])
    var stream_handle_addr: Int = Int(py=args[18])
    var seq_idx_addr: Int = Int(py=args[20])
    var seq_idx_b_stride: Int = Int(py=args[21])
    var seq_idx_l_stride: Int = Int(py=args[22])
    var initial_states_addr: Int = Int(py=args[25])
    var initial_states_b_stride: Int = Int(py=args[26])
    var initial_states_c_stride: Int = Int(py=args[27])
    var initial_states_l_stride: Int = Int(py=args[28])
    # args[29] is `use_external_stream` (already baked into the comptime
    # template params — we don't read it at runtime); ctx_handle is the
    # 30th positional, appended by `call_fwd`.
    var ctx_handle_addr: Int = Int(py=args[30])

    if batch_int == 0 or dim_int == 0 or seqlen_int == 0:
        return PythonObject(None)

    launch_fwd[
        {_DTYPE_EXPR[dtype_code]},
        {width},
        {has_bias},
        {has_seq_idx},
        {has_initial_states},
        {apply_silu},
        {contig_inner},
        {aligned_seq},
        {vec_aligned},
        {use_external_stream},
    ](
        batch_int,
        dim_int,
        seqlen_int,
        x_addr,
        w_addr,
        b_addr,
        o_addr,
        seq_idx_addr,
        initial_states_addr,
        x_b_stride,
        x_c_stride,
        x_l_stride,
        w_c_stride,
        w_w_stride,
        o_b_stride,
        o_c_stride,
        o_l_stride,
        seq_idx_b_stride,
        seq_idx_l_stride,
        initial_states_b_stride,
        initial_states_c_stride,
        initial_states_l_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_{mod_name}() -> PythonObject:
    try:
        var m = PythonModuleBuilder("{mod_name}")
        m.def_py_function[causal_conv1d_fwd_variant]("causal_conv1d_fwd_variant")
        m.def_py_function[causal_conv1d_fwd_acquire_ctx]("causal_conv1d_fwd_acquire_ctx")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
'''
