"""JIT-on-first-use dispatcher for causal_conv1d bwd_full.

Each unique runtime config (dtype × n_elts × width × has_bias ×
has_seq_idx × has_initial_states × apply_silu × contig_inner ×
aligned_seq × use_external_stream) compiles the static
``bwd_full/variant.mojo`` once via ``mojo build -D KEY=VALUE …`` and
caches the resulting ``.so`` on disk.

Performance note (AMD-specific): The Mojo `DeviceContext()`
constructor issues `hipStreamCreate` and the matching `__del__`
issues `hipStreamDestroy`. At small-batch shapes the bwd kernel is
only ~6-10 us of GPU work, so per-call stream churn shows up in
torch.profiler. Each variant exposes a
``causal_conv1d_bwd_full_acquire_ctx`` entry point; the first call
from Python invokes it to obtain a process-lifetime DeviceContext
handle (refcount-retained so the wrapper destructor is a no-op), and
caches it. Subsequent dispatches pass that handle in to
`launch_bwd_full`, which wraps it via the doc-hidden non-owning
constructor — no new hipStream per call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_BWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _BWD_DIR.parent
_VARIANT_MOJO = _BWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}

# Wide per-thread element count (16-byte LDG): 8 for fp16/bf16, 4 for
# fp32. Mirrors `kNEltsBwd_for` in bwd_full/common.mojo.
_KN_ELTS_WIDE = {0: 8, 1: 8, 2: 4}
_KN_ELTS_NARROW = 4
# Block size (`kNThreads` in bwd_full/common.mojo).
_KNTHREADS = 128


def call_bwd_full(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single bwd_full call."""
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    # Tack ctx_handle on as the 41st positional arg — the variant
    # entry point destructures `args[40]` for it. (args[39] is the
    # `use_external_stream` flag, already a comptime define.)
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[23]
    width = args[25]
    has_bias = bool(args[21])
    apply_silu = bool(args[22])
    has_seq_idx = bool(args[26])
    has_initial_states = bool(args[30])
    seqlen = args[9]
    contig_inner = (
        args[12] == 1  # x_l_stride
        and args[14] == 1  # w_w_stride
        and args[17] == 1  # dout_l_stride
        and args[20] == 1  # dx_l_stride
    )

    # Match the original dispatcher's runtime n_elts pick: wide only if
    # (a) wide differs from narrow (i.e. dtype is 16-bit) and (b) seqlen
    # is aligned to kNThreads * wide. Else narrow (4).
    n_elts_wide = _KN_ELTS_WIDE[dtype_code]
    use_wide = (
        n_elts_wide != _KN_ELTS_NARROW and (seqlen % (_KNTHREADS * n_elts_wide)) == 0
    )
    n_elts = n_elts_wide if use_wide else _KN_ELTS_NARROW
    aligned_seq = (seqlen % (_KNTHREADS * n_elts)) == 0
    # See fwd/_jit.py for why this is comptime instead of a runtime
    # branch on `stream_handle_addr`. Python wrapper sets 1 for CUDA,
    # 0 for Metal.
    use_external_stream = bool(args[39])

    return (
        dtype_code,
        n_elts,
        width,
        has_bias,
        has_seq_idx,
        has_initial_states,
        apply_silu,
        contig_inner,
        aligned_seq,
        use_external_stream,
    )


def _mod_name(config: tuple) -> str:
    (dt, ne, w, hb, hs, hi, silu, c, a, ues) = config
    return (
        f"{_DTYPE_NAME[dt]}_n{ne}_w{w}"
        f"_hb{int(hb)}_hs{int(hs)}_hi{int(hi)}_silu{int(silu)}"
        f"_contig{int(c)}_aligned{int(a)}_extstr{int(ues)}"
    )


def _defines(config: tuple) -> dict[str, str]:
    (dt, ne, w, hb, hs, hi, silu, c, a, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "N_ELTS": str(ne),
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "HAS_SEQ_IDX": b(hs),
        "HAS_INITIAL_STATES": b(hi),
        "APPLY_SILU": b(silu),
        "CONTIG_INNER": b(c),
        "ALIGNED_SEQ": b(a),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple):
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="bwd_full",
        source_file=_VARIANT_MOJO,
        include_dirs=(_BWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_bwd_full_variant
    acquire = module.causal_conv1d_bwd_full_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
