"""Shared `_silu_f32` helper for the GPU and CPU subpackages.

Lives at the package root so every subpackage (`fwd`, `update`,
`fwd_cpu`, `update_cpu`) can pull it in via the `_PKG_DIR` entry in
the `include_dirs=...` passed to `compile_and_load_variant` /
`compile_and_load_static`.
"""

from std.math import exp, recip


def _silu_f32(x: Float32) -> Float32:
    """SiLU activation: `x * sigmoid(x)`.

    Expressed as `x * recip(1 + exp(-x))` so the division lowers to
    a single fast reciprocal (`rcp.approx.ftz.f32` on nvptx, `v_rcp_f32`
    on amdgcn) + multiply rather than the full IEEE-compliant divide
    expansion (~12 instructions on amdgcn). The ~1 ulp accuracy loss
    on the reciprocal is well within the silu tolerance — all GPU
    tests pass with the same numerical bounds.
    """
    return x * recip(Float32(1) + exp(-x))
