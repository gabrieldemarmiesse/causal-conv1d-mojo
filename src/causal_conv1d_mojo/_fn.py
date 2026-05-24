"""`causal_conv1d_fn` — the public forward+backward API.

`causal_conv1d_fn` is the user-facing entry point that wraps the
`_CausalConv1dFn` autograd.Function. The autograd op dispatches to the
GPU kernels (`fwd` + `bwd_full` subpackages) when `x.is_cuda` and to
the CPU kernels otherwise; `bias` may be None on either path.

`final_states_out`, when provided, is written in-place with the last
`W-1` cols of `x` (left zero-padded if `seqlen < W-1`); the backward
adds `dfinal_states` into the matching slice of `dx`.
"""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE
from causal_conv1d_mojo.bwd_full import native_bwd_full, native_bwd_full_mps
from causal_conv1d_mojo.bwd_full_cpu import native_bwd_full_cpu
from causal_conv1d_mojo.fwd import native_fwd, native_fwd_mps
from causal_conv1d_mojo.fwd_cpu import native_fwd_cpu
from causal_conv1d_mojo.reference import causal_conv1d_ref

# MPS launch overhead dominates the kernel for small shapes — below this
# many elements (B*D*L) the pure-PyTorch path (F.conv1d) is faster on
# Apple GPUs. Empirically (Apple M4, fp16): mojo is ~0.3× pyTorch at
# B*D*L=2M and ~1.4× at 8M. 4M is the conservative crossover.
_MPS_FWD_FALLBACK_THRESHOLD = 4 * 1024 * 1024


def _write_final_states(
    x: torch.Tensor, final_states_out: torch.Tensor, width: int
) -> None:
    """`final_states_out[b, c, i]` is the value of `x[b, c, t]` at the
    `width-1` most-recent positions, left zero-padded if `seqlen <
    width-1`. Used by the chunked / stateful execution path: feed
    `final_states_out` of chunk `i` as `initial_states` of chunk `i+1`.

    Mirrors upstream's `F.pad(x, (W-1-seqlen, 0))[..., -W+1:]` slice in
    `causal_conv1d_ref`.
    """
    seqlen = x.shape[-1]
    pad_left = (width - 1) - seqlen
    if pad_left > 0:
        # x is shorter than W-1: copy all of x into the right portion
        # and zero the left.
        final_states_out[..., :pad_left].zero_()
        final_states_out[..., pad_left:].copy_(x)
    else:
        final_states_out.copy_(x[..., -(width - 1) :])


class _CausalConv1dFn(torch.autograd.Function):
    """fp16/bf16/fp32 autograd op (CUDA + CPU).

    Widths: 2..9 for fp16/bf16, 2..5 for fp32; backward additionally
    requires seqlen aligned to 1024 when width > 5.

    Dispatches to the GPU launcher when `x.is_cuda`, otherwise to the
    pure-mojo CPU launcher (parallelized over (B, D) via
    `sync_parallelize`). `apply_silu` and the bias-presence flag are
    plumbed through both paths; `bias` may be None.

    `final_states_out`, if provided, is written to in-place with the
    last `W-1` cols of `x` (left zero-padded if seqlen < W-1). The
    backward adds `dfinal_states` into the corresponding slice of
    `dx`.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        seq_idx: torch.Tensor | None,
        initial_states: torch.Tensor | None,
        apply_silu: bool,
        final_states_out: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        if x.is_cuda:
            native_fwd(x, weight, bias, seq_idx, initial_states, out, apply_silu)
        elif x.device.type == "mps":
            native_fwd_mps(x, weight, bias, seq_idx, initial_states, out, apply_silu)
        else:
            native_fwd_cpu(x, weight, bias, seq_idx, initial_states, out, apply_silu)
        if final_states_out is not None:
            _write_final_states(x, final_states_out, weight.shape[1])
        # `save_for_backward` accepts None — the slot just won't have a
        # tensor on retrieval. seq_idx and initial_states are
        # non-differentiable inputs (well, initial_states *can* be
        # differentiable; we save it either way and decide in backward).
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.apply_silu = apply_silu
        ctx.has_bias = bias is not None
        ctx.return_final_states = final_states_out is not None
        if final_states_out is not None:
            return out, final_states_out
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        # `grad_outputs` is `(dout,)` or `(dout, dfinal_states)` depending
        # on whether forward returned a tuple. dfinal_states is None when
        # the user never read .grad on final_states (no consumer).
        dout = grad_outputs[0]
        dfinal_states = grad_outputs[1] if ctx.return_final_states else None
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        apply_silu = ctx.apply_silu
        has_bias = ctx.has_bias
        D, W = weight.shape
        seqlen = x.shape[-1]

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        dx = torch.empty_like(x)
        # Per-block dweight/dbias contributions are atomic-added in fp32
        # to avoid losing mantissa bits across batches. dbias_acc only
        # allocated when there's a bias to differentiate.
        dweight_acc = torch.zeros(D, W, dtype=torch.float32, device=x.device)
        dbias_acc = (
            torch.zeros(D, dtype=torch.float32, device=x.device) if has_bias else None
        )
        # Always allocate dinitial_states when initial_states is set —
        # the kernel writes it unconditionally to keep the dispatch lean
        # (one comptime flag instead of two). We only return it when the
        # user actually wants the gradient, but the kernel populates it
        # in place either way.
        dinitial_states = (
            torch.empty_like(initial_states) if initial_states is not None else None
        )

        if x.is_cuda:
            native_bwd_full(
                x,
                weight,
                bias,
                dout,
                seq_idx,
                initial_states,
                dx,
                dweight_acc,
                dbias_acc,
                dinitial_states,
                apply_silu,
            )
        elif x.device.type == "mps":
            native_bwd_full_mps(
                x,
                weight,
                bias,
                dout,
                seq_idx,
                initial_states,
                dx,
                dweight_acc,
                dbias_acc,
                dinitial_states,
                apply_silu,
            )
        else:
            native_bwd_full_cpu(
                x,
                weight,
                bias,
                dout,
                seq_idx,
                initial_states,
                dx,
                dweight_acc,
                dbias_acc,
                dinitial_states,
                apply_silu,
            )

        if dfinal_states is not None:
            # final_states[b, c, i] = x[b, c, seqlen - (W-1) + i] for
            # i s.t. that index is in-range; the rest is zero-padded
            # and contributes no gradient. So dx gets the matching
            # tail incremented.
            tail = min(W - 1, seqlen)
            if tail > 0:
                dx[..., -tail:] += dfinal_states[..., -tail:].to(dx.dtype)

        dbias = dbias_acc.to(bias.dtype) if has_bias else None
        # Forward input order: (x, weight, bias, seq_idx, initial_states,
        # apply_silu, final_states_out). Returns map 1:1; `seq_idx`,
        # `apply_silu`, `final_states_out` are non-differentiable so we
        # return None. `initial_states` gets dinitial_states when it was
        # provided (the kernel always populated it; autograd will only
        # use it if init.requires_grad).
        return (
            dx,
            dweight_acc.to(weight.dtype),
            dbias,
            None,
            dinitial_states,
            None,
            None,
        )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    # MPS small-shape fast path — bypass all validation so tiny calls
    # don't pay ~9μs of Python checks on top of an already-cheap conv.
    # The Mojo kernel beats pure-PyTorch only at B*D*L >= ~4M on Apple
    # GPUs, so below that we route straight to causal_conv1d_ref.
    # `causal_conv1d_ref` and the F.conv1d it calls do their own input
    # validation, so malformed inputs still error (with less specific
    # messages than the wrapper would give). Only safe when seq_idx is
    # None (ref doesn't support packed sequences) and dim > 0 (F.conv1d
    # rejects groups=0).
    if x.device.type == "mps" and seq_idx is None:
        B, D, L = x.shape
        if D > 0 and 0 < B * D * L < _MPS_FWD_FALLBACK_THRESHOLD:
            return causal_conv1d_ref(
                x,
                weight,
                bias=bias,
                initial_states=initial_states,
                return_final_states=return_final_states,
                final_states_out=final_states_out,
                activation=activation,
            )

    if activation not in (None, "silu", "swish"):
        raise NotImplementedError(
            "only activation in {None, 'silu', 'swish'} is supported"
        )
    if x.dtype not in _DTYPE_CODE:
        raise NotImplementedError(
            f"unsupported dtype {x.dtype}; only fp16/bf16/fp32 are supported"
        )
    if weight.dtype != x.dtype:
        raise NotImplementedError(
            f"weight.dtype ({weight.dtype}) must match x.dtype ({x.dtype})"
        )
    if bias is not None and bias.dtype != x.dtype:
        raise NotImplementedError(
            f"bias.dtype ({bias.dtype}) must match x.dtype ({x.dtype})"
        )
    # Width support is dtype- and seqlen-dependent because the kernels
    # share boundary x values across threads via a smem ring buffer
    # holding `kNElts` slots — the (W-1) halo must fit into one slot.
    # kNElts is 8 for fp16/bf16 and 4 for fp32 (16 bytes / dtype). The
    # bwd kernel additionally drops to kNElts=4 on the unaligned tail
    # path (seqlen % 1024 != 0), so width >5 on the unaligned path
    # would corrupt the dx halo accumulation. Conservative limit:
    # widths 2..9 for fp16/bf16, 2..5 for fp32, and require seqlen
    # aligned to 1024 elements when width > 5. Wider widths on fp32 or
    # the bwd's unaligned path need a redesigned halo dance — open an
    # issue if you need them.
    width = weight.shape[1]
    max_width = 5 if x.dtype == torch.float32 else 9
    if width < 2 or width > max_width:
        dtype_clause = "" if x.dtype == torch.float32 else " for fp32"
        raise NotImplementedError(
            f"width must be in 2..{max_width}{dtype_clause} (got {width}). "
            f"Open an issue at "
            f"https://github.com/gabrieldemarmiesse/causal-conv1d-mojo/issues "
            f"if you need wider."
        )
    if width > 5 and x.requires_grad:
        if x.shape[-1] % 1024 != 0:
            raise NotImplementedError(
                f"width > 5 with autograd requires seqlen aligned to 1024 "
                f"(got seqlen={x.shape[-1]}, width={width}). The bwd halo "
                f"falls back to a 4-slot ring on unaligned seqlens; widths "
                f"6..9 read past its boundary. Open an issue if you need "
                f"this combination."
            )
    if x.device != weight.device or (bias is not None and x.device != bias.device):
        raise NotImplementedError(
            f"x, weight, bias must all be on the same device "
            f"(got x={x.device}, weight={weight.device}, "
            f"bias={'None' if bias is None else bias.device})"
        )

    if final_states_out is not None and not return_final_states:
        raise ValueError(
            "final_states_out is only meaningful when return_final_states=True"
        )

    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # seq_idx + return_final_states are mutually exclusive (matches
    # upstream): with packed sequences in one batch row, "the last W-1
    # cols" doesn't have a single owning sequence.
    # seq_idx + initial_states are also mutually exclusive: per-position
    # masking and a shared "before t=0" context aren't compatible.
    if seq_idx is not None:
        if return_final_states:
            raise ValueError(
                "seq_idx and return_final_states are mutually exclusive "
                "(packed sequences have no single 'last W-1 cols')"
            )
        if initial_states is not None:
            raise ValueError("seq_idx and initial_states are mutually exclusive")
        if seq_idx.shape != (batch, seqlen):
            raise ValueError(
                f"seq_idx shape {tuple(seq_idx.shape)} != expected {(batch, seqlen)}"
            )
        if seq_idx.dtype != torch.int32:
            raise ValueError(f"seq_idx.dtype must be int32 (got {seq_idx.dtype})")
        if seq_idx.device != x.device:
            raise ValueError(
                f"seq_idx.device ({seq_idx.device}) must match x.device ({x.device})"
            )
        if not seq_idx.is_contiguous():
            seq_idx = seq_idx.contiguous()

    if initial_states is not None:
        if initial_states.shape != (batch, dim, width - 1):
            raise ValueError(
                f"initial_states shape {tuple(initial_states.shape)} != "
                f"expected {(batch, dim, width - 1)}"
            )
        if initial_states.dtype != x.dtype:
            raise ValueError(
                f"initial_states.dtype ({initial_states.dtype}) must match "
                f"x.dtype ({x.dtype})"
            )
        if initial_states.device != x.device:
            raise ValueError(
                f"initial_states.device ({initial_states.device}) must "
                f"match x.device ({x.device})"
            )
        # Inner-axis stride must be unit so the kernel's
        # `is_idx * initial_states_l_stride` indexing reads contiguous
        # memory; outer axes can be any stride (handled by the
        # batch/channel base).
        if initial_states.stride(2) != 1:
            initial_states = initial_states.contiguous()

    if return_final_states:
        if final_states_out is None:
            final_states_out = torch.empty(
                batch, dim, width - 1, dtype=x.dtype, device=x.device
            )
        else:
            if final_states_out.shape != (batch, dim, width - 1):
                raise ValueError(
                    f"final_states_out shape {tuple(final_states_out.shape)} "
                    f"!= expected {(batch, dim, width - 1)}"
                )
            if final_states_out.dtype != x.dtype:
                raise ValueError(
                    f"final_states_out.dtype ({final_states_out.dtype}) "
                    f"must match x.dtype ({x.dtype})"
                )
            if final_states_out.device != x.device:
                raise ValueError(
                    f"final_states_out.device ({final_states_out.device}) "
                    f"must match x.device ({x.device})"
                )

    # silu and swish are the same function (x * sigmoid(x)); activation=None
    # is the bias-only path.
    apply_silu = activation in ("silu", "swish")
    result = _CausalConv1dFn.apply(
        x, weight, bias, seq_idx, initial_states, apply_silu, final_states_out
    )
    if return_final_states:
        # _CausalConv1dFn returns (out, final_states) when final_states_out
        # is non-None.
        return result
    return result
