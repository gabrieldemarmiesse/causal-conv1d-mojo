"""causal_conv1d, fused into Mojo kernels and called via direct
Python <-> Mojo CPython extensions (no MAX framework).

Both forward and backward go through native Mojo kernels. On CUDA the
backward is a single fused kernel: dx + dweight + dbias accumulation
in one launch (mirrors upstream's `causal_conv1d_bwd_kernel`). On CPU
there's a parallel-over-(B,D) implementation that exists so the
package works on a GPU-less machine without users needing to
`pip install causal-conv1d` (which requires a C++ toolchain).

Layout: each of the six Python entry points lives in its own
subpackage (`fwd/`, `bwd_full/`, `fwd_cpu/`, `bwd_full_cpu/`,
`update/`, `update_cpu/`). Every subpackage bundles its Mojo kernel,
Mojo dispatcher, and Python wrapper. First-time use of one of the
public APIs lazily imports — and therefore lazily compiles via
`mojo.importer` — only the subpackages it needs, instead of paying
for all six dispatch trees upfront.
"""

from __future__ import annotations

import torch

# `mojo.importer` registers a Python import hook so that
#   from causal_conv1d_mojo.<subpkg> import dispatch
# triggers a one-time `mojo build --emit shared-lib` of the matching
# .mojo source on first import, caching the resulting .so under
# `<subpkg>/__mojocache__/`. No manual build step needed.
import mojo.importer  # noqa: F401  (registers the import hook)

from causal_conv1d_mojo._dtype import _DTYPE_CODE
from causal_conv1d_mojo.bwd_full import native_bwd_full
from causal_conv1d_mojo.bwd_full_cpu import native_bwd_full_cpu
from causal_conv1d_mojo.fwd import native_fwd
from causal_conv1d_mojo.fwd_cpu import native_fwd_cpu
from causal_conv1d_mojo.update import native_update
from causal_conv1d_mojo.update_cpu import native_update_cpu


__version__ = "1.6.1"


def _write_final_states(x, final_states_out, width):
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
    """fp16/bf16/fp32, width=4 autograd op (CUDA + CPU).

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
        ctx, x, weight, bias, seq_idx, initial_states, apply_silu, final_states_out
    ):
        out = torch.empty_like(x)
        if x.is_cuda:
            native_fwd(x, weight, bias, seq_idx, initial_states, out, apply_silu)
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
    def backward(ctx, *grad_outputs):
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
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
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
    if weight.shape[1] not in (2, 3, 4):
        raise NotImplementedError(
            f"only width in {{2, 3, 4}} is supported (got {weight.shape[1]})"
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


# ===---------- causal_conv1d_update (single-step / KV-cache decode) ----------=== #


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
    conv_state_indices=None,
):
    """Single-step (or short-burst) causal conv1d update for autoregressive
    decoding.

    x: (batch, dim) or (batch, dim, seqlen)  -- the new tokens
    conv_state: (batch_or_pool_size, dim, state_len), state_len >= width - 1
        Mutated in place. Default mode: oldest `seqlen` values are
        dropped, new x values are appended on the right. Circular mode
        (cache_seqlens != None): writes happen at `cache_seqlens[b]`
        with wrap-around modulo `state_len`.
    weight: (dim, width)
    bias: (dim,) or None
    activation: None | "silu" | "swish"
    cache_seqlens: (batch,) int32 or None. When set, conv_state is
        treated as a circular buffer; cache_seqlens[b] is the per-batch
        write head (only its value mod state_len matters). The kernel
        does NOT advance cache_seqlens; the caller does.
    conv_state_indices: (batch,) int32 or None. When set, the conv state
        for batch element `b` lives at row `conv_state_indices[b]` of
        `conv_state` (decoupling input batch from cache slot — used by
        paged-cache servers). A negative index marks a padding token:
        the output for that batch is zeroed and the state row is left
        untouched. cache_seqlens is still indexed by `b`, not the
        redirected coord (matching upstream).

    Returns: out tensor with the same shape as `x`.
    """
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
    if conv_state.dtype != x.dtype:
        raise NotImplementedError(
            f"conv_state.dtype ({conv_state.dtype}) must match x.dtype ({x.dtype})"
        )
    if weight.shape[1] not in (2, 3, 4):
        raise NotImplementedError(
            f"only width in {{2, 3, 4}} is supported (got {weight.shape[1]})"
        )

    # Match upstream's calling convention: x can be 2-D (no seqlen
    # dimension) for the common single-token-per-call decode path. We
    # unsqueeze internally and squeeze at the end.
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    # With conv_state_indices, conv_state.shape[0] is a *pool size*, not
    # `batch`. Without it, the two must match.
    if conv_state_indices is None:
        if conv_state.shape != (batch, dim, state_len):
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)} != expected "
                f"{(batch, dim, state_len)}"
            )
    else:
        if conv_state.shape[1] != dim or conv_state.shape[2] != state_len:
            raise ValueError(
                f"conv_state shape {tuple(conv_state.shape)}: expected "
                f"(*, {dim}, {state_len})"
            )
    if state_len < width - 1:
        raise ValueError(
            f"conv_state.shape[-1]={state_len} must be >= width-1={width - 1}"
        )
    if (
        x.device != weight.device
        or x.device != conv_state.device
        or (bias is not None and x.device != bias.device)
    ):
        raise NotImplementedError(
            "x, weight, bias, conv_state must all be on the same device"
        )

    if cache_seqlens is not None:
        if cache_seqlens.shape != (batch,):
            raise ValueError(
                f"cache_seqlens shape {tuple(cache_seqlens.shape)} != "
                f"expected {(batch,)}"
            )
        if cache_seqlens.dtype != torch.int32:
            raise ValueError(
                f"cache_seqlens.dtype must be int32 (got {cache_seqlens.dtype})"
            )
        if cache_seqlens.device != x.device:
            raise ValueError(
                f"cache_seqlens.device ({cache_seqlens.device}) must match "
                f"x.device ({x.device})"
            )
        if not cache_seqlens.is_contiguous():
            cache_seqlens = cache_seqlens.contiguous()

    if conv_state_indices is not None:
        if conv_state_indices.shape != (batch,):
            raise ValueError(
                f"conv_state_indices shape {tuple(conv_state_indices.shape)} "
                f"!= expected {(batch,)}"
            )
        if conv_state_indices.dtype != torch.int32:
            raise ValueError(
                f"conv_state_indices.dtype must be int32 "
                f"(got {conv_state_indices.dtype})"
            )
        if conv_state_indices.device != x.device:
            raise ValueError(
                f"conv_state_indices.device ({conv_state_indices.device}) "
                f"must match x.device ({x.device})"
            )
        if not conv_state_indices.is_contiguous():
            conv_state_indices = conv_state_indices.contiguous()

    out = torch.empty_like(x)
    apply_silu = activation in ("silu", "swish")

    if x.is_cuda:
        native_update(
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            cache_seqlens,
            out,
            apply_silu,
        )
    else:
        native_update_cpu(
            x,
            weight,
            bias,
            conv_state,
            conv_state_indices,
            cache_seqlens,
            out,
            apply_silu,
        )

    if unsqueeze:
        out = out.squeeze(-1)
    return out
