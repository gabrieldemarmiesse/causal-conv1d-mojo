"""`causal_conv1d_update` — single-step / KV-cache decode API.

Dispatches to the GPU update kernel when `x.is_cuda`, otherwise to the
CPU update kernel. The rolling `conv_state` buffer is mutated in
place; there is no backward (decoding is inference-only).
"""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE
from causal_conv1d_mojo.reference import causal_conv1d_update_ref
from causal_conv1d_mojo.update import native_update, native_update_mps
from causal_conv1d_mojo.update_cpu import native_update_cpu

# MPS launch overhead dominates for small decode shapes — below this
# many elements (B*D*seqlen) the pure-PyTorch update_ref is faster than
# the Mojo update kernel on Apple GPUs. Empirically (Apple M4, fp16):
# mojo is ~0.35× ref at B*D=32K and ~3.4× at 128K. 64K is the crossover.
_MPS_UPDATE_FALLBACK_THRESHOLD = 64 * 1024


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
) -> torch.Tensor:
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
    # MPS small-shape fast path — bypass all validation so tiny decode
    # calls don't pay Python overhead on top of an already-cheap conv.
    # The Mojo update kernel only beats pure-PyTorch above B*D ~ 64K on
    # Apple GPUs. Ref handles both 2-D and 3-D x internally. Only safe
    # when conv_state_indices is None (ref doesn't support paged caches)
    # and dim > 0 (F.conv1d rejects groups=0).
    if x.device.type == "mps" and conv_state_indices is None:
        # x is (B, D) or (B, D, L); D and the element count are what
        # gate the fallback, and ref handles both ranks.
        x_shape = x.shape
        if len(x_shape) == 2:
            n_elts = x_shape[0] * x_shape[1]
            D = x_shape[1]
        else:
            n_elts = x_shape[0] * x_shape[1] * x_shape[2]
            D = x_shape[1]
        if D > 0 and 0 < n_elts < _MPS_UPDATE_FALLBACK_THRESHOLD:
            return causal_conv1d_update_ref(
                x,
                conv_state,
                weight,
                bias=bias,
                activation=activation,
                cache_seqlens=cache_seqlens,
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
    elif x.device.type == "mps":
        native_update_mps(
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
