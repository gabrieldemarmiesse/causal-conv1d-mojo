"""`causal_conv1d_update` — single-step / KV-cache decode API.

Dispatches to the GPU update kernel when `x.is_cuda`, otherwise to the
CPU update kernel. The rolling `conv_state` buffer is mutated in
place; there is no backward (decoding is inference-only).
"""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE
from causal_conv1d_mojo.update import native_update, native_update_mps
from causal_conv1d_mojo.update_cpu import native_update_cpu


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

    Accepts either `torch.Tensor` or `jax.Array` inputs. jax arrays
    are bridged to torch views via DLPack (zero-copy), the torch
    impl runs (mutating `conv_state` in place — the mutation is
    visible to jax through the shared buffer), and `out` is
    converted back to jax.
    """
    from causal_conv1d_mojo._jax_bridge import any_jax, jax_to_torch, torch_to_jax

    if any_jax(x, conv_state, weight, bias, cache_seqlens, conv_state_indices):
        # See `_fn.py`'s causal_conv1d_fn jax bridge for the rationale —
        # torch tensors pass through unchanged so callers can mix
        # backends. `conv_state` is the in-place arg; the DLPack view
        # shares storage with the jax array, so the kernel's writes
        # are visible to jax through the same buffer.
        from causal_conv1d_mojo._jax_bridge import is_jax_array

        def _to_t(a):
            if a is None or not is_jax_array(a):
                return a
            return jax_to_torch(a)

        out_t = causal_conv1d_update(
            _to_t(x),
            _to_t(conv_state),
            _to_t(weight),
            _to_t(bias),
            activation,
            _to_t(cache_seqlens),
            _to_t(conv_state_indices),
        )
        return torch_to_jax(out_t)

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
