"""`flash_attn_func` — the public forward+backward API.

Mirrors upstream `flash_attn.flash_attn_func` (the v2.x API). The
autograd op dispatches to the GPU kernels (`fwd` + `bwd` subpackages)
when `q.is_cuda`; CPU fallback uses `flash_attn_ref` (pure-PyTorch
SDPA).

STATUS: scaffolding only. The Mojo kernels are stubbed out and raise
`NotImplementedError`. The infrastructure around them (autograd
Function, torch.library.custom_op registration, fake-tensor metadata)
is in place so the kernel work, when added, slots in without further
refactoring.
"""

from __future__ import annotations

import torch

from flash_attn_mojo.reference import flash_attn_ref


# Sentinel for the "no window" case in flash-attn 2's sliding-window
# parameter — `window_size=(-1, -1)` means full attention.
_NO_WINDOW = (-1, -1)


def _fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward dispatch. Returns (out, lse).

    Current kernel limitations (the simplest viable initial impl):
    only fp16, only head_dim=64, no causal, no dropout, no alibi,
    no softcap, no window, no MQA/GQA. Anything outside that envelope
    raises NotImplementedError so callers see a clear error rather
    than silently-wrong results.
    """
    from flash_attn_mojo.fwd import native_fwd

    # The mha_single_batch port goes through `linalg.matmul.gpu.multistage_mma`,
    # which derives its MMA shape via `get_mma_shape[input, accum]`. For input=bf16
    # the chosen shape is m16n8k16 (one PTX `mma.sync` instruction with the largest
    # K we get on Ampere/Ada); for input=fp16 the published Mojo stdlib at the
    # version we pin (mojo-compiler 1.0.0b1) only ships m16n8k8, so the multi-stage
    # gemm fails to instantiate. We could route fp16 through a hand-rolled m16n8k8
    # gemm but that defeats the point of using `multistage_mma`. So this entry
    # point is bf16-only for now; an fp16 path lands once Mojo gains m16n8k16 fp16
    # in the public stdlib (it's already in MAX's `tensor_core.get_mma_shape`).
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "flash_attn_mojo current kernel supports bf16 only "
            f"(got {q.dtype}). See `_fn.py` for the why."
        )
    if q.shape[-1] != 64:
        raise NotImplementedError(
            f"flash_attn_mojo current kernel supports head_dim=64 only "
            f"(got {q.shape[-1]})."
        )
    if causal:
        raise NotImplementedError("flash_attn_mojo: causal not yet implemented.")
    if dropout_p != 0.0:
        raise NotImplementedError("flash_attn_mojo: dropout not yet implemented.")
    if softcap != 0.0:
        raise NotImplementedError("flash_attn_mojo: softcap not yet implemented.")
    if window_size != _NO_WINDOW:
        raise NotImplementedError("flash_attn_mojo: window_size not yet implemented.")
    if alibi_slopes is not None:
        raise NotImplementedError("flash_attn_mojo: alibi_slopes not yet implemented.")
    if q.shape[2] != k.shape[2]:
        raise NotImplementedError(
            "flash_attn_mojo: MQA/GQA (nheads_q != nheads_kv) not yet implemented."
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out = torch.empty_like(q)
    # Empty lse for now — backward isn't wired up, this is just a
    # placeholder so the autograd Function's `save_for_backward`
    # signature stays stable.
    lse = torch.empty(
        q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device
    )

    native_fwd(q, k, v, out, softmax_scale)
    return out, lse


def _bwd_dispatch(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward dispatch. Returns (dq, dk, dv).

    TODO: replace with the Mojo kernel once `bwd/` is implemented.
    """
    raise NotImplementedError(
        "flash_attn_mojo: GPU backward kernel not yet implemented."
    )


class _FlashAttnFn(torch.autograd.Function):
    """fp16/bf16 autograd op for full (non-varlen) attention.

    Matches upstream's `_flash_attn_func` autograd.Function semantics.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        return_attn_probs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _fwd_dispatch(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, deterministic,
        )
        ctx.save_for_backward(q, k, v, out, lse, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        if return_attn_probs:
            # Upstream also exposes the softmax denominator and (with
            # dropout) the RNG mask. We return `lse` and `None` for the
            # RNG slot until dropout is implemented.
            return out, lse, None
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        q, k, v, out, lse, alibi_slopes = ctx.saved_tensors
        dq, dk, dv = _bwd_dispatch(
            dout, q, k, v, out, lse,
            ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            ctx.window_size, ctx.softcap, alibi_slopes, ctx.deterministic,
        )
        # forward arg order: q, k, v, dropout_p, softmax_scale, causal,
        # window_size, softcap, alibi_slopes, deterministic,
        # return_attn_probs. Returns map 1:1 with None for
        # non-differentiable inputs.
        return dq, dk, dv, None, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-head scaled-dot-product attention with Flash Attention's
    block-tiled algorithm.

    q, k, v: (batch, seqlen, nheads, headdim). Note: nheads_kv may differ
        from nheads_q (multi-query/grouped-query attention) — k and v
        share the same nheads_kv.
    dropout_p: dropout probability on the attention matrix.
    softmax_scale: scale applied before softmax. Defaults to
        `1 / sqrt(headdim)`.
    causal: if True, apply lower-triangular causal mask.
    window_size: `(left, right)` sliding-window mask, both in tokens.
        `(-1, -1)` = no window (the default). With causal=True, only
        the `left` value matters.
    softcap: if > 0, apply `softcap * tanh(scores / softcap)` for
        attention-softcap (Gemma 2 / Grok). 0 disables.
    alibi_slopes: (nheads,) or (batch, nheads) ALiBi slopes.
    deterministic: if True, force the deterministic (slower) backward.
    return_attn_probs: if True, return `(out, softmax_lse, rng_state)`
        — needed for debugging or for stacking attention layers.

    Returns: out of shape (batch, seqlen, nheads, headdim).
    """
    if q.device.type != "cuda":
        # No Mojo CPU kernel yet — fall back to the pure-PyTorch
        # reference for CPU inputs.
        return flash_attn_ref(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
    result = _FlashAttnFn.apply(
        q, k, v, dropout_p, softmax_scale, causal, window_size,
        softcap, alibi_slopes, deterministic, return_attn_probs,
    )
    return result
