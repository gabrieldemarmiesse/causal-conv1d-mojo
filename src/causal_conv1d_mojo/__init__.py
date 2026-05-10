"""causal_conv1d, fused into Mojo kernels and called via a direct
Python <-> Mojo CPython extension (no MAX framework).

Both forward and backward go through native Mojo kernels. On CUDA the
backward is a single fused kernel: dx + dweight + dbias accumulation
in one launch (mirrors upstream's `causal_conv1d_bwd_kernel`). On CPU
there's a parallel-over-(B,D) implementation that exists so the
package works on a GPU-less machine without users needing to
`pip install causal-conv1d` (which requires a C++ toolchain).
"""

from __future__ import annotations

import torch

# `mojo.importer` registers a Python import hook so that
#   from causal_conv1d_mojo._native import causal_conv1d_native
# triggers a one-time `mojo build --emit shared-lib` of the matching
# .mojo source on first import, caching the resulting .so under
# __mojocache__/. No manual `pixi run build-native` needed.
import mojo.importer  # noqa: F401  (registers the import hook)

from causal_conv1d_mojo._native import causal_conv1d_native as _native_mod


__version__ = "1.6.1"


# `bias` and `dbias_acc` may be None when the user omits bias; in that
# case we pass 0 for the data pointer. The Mojo kernels never
# dereference these pointers when the comptime `has_bias=False`.
def _ptr(t):
    return 0 if t is None else t.data_ptr()


# Must match the dispatch in the Mojo entry points.
_DTYPE_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}


def _native_fwd(x, weight, bias, seq_idx, initial_states, out, apply_silu):
    _native_mod.causal_conv1d_fwd(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        torch.cuda.current_stream().cuda_stream,
        int(seq_idx is not None),
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        weight.shape[1],
        int(initial_states is not None),
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
    )


def _native_bwd_full(x, weight, bias, dout, dx, dweight_acc, dbias_acc, apply_silu):
    _native_mod.causal_conv1d_bwd_full(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        dout.data_ptr(),
        dx.data_ptr(),
        dweight_acc.data_ptr(),
        _ptr(dbias_acc),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        torch.cuda.current_stream().cuda_stream,
        weight.shape[1],
    )


def _native_fwd_cpu(x, weight, bias, seq_idx, initial_states, out, apply_silu):
    _native_mod.causal_conv1d_fwd_cpu(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        int(seq_idx is not None),
        _ptr(seq_idx),
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        weight.shape[1],
        int(initial_states is not None),
        _ptr(initial_states),
        initial_states.stride(0) if initial_states is not None else 0,
        initial_states.stride(1) if initial_states is not None else 0,
        initial_states.stride(2) if initial_states is not None else 0,
    )


def _native_bwd_full_cpu(x, weight, bias, dout, dx, dweight_acc, dbias_acc, apply_silu):
    _native_mod.causal_conv1d_bwd_full_cpu(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        dout.data_ptr(),
        dx.data_ptr(),
        dweight_acc.data_ptr(),
        _ptr(dbias_acc),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        weight.shape[1],
    )


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
            _native_fwd(x, weight, bias, seq_idx, initial_states, out, apply_silu)
        else:
            _native_fwd_cpu(x, weight, bias, seq_idx, initial_states, out, apply_silu)
        if final_states_out is not None:
            _write_final_states(x, final_states_out, weight.shape[1])
        # `save_for_backward` accepts None — the slot just won't have a
        # tensor on retrieval.
        ctx.save_for_backward(x, weight, bias)
        ctx.apply_silu = apply_silu
        ctx.has_bias = bias is not None
        ctx.has_seq_idx = seq_idx is not None
        ctx.has_initial_states = initial_states is not None
        ctx.return_final_states = final_states_out is not None
        if final_states_out is not None:
            return out, final_states_out
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        # `grad_outputs` is `(dout,)` or `(dout, dfinal_states)` depending
        # on whether forward returned a tuple. dfinal_states is None when
        # the user never read .grad on final_states (no consumer).
        if ctx.has_seq_idx:
            # Forward seq_idx is implemented in the kernels; the chunked
            # bwd kernel's smem-halo dance hasn't been extended yet.
            # Inference (no .backward()) works fine; this only fires when
            # the user actually backprops through a seq_idx forward.
            raise NotImplementedError(
                "backward through seq_idx is not implemented yet — only the "
                "forward path supports seq_idx. Call the op under "
                "`torch.no_grad()` for inference."
            )
        if ctx.has_initial_states:
            # Forward initial_states is implemented in the kernels; the
            # bwd kernel doesn't yet read initial_states for the silu'
            # recomputation, doesn't accumulate the boundary dweight
            # contributions, and doesn't emit dinitial_states. Inference
            # works fine.
            raise NotImplementedError(
                "backward through initial_states is not implemented yet — "
                "only the forward path supports initial_states. Call the op "
                "under `torch.no_grad()` for inference."
            )
        dout = grad_outputs[0]
        dfinal_states = grad_outputs[1] if ctx.return_final_states else None
        x, weight, bias = ctx.saved_tensors
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

        if x.is_cuda:
            _native_bwd_full(
                x, weight, bias, dout, dx, dweight_acc, dbias_acc, apply_silu
            )
        else:
            _native_bwd_full_cpu(
                x, weight, bias, dout, dx, dweight_acc, dbias_acc, apply_silu
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
        # `seq_idx`, `initial_states`, `apply_silu`, and `final_states_out`
        # are non-tensor / non-diff inputs — autograd expects one return
        # per forward input, with `None` for non-differentiable inputs.
        return dx, dweight_acc.to(weight.dtype), dbias, None, None, None, None


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


def _native_update(
    x, weight, bias, conv_state, state_indices, cache_seqlens, out, apply_silu
):
    _native_mod.causal_conv1d_update(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        conv_state.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        conv_state.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        torch.cuda.current_stream().cuda_stream,
        weight.shape[1],
        int(state_indices is not None),
        _ptr(state_indices),
        int(cache_seqlens is not None),
        _ptr(cache_seqlens),
    )


def _native_update_cpu(
    x, weight, bias, conv_state, state_indices, cache_seqlens, out, apply_silu
):
    _native_mod.causal_conv1d_update_cpu(
        x.data_ptr(),
        weight.data_ptr(),
        _ptr(bias),
        conv_state.data_ptr(),
        out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        conv_state.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(bias is not None),
        int(apply_silu),
        _DTYPE_CODE[x.dtype],
        weight.shape[1],
        int(state_indices is not None),
        _ptr(state_indices),
        int(cache_seqlens is not None),
        _ptr(cache_seqlens),
    )


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
        _native_update(
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
        _native_update_cpu(
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
