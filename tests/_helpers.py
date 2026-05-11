"""Test helpers shared by `test_fwd.py`, `test_bwd.py`, `test_update.py`.

Tolerances are dominated by accumulator precision in the inner conv
(forward) and by the size of the (B, L) reduction (dweight, dbias).
bf16 has half the mantissa of fp16 — both forward and the per-element
backward (dx) get noticeably looser; the reduction tolerances scale
with B*L regardless of dtype but are roughly the same across fp16/bf16
in practice (the fp32 accumulators absorb most of it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


# Per-dtype tolerances reused by every test file.
_FWD_TOL = {torch.float16: 2e-2, torch.bfloat16: 2e-1, torch.float32: 1e-4}
_DX_TOL = {torch.float16: 1e-1, torch.bfloat16: 5e-1, torch.float32: 1e-3}
# dweight/dbias are sums over B*L terms — same fp32 accumulator on all
# paths, so the only delta is the cast back to the input dtype at the
# boundary. fp32 keeps the full accumulator so the tolerance collapses.
_DW_TOL = {torch.float16: 1.0, torch.bfloat16: 2.0, torch.float32: 1e-2}


def _make_bias(D, *, dtype, device, present, requires_grad=False):
    if not present:
        return None
    return torch.randn(D, dtype=dtype, device=device, requires_grad=requires_grad)


def _expected(x, weight, bias, activation):
    return causal_conv1d_ref(x, weight, bias=bias, activation=activation)


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _ref_grads(x, weight, bias, dout, activation):
    """Reference: pytorch impl of causal_conv1d_fwd, backward via autograd."""
    x_g = x.detach().requires_grad_()
    w_g = weight.detach().requires_grad_()
    b_g = bias.detach().requires_grad_() if bias is not None else None
    D, W = w_g.shape
    L = x_g.shape[-1]
    pre = F.conv1d(x_g, w_g.unsqueeze(1), b_g, padding=W - 1, groups=D)[..., :L]
    out = F.silu(pre) if activation in ("silu", "swish") else pre
    out.backward(dout)
    return x_g.grad, w_g.grad, (b_g.grad if b_g is not None else None)


def _expected_final_states(x, width):
    """Reference: last `width-1` columns of x with left zero-pad if
    seqlen < width-1. Same as `F.pad(x, (W-1-L, 0))[..., -W+1:]` from
    upstream's `causal_conv1d_ref`.
    """
    seqlen = x.shape[-1]
    pad_left = max(0, (width - 1) - seqlen)
    if pad_left > 0:
        return F.pad(x, (pad_left, 0))
    return x[..., -(width - 1) :].contiguous()


def _expected_with_initial_states(x, weight, bias, initial_states, activation):
    """Reference: pre-pend initial_states to x along seqlen, run conv with
    padding=0, slice back. Mirrors upstream's `causal_conv1d_ref` branch.
    """
    seqlen = x.shape[-1]
    D, W = weight.shape
    x_full = torch.cat([initial_states, x], dim=-1)
    pre = F.conv1d(x_full, weight.unsqueeze(1), bias, padding=0, groups=D)[..., :seqlen]
    return F.silu(pre) if activation in ("silu", "swish") else pre


def _ref_with_seq_idx(x, weight, bias, seq_idx, activation):
    """Reference for packed-sequence forward: each contiguous run of
    equal seq_idx values is treated as an independent sequence (the
    conv shouldn't read across boundaries). Padding (seq_idx < 0)
    rows output 0.
    """
    B, D, L = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        ids = seq_idx[b].cpu().numpy()
        start = 0
        while start < L:
            end = start + 1
            while end < L and ids[end] == ids[start]:
                end += 1
            run_id = int(ids[start])
            if run_id < 0:
                start = end
                continue
            seg = x[b : b + 1, :, start:end]
            seg_out = causal_conv1d_ref(seg, weight, bias=bias, activation=activation)
            out[b : b + 1, :, start:end] = seg_out
            start = end
    return out


def _ref_grads_with_seq_idx(x, weight, bias, seq_idx, dout, activation):
    """Reference gradients for the seq_idx forward: for each contiguous
    run of equal seq_idx in each batch row, run the standard
    causal_conv1d_ref on that segment with autograd; then accumulate
    grads. Padding rows (seq_idx < 0) contribute zero (their forward
    output was forced to 0 so dpre is 0).

    Returns (dx, dweight, dbias).
    """
    B, D, L = x.shape

    x_g = x.detach().clone().requires_grad_()
    w_g = weight.detach().clone().requires_grad_()
    b_g = bias.detach().clone().requires_grad_() if bias is not None else None

    for b in range(B):
        ids = seq_idx[b].cpu().numpy()
        start = 0
        while start < L:
            end = start + 1
            while end < L and ids[end] == ids[start]:
                end += 1
            run_id = int(ids[start])
            if run_id >= 0:
                seg_x = x_g[b : b + 1, :, start:end]
                seg_out = causal_conv1d_ref(seg_x, w_g, bias=b_g, activation=activation)
                seg_out.backward(dout[b : b + 1, :, start:end])
            start = end

    dx_ref = x_g.grad if x_g.grad is not None else torch.zeros_like(x)
    dw_ref = w_g.grad if w_g.grad is not None else torch.zeros_like(weight)
    db_ref = b_g.grad if (b_g is not None and b_g.grad is not None) else None
    return dx_ref, dw_ref, db_ref
