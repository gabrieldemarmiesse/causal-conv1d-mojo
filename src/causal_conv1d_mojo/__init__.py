from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from max.experimental.torch import CustomOpLibrary


__version__ = "1.6.1"


_KERNEL_DIR = Path(__file__).parent / "kernels"
_op_library = CustomOpLibrary(_KERNEL_DIR)


@lru_cache(maxsize=None)
def _get_op(width: int, has_bias: bool, has_initial_states: bool, activation: str):
    return _op_library.causal_conv1d_fn[
        {
            "width": width,
            "has_bias": has_bias,
            "has_initial_states": has_initial_states,
            "activation": activation,
        }
    ]


def _compute_final_states(x, initial_states, width: int):
    extended = (
        torch.cat([initial_states, x], dim=-1) if initial_states is not None else x
    )
    return F.pad(extended, (width - 1 - extended.shape[-1], 0))


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
        raise NotImplementedError("activation must be None, silu, or swish")
    if seq_idx is not None:
        raise NotImplementedError("seq_idx is not supported")

    _, width = weight.shape
    has_bias = bias is not None
    has_initial_states = initial_states is not None
    activation_kind = "silu" if activation in ("silu", "swish") else "none"

    bias_arg = bias if has_bias else torch.zeros(1, dtype=x.dtype, device=x.device)
    initial_states_arg = (
        initial_states
        if has_initial_states
        else torch.zeros(1, 1, 1, dtype=x.dtype, device=x.device)
    )

    out = torch.empty_like(x)
    op = _get_op(width, has_bias, has_initial_states, activation_kind)
    op(out, x, weight, bias_arg, initial_states_arg)

    if not return_final_states:
        return out

    final = _compute_final_states(x, initial_states, width)
    if final_states_out is not None:
        final_states_out.copy_(final)
        final = final_states_out
    return out, final
