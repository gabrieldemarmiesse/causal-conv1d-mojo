from functools import lru_cache
from pathlib import Path

import torch
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

    batch, dim, seqlen = x.shape
    _, width = weight.shape

    has_bias = bias is not None
    has_initial_states = initial_states is not None
    activation_kind = "silu" if activation in ("silu", "swish") else "none"

    out = torch.empty_like(x)
    if return_final_states and final_states_out is not None:
        final_states = final_states_out
    else:
        final_states = torch.empty(
            batch, dim, width - 1, dtype=x.dtype, device=x.device
        )

    # Mojo kernel always takes bias/initial_states tensors; when absent, the
    # kernel branches them out at compile time and never reads them, but MAX
    # still requires real (non-empty) tensors at the API level.
    bias_arg = bias if has_bias else torch.zeros(1, dtype=x.dtype, device=x.device)
    initial_states_arg = (
        initial_states
        if has_initial_states
        else torch.zeros(1, 1, 1, dtype=x.dtype, device=x.device)
    )

    op = _get_op(width, has_bias, has_initial_states, activation_kind)
    op(out, final_states, x, weight, bias_arg, initial_states_arg)

    if return_final_states:
        return out, final_states
    return out
