from functools import lru_cache
from pathlib import Path

import torch
from max.experimental.torch import CustomOpLibrary


__version__ = "1.6.1"


_KERNEL_DIR = Path(__file__).parent / "kernels"
_op_library = CustomOpLibrary(_KERNEL_DIR)


@lru_cache(maxsize=None)
def _get_op(
    width: int,
    has_bias: bool,
    has_initial_states: bool,
    compute_final_states: bool,
    activation: str,
):
    return _op_library.causal_conv1d_fn[
        {
            "width": width,
            "has_bias": has_bias,
            "has_initial_states": has_initial_states,
            "compute_final_states": compute_final_states,
            "activation": activation,
        }
    ]


_placeholder_cache: dict[tuple, torch.Tensor] = {}


def _placeholder(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached zero tensor used as a kernel-arg placeholder.

    The Mojo kernel takes optional inputs/outputs as real tensors but
    `comptime if` branches them out when their feature flag is False.
    Caching avoids a per-call allocation + zero-init kernel launch.
    """
    key = (shape, dtype, device.type, device.index)
    t = _placeholder_cache.get(key)
    if t is None:
        t = torch.zeros(shape, dtype=dtype, device=device)
        _placeholder_cache[key] = t
    return t


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

    batch, dim, _ = x.shape
    _, width = weight.shape

    has_bias = bias is not None
    has_initial_states = initial_states is not None
    activation_kind = "silu" if activation in ("silu", "swish") else "none"

    bias_arg = bias if has_bias else _placeholder((1,), x.dtype, x.device)
    initial_states_arg = (
        initial_states
        if has_initial_states
        else _placeholder((1, 1, 1), x.dtype, x.device)
    )

    out = torch.empty_like(x)
    if return_final_states:
        final = (
            final_states_out
            if final_states_out is not None
            else torch.empty(batch, dim, width - 1, dtype=x.dtype, device=x.device)
        )
    else:
        final = _placeholder((1, 1, 1), x.dtype, x.device)

    op = _get_op(
        width, has_bias, has_initial_states, return_final_states, activation_kind
    )
    op(out, final, x, weight, bias_arg, initial_states_arg)

    if return_final_states:
        return out, final
    return out
