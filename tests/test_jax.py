"""End-to-end test: the public Python API accepts `jax.Array` inputs.

`causal_conv1d_fn` and `causal_conv1d_update` route jax inputs through
a zero-copy DLPack bridge to the torch implementation. These tests
verify that:

- Calling the public API with jax arrays returns jax arrays.
- The values match a parallel torch call on the same data.
- Mutable args (`final_states_out`, `conv_state`) are written through
  the shared DLPack buffer so the jax-visible memory sees the update.

Requires `jax` to be importable; on a torch-only environment the
whole module is skipped at collection time.
"""

from __future__ import annotations

import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import causal_conv1d_mojo  # noqa: E402


# jax exposes both CUDA and CPU; align with whichever the torch env
# has. The Mojo kernels go through the CUDA dispatch for CUDA arrays
# and through the CPU dispatch otherwise.
_HAS_CUDA = torch.cuda.is_available() and any(
    d.platform == "gpu" for d in jax.devices()
)
_JAX_DEVICES = ["cuda"] if _HAS_CUDA else ["cpu"]


@pytest.fixture(params=_JAX_DEVICES)
def jax_device(request):
    return request.param


_JAX_DTYPES = [
    (jnp.float16, torch.float16),
    (jnp.bfloat16, torch.bfloat16),
    (jnp.float32, torch.float32),
]


@pytest.fixture(
    params=_JAX_DTYPES, ids=["fp16", "bf16", "fp32"]
)
def dtype_pair(request):
    return request.param


def _torch_from_jax(arr):
    """torch tensor that shares storage with `arr` (zero-copy)."""
    return torch.utils.dlpack.from_dlpack(arr)


def _make_jax_inputs(*, batch, dim, seqlen, width, dtype_jax, device):
    """Create jax inputs on the given device, with deterministic values.

    We seed via numpy + `jax.device_put` rather than `jax.random` to
    keep the values bit-identical to a parallel torch run (the
    DLPack-bridged torch view shares the same bytes).
    """
    import numpy as np

    rng = np.random.default_rng(0)
    np_dtype = {jnp.float16: np.float16, jnp.bfloat16: np.float32, jnp.float32: np.float32}[dtype_jax]
    x_np = rng.standard_normal((batch, dim, seqlen)).astype(np_dtype)
    w_np = rng.standard_normal((dim, width)).astype(np_dtype)
    b_np = rng.standard_normal((dim,)).astype(np_dtype)
    target = jax.devices(device)[0]
    return {
        "x": jax.device_put(jnp.asarray(x_np, dtype=dtype_jax), target),
        "weight": jax.device_put(jnp.asarray(w_np, dtype=dtype_jax), target),
        "bias": jax.device_put(jnp.asarray(b_np, dtype=dtype_jax), target),
    }


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_fwd_jax_matches_torch(jax_device, dtype_pair, width, activation, with_bias):
    """`causal_conv1d_fn` on jax inputs matches a parallel torch call."""
    dtype_jax, dtype_torch = dtype_pair
    B, D, L = 2, 64, 128
    inputs_jax = _make_jax_inputs(
        batch=B, dim=D, seqlen=L, width=width, dtype_jax=dtype_jax, device=jax_device
    )
    bias_jax = inputs_jax["bias"] if with_bias else None

    out_jax = causal_conv1d_mojo.causal_conv1d_fn(
        inputs_jax["x"], inputs_jax["weight"], bias=bias_jax, activation=activation
    )

    # Sanity: output is a jax.Array on the same device as the input.
    assert isinstance(out_jax, jax.Array), f"expected jax.Array, got {type(out_jax)}"
    assert out_jax.dtype == dtype_jax
    assert out_jax.shape == (B, D, L)

    # Torch reference: build identical tensors (zero-copy DLPack view of
    # the same jax buffer) and run the torch path of the same public API.
    x_t = _torch_from_jax(inputs_jax["x"])
    w_t = _torch_from_jax(inputs_jax["weight"])
    b_t = _torch_from_jax(inputs_jax["bias"]) if with_bias else None
    out_t = causal_conv1d_mojo.causal_conv1d_fn(
        x_t, w_t, bias=b_t, activation=activation
    )

    # Bridge the jax result back to torch for value comparison.
    out_jax_as_t = _torch_from_jax(out_jax)
    assert out_jax_as_t.dtype == dtype_torch
    # Identical inputs + identical kernel + same dtype ⇒ bit-equal output.
    # (No reductions, only deterministic FMAs.) We compare via `equal`
    # to catch any silent layout/striding bug the bridge could introduce.
    assert torch.equal(out_jax_as_t, out_t), (
        f"jax-bridge output differs from torch output "
        f"(max |diff|={(out_jax_as_t.float() - out_t.float()).abs().max().item():.3e})"
    )


def test_fwd_jax_return_final_states(jax_device):
    """`return_final_states=True` returns a (out, final_states) jax tuple."""
    B, D, L, W = 2, 32, 64, 4
    dtype_jax, _ = _JAX_DTYPES[0]  # fp16
    inputs = _make_jax_inputs(
        batch=B, dim=D, seqlen=L, width=W, dtype_jax=dtype_jax, device=jax_device
    )
    out, final_states = causal_conv1d_mojo.causal_conv1d_fn(
        inputs["x"],
        inputs["weight"],
        bias=inputs["bias"],
        activation="silu",
        return_final_states=True,
    )
    assert isinstance(out, jax.Array)
    assert isinstance(final_states, jax.Array)
    assert out.shape == (B, D, L)
    assert final_states.shape == (B, D, W - 1)
    # final_states should be the last (W-1) cols of x (the public API
    # writes that directly; no kernel involvement).
    x_tail = inputs["x"][..., -(W - 1) :]
    assert jnp.array_equal(final_states, x_tail), (
        f"final_states != x[..., -W+1:] "
        f"(max |diff|={jnp.abs(final_states.astype(jnp.float32) - x_tail.astype(jnp.float32)).max():.3e})"
    )


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_update_jax_matches_torch(jax_device, width, activation):
    """`causal_conv1d_update` on jax inputs matches a parallel torch call.

    Also exercises the in-place mutation of `conv_state`: the DLPack
    view shares storage with the jax array, so after the call the
    jax-visible `conv_state` reflects the kernel's writes.
    """
    import numpy as np

    B, D = 4, 64
    state_len = width - 1
    dtype_jax = jnp.float16
    rng = np.random.default_rng(0)
    np_dtype = np.float16
    target = jax.devices(jax_device)[0]
    x = jax.device_put(
        jnp.asarray(rng.standard_normal((B, D)).astype(np_dtype)), target
    )
    state = jax.device_put(
        jnp.asarray(rng.standard_normal((B, D, state_len)).astype(np_dtype)), target
    )
    w = jax.device_put(
        jnp.asarray(rng.standard_normal((D, width)).astype(np_dtype)), target
    )
    b = jax.device_put(
        jnp.asarray(rng.standard_normal((D,)).astype(np_dtype)), target
    )

    # Snapshot the pre-update state so the torch reference run has the
    # same starting buffer (the jax call below will mutate `state` in
    # place via the DLPack bridge).
    state_t_ref = _torch_from_jax(state).clone()
    x_t = _torch_from_jax(x).clone()
    w_t = _torch_from_jax(w).clone()
    b_t = _torch_from_jax(b).clone()

    out_jax = causal_conv1d_mojo.causal_conv1d_update(
        x, state, w, bias=b, activation=activation
    )
    assert isinstance(out_jax, jax.Array)
    assert out_jax.shape == (B, D)

    out_t = causal_conv1d_mojo.causal_conv1d_update(
        x_t, state_t_ref, w_t, bias=b_t, activation=activation
    )

    # Output matches.
    assert torch.equal(_torch_from_jax(out_jax), out_t)
    # State mutation matches (the DLPack bridge mutated the shared
    # buffer, so the jax-visible `state` now equals torch's updated
    # state_t_ref).
    assert torch.equal(_torch_from_jax(state), state_t_ref)


def test_mixed_jax_torch_inputs_route_via_bridge(jax_device):
    """Even one jax input is enough to trigger the dlpack bridge.

    Documents the current contract: any jax input puts the call on
    the jax path. Mixed inputs are converted uniformly (torch tensors
    are also DLPack-bridgeable, so the recursion stays well-defined).
    """
    B, D, L, W = 1, 16, 32, 3
    inputs = _make_jax_inputs(
        batch=B, dim=D, seqlen=L, width=W, dtype_jax=jnp.float32, device=jax_device
    )
    # weight as torch, x as jax: the bridge should still produce a jax output.
    w_t = _torch_from_jax(inputs["weight"]).clone()
    out = causal_conv1d_mojo.causal_conv1d_fn(
        inputs["x"], w_t, bias=inputs["bias"], activation=None
    )
    assert isinstance(out, jax.Array)
    assert out.shape == (B, D, L)
