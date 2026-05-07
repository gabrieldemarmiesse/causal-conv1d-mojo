import pytest
import torch

from causal_conv1d_mojo import causal_conv1d_fn

from conftest import assert_close, make_inputs


# Reference: pure-PyTorch implementation that ships with the upstream package.
# It's mathematically identical to causal_conv1d.causal_conv1d_fn (the CUDA op);
# we use it as the ground truth on CPU since the CUDA op only runs on GPU.
from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref


SHAPES = [
    pytest.param(1, 4, 1, 4, id="b1_d4_l1_w4"),
    pytest.param(1, 4, 8, 4, id="b1_d4_l8_w4"),
    pytest.param(2, 8, 16, 3, id="b2_d8_l16_w3"),
    pytest.param(3, 16, 32, 2, id="b3_d16_l32_w2"),
    pytest.param(2, 4, 3, 4, id="b2_d4_l3_w4_short"),
    pytest.param(1, 32, 64, 4, id="b1_d32_l64_w4"),
]

DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]

ACTIVATIONS = [None, "silu", "swish"]


def _devices() -> list[pytest.param]:
    devs = [pytest.param(torch.device("cpu"), id="cpu")]
    if torch.cuda.is_available():
        devs.append(pytest.param(torch.device("cuda"), id="cuda"))
    return devs


@pytest.fixture(params=_devices())
def device(request):
    return request.param


@pytest.mark.parametrize("batch,dim,seqlen,width", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("has_bias", [False, True])
def test_matches_reference_basic(batch, dim, seqlen, width, dtype, activation, has_bias, device):
    inputs = make_inputs(
        batch, dim, seqlen, width,
        dtype=dtype, device=device, has_bias=has_bias, has_initial_states=False,
    )
    out = causal_conv1d_fn(
        inputs["x"], inputs["weight"], bias=inputs["bias"], activation=activation,
    )
    expected = causal_conv1d_ref(
        inputs["x"], inputs["weight"], bias=inputs["bias"], activation=activation,
    )
    assert_close(out, expected)


@pytest.mark.parametrize("batch,dim,seqlen,width", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", [None, "silu"])
def test_matches_reference_with_initial_states(batch, dim, seqlen, width, dtype, activation, device):
    inputs = make_inputs(
        batch, dim, seqlen, width,
        dtype=dtype, device=device, has_bias=True, has_initial_states=True,
    )
    out = causal_conv1d_fn(
        inputs["x"], inputs["weight"], bias=inputs["bias"],
        initial_states=inputs["initial_states"], activation=activation,
    )
    expected = causal_conv1d_ref(
        inputs["x"], inputs["weight"], bias=inputs["bias"],
        initial_states=inputs["initial_states"], activation=activation,
    )
    assert_close(out, expected)


@pytest.mark.parametrize("batch,dim,seqlen,width", [(2, 8, 16, 4), (1, 4, 1, 3)])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_initial_states", [False, True])
def test_return_final_states(batch, dim, seqlen, width, dtype, has_initial_states, device):
    inputs = make_inputs(
        batch, dim, seqlen, width,
        dtype=dtype, device=device, has_bias=True, has_initial_states=has_initial_states,
    )
    out, final = causal_conv1d_fn(
        inputs["x"], inputs["weight"], bias=inputs["bias"],
        initial_states=inputs["initial_states"], return_final_states=True,
    )
    expected_out, expected_final = causal_conv1d_ref(
        inputs["x"], inputs["weight"], bias=inputs["bias"],
        initial_states=inputs["initial_states"], return_final_states=True,
    )
    assert_close(out, expected_out, msg="out")
    assert_close(final, expected_final, msg="final_states")
    assert final.shape == (batch, dim, width - 1)


def test_final_states_out_is_filled_in_place(device):
    batch, dim, seqlen, width = 2, 8, 16, 4
    inputs = make_inputs(
        batch, dim, seqlen, width,
        dtype=torch.float32, device=device, has_bias=False, has_initial_states=False,
    )
    final_buf = torch.empty(batch, dim, width - 1, dtype=torch.float32, device=device)
    final_buf_id = final_buf.data_ptr()
    final_buf.fill_(float("nan"))
    out, final = causal_conv1d_fn(
        inputs["x"], inputs["weight"],
        return_final_states=True, final_states_out=final_buf,
    )
    assert final.data_ptr() == final_buf_id, "final_states_out should be filled in place"
    assert torch.isfinite(final).all(), "final_states_out still contains NaN -- not written"


def test_silu_equals_swish(device):
    inputs = make_inputs(
        2, 8, 16, 4,
        dtype=torch.float32, device=device, has_bias=True, has_initial_states=False,
    )
    out_silu = causal_conv1d_fn(inputs["x"], inputs["weight"], bias=inputs["bias"], activation="silu")
    out_swish = causal_conv1d_fn(inputs["x"], inputs["weight"], bias=inputs["bias"], activation="swish")
    assert_close(out_silu, out_swish)


def test_invalid_activation_raises(device):
    inputs = make_inputs(
        1, 4, 8, 4,
        dtype=torch.float32, device=device, has_bias=False, has_initial_states=False,
    )
    with pytest.raises(NotImplementedError):
        causal_conv1d_fn(inputs["x"], inputs["weight"], activation="relu")


def test_seq_idx_raises(device):
    inputs = make_inputs(
        1, 4, 8, 4,
        dtype=torch.float32, device=device, has_bias=False, has_initial_states=False,
    )
    seq_idx = torch.zeros(1, 8, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        causal_conv1d_fn(inputs["x"], inputs["weight"], seq_idx=seq_idx)


def test_output_dtype_preserved(device):
    inputs = make_inputs(
        1, 4, 8, 4,
        dtype=torch.float16, device=device, has_bias=True, has_initial_states=False,
    )
    out = causal_conv1d_fn(inputs["x"], inputs["weight"], bias=inputs["bias"])
    assert out.dtype == torch.float16
    assert out.shape == inputs["x"].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reference (causal_conv1d_fn) requires GPU")
@pytest.mark.parametrize("batch,dim,seqlen,width", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("has_bias", [False, True])
def test_matches_cuda_reference(batch, dim, seqlen, width, dtype, activation, has_bias):
    """On GPU we can compare against the actual upstream causal_conv1d_fn (CUDA op)."""
    from causal_conv1d import causal_conv1d_fn as upstream_fn

    device = torch.device("cuda")
    inputs = make_inputs(
        batch, dim, seqlen, width,
        dtype=dtype, device=device, has_bias=has_bias, has_initial_states=False,
    )
    out = causal_conv1d_fn(
        inputs["x"], inputs["weight"], bias=inputs["bias"], activation=activation,
    )
    expected = upstream_fn(
        inputs["x"], inputs["weight"], bias=inputs["bias"], activation=activation,
    )
    assert_close(out, expected)
