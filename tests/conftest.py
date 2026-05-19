import pytest
import torch


# Make every test deterministic. Failures near the tolerance threshold
# should be reproducible — without this, a flaky test would only fail
# under whatever RNG state pytest happened to leave behind from the
# previous test in the order.
@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


# Devices to run every test against. CPU is always available; CUDA
# and MPS are parametrised in only if the box has the matching
# accelerator. The Mojo GPU kernels are dispatched on Apple Metal via
# `DeviceContext`; the MPS path in `_fn.py` extracts each tensor's
# Metal-3.1 `gpuAddress` from its `MTLBuffer` so the kernels read /
# write torch's tensors directly (no host roundtrip). See `_mps.py`.
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
if torch.backends.mps.is_available():
    _DEVICES.append("mps")


@pytest.fixture(params=_DEVICES)
def device(request):
    return request.param


# Activations the public API accepts. silu and swish are the same op; None
# is the bias-only forward (no activation). Tests run all three.
@pytest.fixture(params=[None, "silu", "swish"])
def activation(request):
    return request.param


# `bias=None` is the bias-free forward. The kernel's `has_bias` comptime
# parameter selects the path.
@pytest.fixture(params=[True, False], ids=["with_bias", "no_bias"])
def bias_present(request):
    return request.param


# Dtypes supported by both the GPU and CPU paths. bf16 has only 7
# mantissa bits (vs fp16's 10), so reduction error on the backward pass
# is the loosest of the three; fp32 is the tightest.
@pytest.fixture(
    params=[torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def dtype(request):
    return request.param


# Width sweep used by the cross-cutting `test_width_*` tests + by the
# update tests (which test all three widths since `state_len >= W-1`).
@pytest.fixture(params=[2, 3, 4], ids=["w2", "w3", "w4"])
def width(request):
    return request.param


_DTYPE_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=5e-3, atol=5e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


def assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, msg: str = ""
) -> None:
    assert actual.shape == expected.shape, (
        f"{msg} shape: {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{msg} dtype: {actual.dtype} vs {expected.dtype}"
    )
    assert actual.device == expected.device, (
        f"{msg} device: {actual.device} vs {expected.device}"
    )
    tol = _DTYPE_TOL[actual.dtype]
    if not torch.allclose(actual.float(), expected.float(), **tol):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{msg} not close (max |diff|={max_abs:.3e}, tol={tol})")


def make_inputs(
    batch: int,
    dim: int,
    seqlen: int,
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    has_bias: bool,
    has_initial_states: bool,
    seed: int = 0,
) -> dict[str, torch.Tensor | None]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(batch, dim, seqlen, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    weight = torch.randn(dim, width, generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )
    bias = (
        torch.randn(dim, generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        if has_bias
        else None
    )
    initial_states = (
        torch.randn(batch, dim, width - 1, generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        if has_initial_states
        else None
    )
    return {"x": x, "weight": weight, "bias": bias, "initial_states": initial_states}
