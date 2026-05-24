"""Pytest fixtures + beartype claw setup."""

# Runtime type checking under test. `beartype_this_package` would only
# cover *this* conftest module, so we explicitly target the package
# under test with `beartype_package`. This must happen *before*
# `flash_attn_mojo` (or any of its submodules) is imported — the claw
# installs a sys.meta_path hook that rewrites every function in the
# package at import time.
from beartype.claw import beartype_package  # noqa: I001

beartype_package("flash_attn_mojo")

import pytest  # noqa: E402
import torch  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_rng():
    """Every test gets the same RNG state — failures near tolerance
    boundaries stay reproducible."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
if torch.backends.mps.is_available():
    _DEVICES.append("mps")


@pytest.fixture(params=_DEVICES)
def device(request):
    return request.param


@pytest.fixture(
    params=[torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def dtype(request):
    return request.param


_DTYPE_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=5e-3, atol=5e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, msg: str = "") -> None:
    assert actual.shape == expected.shape, (
        f"{msg} shape: {actual.shape} vs {expected.shape}"
    )
    assert actual.dtype == expected.dtype, (
        f"{msg} dtype: {actual.dtype} vs {expected.dtype}"
    )
    tol = _DTYPE_TOL[actual.dtype]
    if not torch.allclose(actual.float(), expected.float(), **tol):
        max_abs = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{msg} not close (max |diff|={max_abs:.3e}, tol={tol})")
