import torch


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
