"""Smoke-test that causal_conv1d_mojo works end-to-end on CUDA."""

from __future__ import annotations

import torch

from causal_conv1d_mojo import causal_conv1d_fn


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available inside the container"
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    batch, dim, seqlen, width = 2, 64, 128, 4
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float16)
    weight = torch.randn(dim, width, device=device, dtype=torch.float16)
    bias = torch.randn(dim, device=device, dtype=torch.float16)

    out = causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    torch.cuda.synchronize()

    assert out.shape == x.shape, f"unexpected shape {out.shape}"
    assert torch.isfinite(out).all(), "output has non-finite values"
    print(f"OK: out.shape={tuple(out.shape)} dtype={out.dtype}")


if __name__ == "__main__":
    main()
