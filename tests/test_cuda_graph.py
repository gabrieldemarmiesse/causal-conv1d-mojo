"""CUDA graph capture compatibility for the Mojo kernels.

The production pattern this enables: warm the JIT cache once, set
`CAUSAL_CONV1D_USE_CACHE_ONLY=1` to forbid further compiles, then
capture a CUDA graph around the request hot path and replay on every
incoming batch.

Empirically (CUDA 12.x + recent PyTorch), capture is permissive
enough that even a cold-cache JIT compile *inside* capture doesn't
error — `cudaStreamBeginCapture` with `cudaStreamCaptureModeThreadLocal`
ignores ops that don't land on the capture stream, and mojo's
compile path (subprocess fork, fs I/O, `dlopen`, `cuModuleLoadDataEx`)
fits that exemption. The captured graph still works correctly on
replay. But that's not a property to rely on:

- A 1.2 s compile during capture defeats the point of capture (you
  adopted it to *avoid* host-side stalls).
- CUDA's stream-capture restrictions have been tightened repeatedly
  across driver releases — a future driver could reject module
  loading during capture.
- A cache miss for a never-warmed shape silently slows your first
  response on that shape by ~1.2 s with no log telling you why.

So `CAUSAL_CONV1D_USE_CACHE_ONLY` is a **performance + predictability**
guard, not a strict correctness guard. These tests cover both
properties: capture works at all (correctness), and the
warmup+CACHE_ONLY+capture flow is a clean production story (no
silent compiles after the warmup window).
"""

import os

import pytest
import torch

import causal_conv1d_mojo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda graph capture needs cuda"
)


def test_cuda_graph_capture_and_replay_matches_eager():
    """Capture a graph around causal_conv1d_fn, replay it, and verify
    the buffered output is bit-identical to an eager re-run on the
    same inputs."""
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out_buf = torch.empty_like(x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # `.copy_` into a pre-allocated buffer so replays write to a
        # stable address — that's how graphs surface results.
        out_buf.copy_(
            causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
        )

    g.replay()
    torch.cuda.synchronize()

    eager = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
    assert torch.equal(out_buf, eager)


def test_cuda_graph_replay_reflects_input_mutations():
    """Replaying the graph after mutating `x` in place must produce
    the new correct output — captured kernels read from the original
    addresses, so in-place updates propagate through replays."""
    B, D, L, W = 1, 32, 64, 3
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    out_buf = torch.empty_like(x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf.copy_(
            causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
        )

    # First replay against the original x.
    g.replay()
    torch.cuda.synchronize()
    eager_v1 = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
    assert torch.equal(out_buf, eager_v1)

    # Mutate x in place — same storage, different values.
    x.add_(0.5)
    g.replay()
    torch.cuda.synchronize()
    eager_v2 = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
    assert torch.equal(out_buf, eager_v2)


def test_cuda_graph_works_with_use_cache_only():
    """The production deploy pattern end-to-end: pre-warm the cache,
    set CAUSAL_CONV1D_USE_CACHE_ONLY=1 to forbid further compiles, then
    capture + replay. The pre-warmup is the only step that must happen
    with the flag unset — once locked, capture and replay run entirely
    against cached `.so`s."""
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, W, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    # Pre-warm with the flag unset so the variant gets compiled.
    assert "CAUSAL_CONV1D_USE_CACHE_ONLY" not in os.environ
    _ = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
    torch.cuda.synchronize()

    # Now lock the cache. Any further attempt to JIT-compile would
    # raise — including inside `graph(...)` if capture happened to
    # encounter an un-warmed variant.
    os.environ["CAUSAL_CONV1D_USE_CACHE_ONLY"] = "1"
    try:
        out_buf = torch.empty_like(x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf.copy_(
                causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
            )
        g.replay()
        torch.cuda.synchronize()

        eager = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
        assert torch.equal(out_buf, eager)
    finally:
        os.environ.pop("CAUSAL_CONV1D_USE_CACHE_ONLY", None)
