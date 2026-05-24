"""Tests for the `CAUSAL_CONV1D_USE_CACHE_ONLY` env-var safety net.

When set, any cache miss inside `compile_and_load` raises instead of
silently triggering a JIT compile. Used in production deploys that
ship a pre-warmed cache directory — a miss there should be loud, not
a 1.2 s stall in the request hot path.
"""

import os
from pathlib import Path

import pytest
import torch

import causal_conv1d_mojo
from causal_conv1d_mojo._jit_common import compile_and_load


@pytest.fixture
def _restore_use_cache_only():
    """Toggle the env var on for a test and clean up afterwards."""
    prev = os.environ.get("CAUSAL_CONV1D_USE_CACHE_ONLY")
    yield
    if prev is None:
        os.environ.pop("CAUSAL_CONV1D_USE_CACHE_ONLY", None)
    else:
        os.environ["CAUSAL_CONV1D_USE_CACHE_ONLY"] = prev


def test_use_cache_only_raises_on_miss(tmp_path, monkeypatch, _restore_use_cache_only):
    """With the flag set and an empty cache, `compile_and_load` must
    raise immediately — no `mojo build` spawned."""
    # Redirect the cache root to an empty tmp dir so we get a guaranteed
    # miss without polluting the user's real cache.
    monkeypatch.setattr(
        "causal_conv1d_mojo._jit_common._CACHE_HOME", tmp_path / "cache"
    )
    monkeypatch.setenv("CAUSAL_CONV1D_USE_CACHE_ONLY", "1")

    # Use any real `variant.mojo` — we'll never get to actually building it.
    src = (
        Path(causal_conv1d_mojo.__file__).resolve().parent / "fwd_cpu" / "variant.mojo"
    )
    with pytest.raises(RuntimeError, match="CAUSAL_CONV1D_USE_CACHE_ONLY is set"):
        compile_and_load(
            subpkg="fwd_cpu",
            source_file=src,
            include_dirs=(src.parent, src.parent.parent),
            defines={
                "DTYPE": "float32",
                "WIDTH": "4",
                "HAS_BIAS": "true",
                "HAS_SEQ_IDX": "false",
                "HAS_INITIAL_STATES": "false",
                "APPLY_SILU": "false",
            },
            mod_name="probe_miss",
            backend="cpu",
        )


def test_use_cache_only_passes_on_hit(_restore_use_cache_only):
    """With the flag set but the variant already cached (from earlier
    in the test session), the call must succeed without recompiling.

    We warm the cache by calling `causal_conv1d_fn` first, then set the
    flag, then call again — the second call must return without spawning
    a compile.
    """
    # Warm up: ensure the relevant variant is compiled and cached.
    x = torch.randn(1, 4, 16, dtype=torch.float32)
    w = torch.randn(4, 4, dtype=torch.float32)
    causal_conv1d_mojo.causal_conv1d_fn(x, w, activation=None)

    # Now flip the flag and call again — should hit the cache cleanly.
    os.environ["CAUSAL_CONV1D_USE_CACHE_ONLY"] = "1"
    out = causal_conv1d_mojo.causal_conv1d_fn(x, w, activation=None)
    assert out.shape == x.shape
