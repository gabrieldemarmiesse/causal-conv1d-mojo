"""JSON-backed baseline cache for the GPU benches.

The mojo kernel is the moving target; the pytorch / torch.compile /
upstream comparisons are stable and re-measuring them every iteration
wastes ~minutes per bench. This module memo-izes their measurements to
`benchmarks/baselines/<bench>.json`, keyed by (impl, config, shape).

Usage:

    from _baseline import BaselineCache

    cache = BaselineCache(__file__)  # uses benchmarks/baselines/<bench>.json
    config = {"dtype": "fp16", "activation": "silu", "iters": 200, ...}

    # Mojo: always measure
    m = bench_kernel(lambda: ...)

    # Baselines: measure on miss, reuse on hit
    u = cache.get_or_run(
        impl="upstream", shape=(b, d, l, w), config=config,
        run=lambda: bench_kernel(lambda: upstream_fn(...)),
    )

The JSON layout is human-readable: a list of `{impl, shape, config,
value_us, gpu, timestamp}` records. Records are matched on (impl,
shape, sorted-config-tuple); a hit returns the cached `value_us`.

Use `BENCH_REFRESH=1` to force re-measurement of every baseline (the
cache file is then rewritten from scratch). Use `BENCH_REFRESH=impl1,impl2`
to refresh only specific impls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch


_BASELINE_DIR = Path(__file__).resolve().parent / "baselines"


def _refresh_set() -> set[str] | None:
    val = os.environ.get("BENCH_REFRESH", "").strip()
    if not val:
        return set()
    if val in ("1", "all", "*"):
        return None  # refresh everything
    return {x.strip() for x in val.split(",") if x.strip()}


def _config_key(config: dict[str, Any]) -> tuple:
    return tuple(sorted(config.items()))


class BaselineCache:
    def __init__(self, bench_path: str | os.PathLike) -> None:
        name = Path(bench_path).stem  # e.g. "plot_bench"
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _BASELINE_DIR / f"{name}.json"
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self._records: list[dict[str, Any]] = []
        self._index: dict[tuple, float] = {}
        self._refresh = _refresh_set()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._records = data.get("records", [])
            except json.JSONDecodeError:
                self._records = []
        self._reindex()

    def _reindex(self) -> None:
        self._index = {}
        for r in self._records:
            if r.get("gpu") != self.gpu:
                continue  # cached baselines from another GPU are not valid
            key = (r["impl"], tuple(r["shape"]), _config_key(r["config"]))
            self._index[key] = r["value_us"]

    def _should_refresh(self, impl: str) -> bool:
        if self._refresh is None:
            return True
        return impl in self._refresh

    def get_or_run(
        self,
        *,
        impl: str,
        shape: tuple,
        config: dict[str, Any],
        run: Callable[[], float],
    ) -> float:
        """Return cached μs for (impl, shape, config), or measure and cache."""
        key = (impl, tuple(shape), _config_key(config))
        if not self._should_refresh(impl) and key in self._index:
            return self._index[key]
        value = run()
        # Drop any stale record with the same key, then append.
        self._records = [
            r
            for r in self._records
            if not (
                r["impl"] == impl
                and tuple(r["shape"]) == tuple(shape)
                and _config_key(r["config"]) == _config_key(config)
                and r.get("gpu") == self.gpu
            )
        ]
        self._records.append(
            {
                "impl": impl,
                "shape": list(shape),
                "config": config,
                "value_us": value,
                "gpu": self.gpu,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        self._index[key] = value
        self.save()
        return value

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"records": self._records}, indent=2, sort_keys=True) + "\n"
        )
