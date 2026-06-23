"""JSON-backed baseline cache shared by the GPU benches.

The mojo kernel is the moving target; the upstream / pytorch /
torch.compile comparisons are stable on a pinned baseline with locked
clocks, so we measure them once and reuse — re-measuring every iteration
wastes ~minutes per bench. This module memo-izes those measurements to
`scripts/baselines/<name>.json`.

Two consumers, one store:

- `bench.py` (the unified driver) caches full *run-distributions* via
  `get_runs()` / `put_runs()`, keying each record on a per-impl version
  tag (so an upstream wheel bump invalidates only upstream) plus an env
  signature that folds in the GPU name and clock-lock state (so unlocking
  the GPU or moving hardware auto-invalidates — same discipline as the
  JIT cache):

      cache = BaselineCache(
          path=_BASELINE_DIR / f"{fn}_{measure}.json",
          env_sig=env_sig, tags={"upstream": wheel_ver, "pytorch": "ref"},
      )
      runs = cache.get_runs("upstream", shape, config)      # list | None
      cache.put_runs("upstream", shape, config, runs_us)

- `plot_bench.py` caches a single representative μs per (impl, shape,
  config) via the pull-based `get_or_run()`, keyed only on the GPU name
  and refreshable through `BENCH_REFRESH`:

      cache = BaselineCache.for_plot(__file__)   # scripts/baselines/<stem>.json
      u = cache.get_or_run(impl="upstream", shape=(b, d, l, w),
                           config=cfg, run=lambda: bench_kernel(...))

The JSON layout is a list of `{key, runs_us}` records; a hit matches on
the full key. `BENCH_REFRESH=1` (or `all`/`*`) forces re-measurement of
every baseline; `BENCH_REFRESH=impl1,impl2` refreshes only those impls.
Records are machine + clock specific, so `scripts/baselines/` is
.gitignored.
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
    """Parse ``BENCH_REFRESH``: ``None`` = refresh all, set = named impls,
    empty set = refresh nothing (the default)."""
    val = os.environ.get("BENCH_REFRESH", "").strip()
    if not val:
        return set()
    if val in ("1", "all", "*"):
        return None
    return {x.strip() for x in val.split(",") if x.strip()}


def _jsonable(key: tuple) -> tuple:
    """Round-trip a key through JSON-native types so cached and live keys
    compare equal (JSON turns tuples into lists, ints stay ints)."""
    return tuple(json.loads(json.dumps(list(key))))


class BaselineCache:
    """Memoizes stable baseline (upstream/pytorch) measurements to JSON.

    A record is keyed on ``(tag, impl, shape, config, env_sig)`` where
    ``tag`` is the per-impl version (``tags[impl]``, ``"n/a"`` if absent)
    and ``env_sig`` is a dict of global discriminators folded into every
    key. Both default to empty, so a bare ``BaselineCache(path=...)``
    keys only on ``(impl, shape, config)``.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike,
        env_sig: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        refresh: set[str] | None = frozenset(),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env_sig = dict(env_sig or {})
        self.tags = dict(tags or {})
        self._refresh = refresh
        self._records: list[dict[str, Any]] = []
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text()).get("records", [])
            except json.JSONDecodeError:
                self._records = []

    @classmethod
    def for_plot(cls, bench_path: str | os.PathLike) -> "BaselineCache":
        """GPU-keyed cache for ``plot_bench.py``: file named ``<stem>.json``,
        keyed on the GPU name, refreshable via ``BENCH_REFRESH``."""
        name = Path(bench_path).stem  # e.g. "plot_bench"
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return cls(
            path=_BASELINE_DIR / f"{name}.json",
            env_sig={"gpu": gpu},
            refresh=_refresh_set(),
        )

    def _key(self, impl: str, shape: tuple, config: dict) -> tuple:
        return (
            self.tags.get(impl, "n/a"),
            impl,
            tuple(shape),
            tuple(sorted(config.items())),
            tuple(sorted(self.env_sig.items())),
        )

    def _should_refresh(self, impl: str) -> bool:
        if self._refresh is None:
            return True
        return impl in self._refresh

    def get_runs(self, impl: str, shape: tuple, config: dict) -> list[float] | None:
        """Cached run-distribution for (impl, shape, config), or None on
        a miss or when this impl is being refreshed."""
        if self._should_refresh(impl):
            return None
        want = _jsonable(self._key(impl, shape, config))
        for r in self._records:
            if tuple(r.get("key", ())) == want:
                return list(r.get("runs_us", []))
        return None

    def put_runs(
        self, impl: str, shape: tuple, config: dict, runs_us: list[float]
    ) -> None:
        key = _jsonable(self._key(impl, shape, config))
        self._records = [r for r in self._records if tuple(r.get("key", ())) != key]
        self._records.append(
            {
                "key": list(key),
                "runs_us": list(runs_us),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        self.save()

    def get_or_run(
        self,
        *,
        impl: str,
        shape: tuple,
        config: dict[str, Any],
        run: Callable[[], float],
    ) -> float:
        """Return cached μs for (impl, shape, config), or measure and cache.

        ``run`` returns a single representative μs; it is stored as a
        one-element run-distribution and the cached value is its minimum.
        """
        hit = self.get_runs(impl, shape, config)
        if hit:
            return min(hit)
        value = run()
        self.put_runs(impl, shape, config, [value])
        return value

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"records": self._records}, indent=2, sort_keys=True) + "\n"
        )
