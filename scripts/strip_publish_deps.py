"""Strip dev/nvidia/rocm bits from pyproject.toml before publishing.

The optional `nvidia`/`rocm` extras and the `[tool.uv]` config exist only to
help local dev pick the right PyTorch index. They leak `nvidia-*` and
ROCm-specific wheels into the published package's metadata, so we drop
them before `uv build` runs in CI.

Rewrites pyproject.toml in place.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import tomli_w


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml")
    data = tomllib.loads(path.read_text())

    data.get("project", {}).pop("optional-dependencies", None)
    data.pop("dependency-groups", None)
    data.get("tool", {}).pop("uv", None)

    path.write_text(tomli_w.dumps(data))


if __name__ == "__main__":
    main()
