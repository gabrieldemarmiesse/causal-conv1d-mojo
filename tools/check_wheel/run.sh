#!/usr/bin/env bash
# Build the causal_conv1d_mojo wheel inside a container and smoke-test it on CUDA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

docker build -t causal-conv1d-mojo -f tools/check_wheel/Dockerfile .
docker run --rm --gpus all causal-conv1d-mojo
