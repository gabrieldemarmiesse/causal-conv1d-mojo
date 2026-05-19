# syntax=docker/dockerfile:1.7

# ---------- Stage 1: build the wheel ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS builder

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv build --wheel --out-dir /out

# ---------- Stage 2: runtime venv + smoke test ----------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:${PATH}" \
    UV_LINK_MODE=copy

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.13 \
 && uv venv --python 3.13 /opt/venv

COPY --from=builder /out/*.whl /tmp/
COPY tools/check_wheel/check_cuda.py /app/check_cuda.py

RUN uv pip install --python /opt/venv/bin/python \
        /tmp/causal_conv1d_mojo-*.whl \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.8.0" \
        "nvidia-cuda-nvcc-cu12==12.8.*"

WORKDIR /app
ENTRYPOINT ["python", "check_cuda.py"]
