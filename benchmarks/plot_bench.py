"""Run the wall-time bench and emit docs/bench.png.

Same workload + paths as bench_vs_pytorch.py; plots a grouped bar chart.
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import causal_conv1d_mojo
from causal_conv1d import causal_conv1d_fn as upstream_fn


SHAPES = [
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 8192, 4),
    (1, 4096, 2048, 4),
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]
WARMUP = 30
ITERS = 500
OUT = Path(__file__).resolve().parent.parent / "docs" / "bench.png"


def call_pytorch(x, weight, bias):
    seqlen = x.shape[-1]
    D, W = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :seqlen]
    return F.silu(out)


def bench_wall(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)

    rows: list[tuple[str, list[float]]] = []
    labels = [f"({b},{d},{l},{w})" for b, d, l, w in SHAPES]
    mojo_t, up_t, pt_t = [], [], []

    for b, d, l, w in SHAPES:
        x = torch.randn(b, d, l, generator=g).to(device=device, dtype=torch.float16)
        weight = torch.randn(d, w, generator=g).to(device=device, dtype=torch.float16)
        bias = torch.randn(d, generator=g).to(device=device, dtype=torch.float16)
        kw = dict(bias=bias, activation="silu")

        m = bench_wall(lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw))
        u = bench_wall(lambda: upstream_fn(x, weight, **kw))
        p = bench_wall(lambda: call_pytorch(x, weight, bias))
        mojo_t.append(m); up_t.append(u); pt_t.append(p)
        print(f"{(b,d,l,w)!s:>22}  mojo={m:7.1f}us  upstream={u:7.1f}us  pytorch={p:7.1f}us")

    # Grouped bar chart, log-scale (PyTorch can be ~10x off on the heaviest shape).
    n = len(SHAPES)
    x_pos = list(range(n))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([p - width for p in x_pos], pt_t, width, label="pure PyTorch (F.conv1d + F.silu)", color="#bbbbbb")
    ax.bar(x_pos, up_t, width, label="upstream (Tri Dao CUDA)", color="#3a78c2")
    ax.bar([p + width for p in x_pos], mojo_t, width, label="mojo (this repo)", color="#d05050")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("wall time per call (μs, lower is better)")
    ax.set_yscale("log")
    ax.set_title(
        f"causal_conv1d forward — {torch.cuda.get_device_name(0)}\n"
        f"fp16, bias, silu, {ITERS} iters, sync after each call"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
