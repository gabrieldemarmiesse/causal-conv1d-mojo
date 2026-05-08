"""Run the wall-time benches (forward + forward+backward) and emit
docs/bench_forward.png and docs/bench_backward.png.
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
WARMUP_FWD = 30
ITERS_FWD = 500
WARMUP_BWD = 20
ITERS_BWD = 200
DOCS = Path(__file__).resolve().parent.parent / "docs"


def pytorch_fwd(x, weight, bias):
    seqlen = x.shape[-1]
    D, W = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :seqlen]
    return F.silu(out)


def bench_wall(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1_000.0


def grouped_bar(labels, pt, up, mojo, *, title, out_path):
    n = len(labels)
    x_pos = list(range(n))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([p - bw for p in x_pos], pt, bw, label="pure PyTorch (F.conv1d + F.silu)", color="#bbbbbb")
    ax.bar(x_pos, up, bw, label="upstream (Tri Dao CUDA)", color="#3a78c2")
    ax.bar([p + bw for p in x_pos], mojo, bw, label="mojo (this repo)", color="#d05050")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("wall time per call (μs, lower is better)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    g = torch.Generator(device="cpu").manual_seed(0)
    gpu_name = torch.cuda.get_device_name(0)
    labels = [f"({b},{d},{l},{w})" for b, d, l, w in SHAPES]

    fwd_mojo, fwd_up, fwd_pt = [], [], []
    bwd_mojo, bwd_up, bwd_pt = [], [], []

    for b, d, l, w in SHAPES:
        # ------- forward only -------
        x = torch.randn(b, d, l, generator=g).to("cuda", torch.float16)
        weight = torch.randn(d, w, generator=g).to("cuda", torch.float16)
        bias = torch.randn(d, generator=g).to("cuda", torch.float16)
        kw = dict(bias=bias, activation="silu")
        m_f = bench_wall(lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw), WARMUP_FWD, ITERS_FWD)
        u_f = bench_wall(lambda: upstream_fn(x, weight, **kw), WARMUP_FWD, ITERS_FWD)
        p_f = bench_wall(lambda: pytorch_fwd(x, weight, bias), WARMUP_FWD, ITERS_FWD)
        fwd_mojo.append(m_f); fwd_up.append(u_f); fwd_pt.append(p_f)

        # ------- forward + backward -------
        dout = torch.randn(b, d, l, generator=g).to("cuda", torch.float16)

        def make_fwd_bwd(impl):
            def step():
                x_g = x.detach().requires_grad_()
                w_g = weight.detach().requires_grad_()
                b_g = bias.detach().requires_grad_()
                if impl == "mojo":
                    out = causal_conv1d_mojo.causal_conv1d_fn(x_g, w_g, bias=b_g, activation="silu")
                elif impl == "upstream":
                    out = upstream_fn(x_g, w_g, bias=b_g, activation="silu")
                else:
                    out = pytorch_fwd(x_g, w_g, b_g)
                out.backward(dout)
            return step

        m_b = bench_wall(make_fwd_bwd("mojo"), WARMUP_BWD, ITERS_BWD)
        u_b = bench_wall(make_fwd_bwd("upstream"), WARMUP_BWD, ITERS_BWD)
        p_b = bench_wall(make_fwd_bwd("pytorch"), WARMUP_BWD, ITERS_BWD)
        bwd_mojo.append(m_b); bwd_up.append(u_b); bwd_pt.append(p_b)

        print(
            f"{(b,d,l,w)!s:>22}  fwd: mojo={m_f:7.1f} up={u_f:7.1f} pt={p_f:7.1f} | "
            f"fwd+bwd: mojo={m_b:7.1f} up={u_b:7.1f} pt={p_b:7.1f}"
        )

    grouped_bar(
        labels, fwd_pt, fwd_up, fwd_mojo,
        title=(
            f"causal_conv1d FORWARD — {gpu_name}\n"
            f"fp16, bias, silu, {ITERS_FWD} iters, sync after each call"
        ),
        out_path=DOCS / "bench_forward.png",
    )
    grouped_bar(
        labels, bwd_pt, bwd_up, bwd_mojo,
        title=(
            f"causal_conv1d FORWARD + BACKWARD — {gpu_name}\n"
            f"fp16, bias, silu, {ITERS_BWD} iters, sync after each call"
        ),
        out_path=DOCS / "bench_backward.png",
    )


if __name__ == "__main__":
    main()
