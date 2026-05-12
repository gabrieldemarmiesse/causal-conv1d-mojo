"""Run the per-kernel GPU-time benches (forward, forward+backward,
single-step update) and emit docs/bench_forward.png,
docs/bench_backward.png, and docs/bench_update.png.

Each impl is timed inside `torch.profiler` via CUPTI: warmup runs
outside the profiler, then ITERS calls inside it, and we sum
`self_device_time_total` across every CUDA event recorded for that
impl's runs. This excludes Python overhead + cudaLaunchKernel + sync
round-trip — the floor that dominated the old `time.perf_counter_ns()`
wall-clock measurement at small shapes.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import causal_conv1d_mojo
from causal_conv1d_mojo.reference import causal_conv1d_ref, causal_conv1d_update_ref
from _baseline import BaselineCache

# pixi run -e bench ...
from causal_conv1d import causal_conv1d_fn as upstream_fn
from causal_conv1d import causal_conv1d_update as upstream_update_fn

SHAPES = [
    # Tiny / low-occupancy: kChunkSize=1024 (fp16) so L<=1024 → 1 chunk,
    # and B*D blocks → most grids don't fill the SMs. This is the
    # regime where launch overhead matters in practice (short prefills).
    (1, 256, 64, 4),
    (1, 1024, 64, 4),
    (1, 1024, 128, 4),
    (1, 1024, 256, 4),
    # Mid: 1-block-per-(B,D) grid still fits the GPU comfortably.
    (1, 1024, 512, 4),
    (1, 1024, 2048, 4),
    (1, 1024, 8192, 4),
    (1, 4096, 2048, 4),
    # Large: fully GPU-bound.
    (4, 4096, 2048, 4),
    (8, 2048, 4096, 4),
]

# Update op: per-call (B, D) with seqlen=1 (one-token-at-a-time decode).
# state_len = W-1 = 3. These are typical Mamba decode shapes; per-call
# kernel time is what matters since the user runs this every token.
UPDATE_SHAPES = [
    # Tiny decode shapes (e.g. single-user inference, small models).
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 1024),
    (4, 2048),
    (4, 4096),
    (16, 2048),
    (32, 4096),
]
WARMUP_FWD = 30
ITERS_FWD = 200
WARMUP_BWD = 20
ITERS_BWD = 100
WARMUP_UPDATE = 50
ITERS_UPDATE = 500

# CPU bench: same shape grid would take minutes per call at the top
# end, so use a smaller grid (matches benchmarks/bench_cpu.py) and
# fewer iters. CPU calls are synchronous, no profiler needed — wall
# time around the call is the kernel time.
CPU_SHAPES = [
    (1, 256, 256, 4),
    (1, 256, 1024, 4),
    (1, 1024, 256, 4),
    (1, 1024, 1024, 4),
    (1, 1024, 2048, 4),
    (1, 2048, 1024, 4),
    (1, 2048, 2048, 4),
    (4, 1024, 1024, 4),
    (4, 2048, 1024, 4),
    (8, 1024, 1024, 4),
]
WARMUP_FWD_CPU = 5
ITERS_FWD_CPU = 50
WARMUP_BWD_CPU = 5
ITERS_BWD_CPU = 25
WARMUP_UPDATE_CPU = 20
ITERS_UPDATE_CPU = 100
DOCS = Path(__file__).resolve().parent.parent / "docs"


def pytorch_fwd(x, weight, bias):
    seqlen = x.shape[-1]
    D, W = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[..., :seqlen]
    return F.silu(out)


# torch.compile'd reference. inductor specializes per shape on first call;
# warmup inside bench_kernel hides the compile cost. We compile the
# functions once at module load — the dynamo cache handles shape
# specialization automatically across the SHAPES loop.
#
# Bump dynamo's recompile_limit: each new (B,D,L) and each requires_grad
# toggle counts as a recompile, and the fwd+bwd path adds a grad-enabled
# specialization per shape. Default 8 falls back to eager partway
# through the bench, which silently makes the last shapes match pure
# PyTorch exactly. 64 is comfortably above our shape × grad count.
torch._dynamo.config.recompile_limit = 64
pytorch_fwd_compiled = torch.compile(pytorch_fwd)
update_ref_compiled = torch.compile(causal_conv1d_update_ref)


def bench_kernel(fn, warmup: int, iters: int) -> float:
    """Mean GPU time per call, μs, via torch.profiler (CUPTI).

    Warmup runs outside the profiler scope; ITERS calls inside; we sum
    `self_device_time_total` (μs) over every CUDA event in the trace
    and divide by ITERS. Captures ALL kernels launched by `fn` — for
    the PyTorch reference this includes the conv1d + silu fusion, and
    for the +bwd path it includes the gradient kernels too — which is
    exactly what we want to compare.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = 0.0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            total_us += evt.self_device_time_total
    return total_us / iters


def bench_kernel_cpu(fn, warmup: int, iters: int) -> float:
    """Min wall-clock CPU time per call, μs.

    CPU calls are synchronous, so perf_counter_ns around the call
    is the kernel time (no async launch overhead like CUDA). Min
    over samples — CPU benches are noisier (other processes,
    allocator jitter) and min picks the cleanest run.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return min(samples) / 1_000.0


def grouped_bar(labels, groups, *, title, out_path):
    """Render a grouped bar chart.

    `groups` is a list of (label, color, values) tuples, one per bar in
    each cluster. Bars are centered on each x-tick and sized to fit
    inside a 0.8-wide slot regardless of group count.
    """
    n = len(labels)
    n_bars = len(groups)
    x_pos = list(range(n))
    bw = 0.8 / n_bars
    offsets = [(i - (n_bars - 1) / 2) * bw for i in range(n_bars)]
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * n + 2), 5))
    for offset, (lbl, color, vals) in zip(offsets, groups):
        ax.bar([p + offset for p in x_pos], vals, bw, label=lbl, color=color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("GPU kernel time per call (μs, lower is better)")
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

    cache = BaselineCache(__file__)

    fwd_mojo, fwd_up, fwd_pt, fwd_pt_c = [], [], [], []
    bwd_mojo, bwd_up, bwd_pt, bwd_pt_c = [], [], [], []

    for b, d, l, w in SHAPES:
        # ------- forward only -------
        x = torch.randn(b, d, l, generator=g).to("cuda", torch.float16)
        weight = torch.randn(d, w, generator=g).to("cuda", torch.float16)
        bias = torch.randn(d, generator=g).to("cuda", torch.float16)
        kw = dict(bias=bias, activation="silu")
        cfg_fwd = {
            "dtype": "fp16",
            "activation": "silu",
            "bias": True,
            "iters": ITERS_FWD,
            "mode": "fwd",
        }
        m_f = bench_kernel(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw),
            WARMUP_FWD,
            ITERS_FWD,
        )
        u_f = cache.get_or_run(
            impl="upstream",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel(
                lambda: upstream_fn(x, weight, **kw), WARMUP_FWD, ITERS_FWD
            ),
        )
        p_f = cache.get_or_run(
            impl="pytorch",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel(
                lambda: pytorch_fwd(x, weight, bias), WARMUP_FWD, ITERS_FWD
            ),
        )
        pc_f = cache.get_or_run(
            impl="pytorch_compiled",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel(
                lambda: pytorch_fwd_compiled(x, weight, bias),
                WARMUP_FWD,
                ITERS_FWD,
            ),
        )
        fwd_mojo.append(m_f)
        fwd_up.append(u_f)
        fwd_pt.append(p_f)
        fwd_pt_c.append(pc_f)

        # ------- forward + backward -------
        dout = torch.randn(b, d, l, generator=g).to("cuda", torch.float16)
        cfg_bwd = {
            "dtype": "fp16",
            "activation": "silu",
            "bias": True,
            "iters": ITERS_BWD,
            "mode": "fwd+bwd",
        }

        def make_fwd_bwd(impl):
            def step():
                x_g = x.detach().requires_grad_()
                w_g = weight.detach().requires_grad_()
                b_g = bias.detach().requires_grad_()
                if impl == "mojo":
                    out = causal_conv1d_mojo.causal_conv1d_fn(
                        x_g, w_g, bias=b_g, activation="silu"
                    )
                elif impl == "upstream":
                    out = upstream_fn(x_g, w_g, bias=b_g, activation="silu")
                elif impl == "pytorch_compiled":
                    out = pytorch_fwd_compiled(x_g, w_g, b_g)
                else:
                    out = pytorch_fwd(x_g, w_g, b_g)
                out.backward(dout)

            return step

        m_b = bench_kernel(make_fwd_bwd("mojo"), WARMUP_BWD, ITERS_BWD)
        u_b = cache.get_or_run(
            impl="upstream",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel(
                make_fwd_bwd("upstream"), WARMUP_BWD, ITERS_BWD
            ),
        )
        p_b = cache.get_or_run(
            impl="pytorch",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel(make_fwd_bwd("pytorch"), WARMUP_BWD, ITERS_BWD),
        )
        pc_b = cache.get_or_run(
            impl="pytorch_compiled",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel(
                make_fwd_bwd("pytorch_compiled"), WARMUP_BWD, ITERS_BWD
            ),
        )
        bwd_mojo.append(m_b)
        bwd_up.append(u_b)
        bwd_pt.append(p_b)
        bwd_pt_c.append(pc_b)

        print(
            f"{(b, d, l, w)!s:>22}  "
            f"fwd: mojo={m_f:7.1f} up={u_f:7.1f} pt={p_f:7.1f} pt-c={pc_f:7.1f} | "
            f"fwd+bwd: mojo={m_b:7.1f} up={u_b:7.1f} pt={p_b:7.1f} pt-c={pc_b:7.1f}"
        )

    grouped_bar(
        labels,
        [
            ("pure PyTorch (F.conv1d + F.silu)", "#bbbbbb", fwd_pt),
            ("torch.compile(pure PyTorch)", "#88c070", fwd_pt_c),
            ("upstream (Tri Dao CUDA)", "#3a78c2", fwd_up),
            ("mojo (this repo)", "#d05050", fwd_mojo),
        ],
        title=(
            f"causal_conv1d FORWARD — {gpu_name}\n"
            f"fp16, bias, silu, {ITERS_FWD} iters, GPU kernel time via torch.profiler"
        ),
        out_path=DOCS / "bench_forward.png",
    )
    grouped_bar(
        labels,
        [
            ("pure PyTorch (F.conv1d + F.silu)", "#bbbbbb", bwd_pt),
            ("torch.compile(pure PyTorch)", "#88c070", bwd_pt_c),
            ("upstream (Tri Dao CUDA)", "#3a78c2", bwd_up),
            ("mojo (this repo)", "#d05050", bwd_mojo),
        ],
        title=(
            f"causal_conv1d FORWARD + BACKWARD — {gpu_name}\n"
            f"fp16, bias, silu, {ITERS_BWD} iters, GPU kernel time via torch.profiler"
        ),
        out_path=DOCS / "bench_backward.png",
    )

    # ---- single-step update bench ----
    update_labels = [f"({b},{d})" for b, d in UPDATE_SHAPES]
    update_mojo, update_up, update_ref, update_ref_c = [], [], [], []
    W = 4
    state_len = W - 1
    cfg_upd = {
        "dtype": "fp16",
        "activation": "silu",
        "bias": True,
        "iters": ITERS_UPDATE,
        "mode": "update",
        "width": W,
        "state_len": state_len,
    }
    for b, d in UPDATE_SHAPES:
        x = torch.randn(b, d, generator=g).to("cuda", torch.float16)
        weight = torch.randn(d, W, generator=g).to("cuda", torch.float16)
        bias = torch.randn(d, generator=g).to("cuda", torch.float16)

        def make_step(impl, x=x, weight=weight, bias=bias):
            # Each call needs its own state (mutated in place); reset
            # before timing so the per-call cost is consistent.
            state = torch.randn(b, d, state_len, generator=g).to("cuda", torch.float16)
            if impl == "mojo":
                fn = causal_conv1d_mojo.causal_conv1d_update
            elif impl == "upstream":
                fn = upstream_update_fn
            elif impl == "ref_compiled":
                fn = update_ref_compiled
            else:
                fn = causal_conv1d_update_ref

            def step():
                fn(x, state, weight, bias=bias, activation="silu")

            return step

        m_u = bench_kernel(make_step("mojo"), WARMUP_UPDATE, ITERS_UPDATE)
        u_u = cache.get_or_run(
            impl="upstream",
            shape=(b, d),
            config=cfg_upd,
            run=lambda: bench_kernel(
                make_step("upstream"), WARMUP_UPDATE, ITERS_UPDATE
            ),
        )
        r_u = cache.get_or_run(
            impl="ref",
            shape=(b, d),
            config=cfg_upd,
            run=lambda: bench_kernel(make_step("ref"), WARMUP_UPDATE, ITERS_UPDATE),
        )
        rc_u = cache.get_or_run(
            impl="ref_compiled",
            shape=(b, d),
            config=cfg_upd,
            run=lambda: bench_kernel(
                make_step("ref_compiled"), WARMUP_UPDATE, ITERS_UPDATE
            ),
        )
        update_mojo.append(m_u)
        update_up.append(u_u)
        update_ref.append(r_u)
        update_ref_c.append(rc_u)
        print(
            f"{(b, d)!s:>14}  update: mojo={m_u:7.1f} up={u_u:7.1f} "
            f"ref={r_u:7.1f} ref-c={rc_u:7.1f}"
        )

    grouped_bar(
        update_labels,
        [
            ("pure PyTorch (causal_conv1d_update_ref)", "#bbbbbb", update_ref),
            ("torch.compile(causal_conv1d_update_ref)", "#88c070", update_ref_c),
            ("upstream (Tri Dao CUDA)", "#3a78c2", update_up),
            ("mojo (this repo)", "#d05050", update_mojo),
        ],
        title=(
            f"causal_conv1d_update (single-step decode) — {gpu_name}\n"
            f"fp16, bias, silu, seqlen=1, state_len=3, "
            f"{ITERS_UPDATE} iters, GPU kernel time via torch.profiler"
        ),
        out_path=DOCS / "bench_update.png",
    )

    run_cpu_benches(g, cache)


def run_cpu_benches(g, cache) -> None:
    """CPU equivalents of the three GPU plots.

    Upstream's `causal_conv1d_fn` is CUDA-only; the analog on CPU is
    the pure-pytorch `causal_conv1d_ref` (and same for the update op).
    """
    cpu_threads = torch.get_num_threads()
    labels = [f"({b},{d},{l},{w})" for b, d, l, w in CPU_SHAPES]

    fwd_mojo, fwd_ref, fwd_pt, fwd_pt_c = [], [], [], []
    bwd_mojo, bwd_ref, bwd_pt, bwd_pt_c = [], [], [], []

    for b, d, l, w in CPU_SHAPES:
        x = torch.randn(b, d, l, generator=g).to(torch.float16)
        weight = torch.randn(d, w, generator=g).to(torch.float16)
        bias = torch.randn(d, generator=g).to(torch.float16)
        kw = dict(bias=bias, activation="silu")
        cfg_fwd = {
            "dtype": "fp16",
            "activation": "silu",
            "bias": True,
            "iters": ITERS_FWD_CPU,
            "mode": "fwd",
            "device": "cpu",
        }
        m_f = bench_kernel_cpu(
            lambda: causal_conv1d_mojo.causal_conv1d_fn(x, weight, **kw),
            WARMUP_FWD_CPU,
            ITERS_FWD_CPU,
        )
        r_f = cache.get_or_run(
            impl="causal_conv1d_ref",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel_cpu(
                lambda: causal_conv1d_ref(x, weight, **kw),
                WARMUP_FWD_CPU,
                ITERS_FWD_CPU,
            ),
        )
        p_f = cache.get_or_run(
            impl="pytorch",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel_cpu(
                lambda: pytorch_fwd(x, weight, bias),
                WARMUP_FWD_CPU,
                ITERS_FWD_CPU,
            ),
        )
        pc_f = cache.get_or_run(
            impl="pytorch_compiled",
            shape=(b, d, l, w),
            config=cfg_fwd,
            run=lambda: bench_kernel_cpu(
                lambda: pytorch_fwd_compiled(x, weight, bias),
                WARMUP_FWD_CPU,
                ITERS_FWD_CPU,
            ),
        )
        fwd_mojo.append(m_f)
        fwd_ref.append(r_f)
        fwd_pt.append(p_f)
        fwd_pt_c.append(pc_f)

        # ------- forward + backward -------
        dout = torch.randn(b, d, l, generator=g).to(torch.float16)
        cfg_bwd = {**cfg_fwd, "iters": ITERS_BWD_CPU, "mode": "fwd+bwd"}

        def make_fwd_bwd_cpu(impl):
            def step():
                x_g = x.detach().requires_grad_()
                w_g = weight.detach().requires_grad_()
                b_g = bias.detach().requires_grad_()
                if impl == "mojo":
                    out = causal_conv1d_mojo.causal_conv1d_fn(
                        x_g, w_g, bias=b_g, activation="silu"
                    )
                elif impl == "ref":
                    out = causal_conv1d_ref(x_g, w_g, bias=b_g, activation="silu")
                elif impl == "pytorch_compiled":
                    out = pytorch_fwd_compiled(x_g, w_g, b_g)
                else:
                    out = pytorch_fwd(x_g, w_g, b_g)
                out.backward(dout)

            return step

        m_b = bench_kernel_cpu(
            make_fwd_bwd_cpu("mojo"), WARMUP_BWD_CPU, ITERS_BWD_CPU
        )
        r_b = cache.get_or_run(
            impl="causal_conv1d_ref",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel_cpu(
                make_fwd_bwd_cpu("ref"), WARMUP_BWD_CPU, ITERS_BWD_CPU
            ),
        )
        p_b = cache.get_or_run(
            impl="pytorch",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel_cpu(
                make_fwd_bwd_cpu("pytorch"), WARMUP_BWD_CPU, ITERS_BWD_CPU
            ),
        )
        pc_b = cache.get_or_run(
            impl="pytorch_compiled",
            shape=(b, d, l, w),
            config=cfg_bwd,
            run=lambda: bench_kernel_cpu(
                make_fwd_bwd_cpu("pytorch_compiled"),
                WARMUP_BWD_CPU,
                ITERS_BWD_CPU,
            ),
        )
        bwd_mojo.append(m_b)
        bwd_ref.append(r_b)
        bwd_pt.append(p_b)
        bwd_pt_c.append(pc_b)

        print(
            f"{(b, d, l, w)!s:>22} CPU  "
            f"fwd: mojo={m_f:8.1f} ref={r_f:8.1f} pt={p_f:8.1f} pt-c={pc_f:8.1f} | "
            f"fwd+bwd: mojo={m_b:8.1f} ref={r_b:8.1f} pt={p_b:8.1f} pt-c={pc_b:8.1f}"
        )

    grouped_bar(
        labels,
        [
            ("pure PyTorch (F.conv1d + F.silu)", "#bbbbbb", fwd_pt),
            ("torch.compile(pure PyTorch)", "#88c070", fwd_pt_c),
            ("causal_conv1d_ref (PyTorch reference)", "#3a78c2", fwd_ref),
            ("mojo (this repo)", "#d05050", fwd_mojo),
        ],
        title=(
            f"causal_conv1d FORWARD — CPU ({cpu_threads} threads)\n"
            f"fp16, bias, silu, {ITERS_FWD_CPU} iters, min wall-clock time"
        ),
        out_path=DOCS / "bench_forward_cpu.png",
    )
    grouped_bar(
        labels,
        [
            ("pure PyTorch (F.conv1d + F.silu)", "#bbbbbb", bwd_pt),
            ("torch.compile(pure PyTorch)", "#88c070", bwd_pt_c),
            ("causal_conv1d_ref (PyTorch reference)", "#3a78c2", bwd_ref),
            ("mojo (this repo)", "#d05050", bwd_mojo),
        ],
        title=(
            f"causal_conv1d FORWARD + BACKWARD — CPU ({cpu_threads} threads)\n"
            f"fp16, bias, silu, {ITERS_BWD_CPU} iters, min wall-clock time"
        ),
        out_path=DOCS / "bench_backward_cpu.png",
    )

    # ---- single-step update bench (CPU) ----
    update_labels = [f"({b},{d})" for b, d in UPDATE_SHAPES]
    update_mojo, update_ref_l, update_ref_c = [], [], []
    W = 4
    state_len = W - 1
    cfg_upd = {
        "dtype": "fp16",
        "activation": "silu",
        "bias": True,
        "iters": ITERS_UPDATE_CPU,
        "mode": "update",
        "width": W,
        "state_len": state_len,
        "device": "cpu",
    }
    for b, d in UPDATE_SHAPES:
        x = torch.randn(b, d, generator=g).to(torch.float16)
        weight = torch.randn(d, W, generator=g).to(torch.float16)
        bias = torch.randn(d, generator=g).to(torch.float16)

        def make_step_cpu(impl, x=x, weight=weight, bias=bias):
            state = torch.randn(b, d, state_len, generator=g).to(torch.float16)
            if impl == "mojo":
                fn = causal_conv1d_mojo.causal_conv1d_update
            elif impl == "ref_compiled":
                fn = update_ref_compiled
            else:
                fn = causal_conv1d_update_ref

            def step():
                fn(x, state, weight, bias=bias, activation="silu")

            return step

        m_u = bench_kernel_cpu(
            make_step_cpu("mojo"), WARMUP_UPDATE_CPU, ITERS_UPDATE_CPU
        )
        r_u = cache.get_or_run(
            impl="ref",
            shape=(b, d),
            config=cfg_upd,
            run=lambda: bench_kernel_cpu(
                make_step_cpu("ref"), WARMUP_UPDATE_CPU, ITERS_UPDATE_CPU
            ),
        )
        rc_u = cache.get_or_run(
            impl="ref_compiled",
            shape=(b, d),
            config=cfg_upd,
            run=lambda: bench_kernel_cpu(
                make_step_cpu("ref_compiled"), WARMUP_UPDATE_CPU, ITERS_UPDATE_CPU
            ),
        )
        update_mojo.append(m_u)
        update_ref_l.append(r_u)
        update_ref_c.append(rc_u)
        print(
            f"{(b, d)!s:>14} CPU  update: mojo={m_u:8.1f} ref={r_u:8.1f} ref-c={rc_u:8.1f}"
        )

    grouped_bar(
        update_labels,
        [
            ("causal_conv1d_update_ref (PyTorch)", "#bbbbbb", update_ref_l),
            ("torch.compile(causal_conv1d_update_ref)", "#88c070", update_ref_c),
            ("mojo (this repo)", "#d05050", update_mojo),
        ],
        title=(
            f"causal_conv1d_update (single-step decode) — CPU ({cpu_threads} threads)\n"
            f"fp16, bias, silu, seqlen=1, state_len=3, "
            f"{ITERS_UPDATE_CPU} iters, min wall-clock time"
        ),
        out_path=DOCS / "bench_update_cpu.png",
    )


if __name__ == "__main__":
    main()
