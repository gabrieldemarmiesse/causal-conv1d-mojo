#!/usr/bin/env python3
"""Extract per-encoder GPU execution intervals from an xctrace trace.

This is the Apple-silicon analog of `bench.py`'s
torch.profiler path: on NVIDIA/AMD we read per-kernel device time from
CUPTI/rocprof, but Metal has no torch device-time hook, so the only way
to get real GPU-side timings is to record an Instruments "Metal System
Trace" and read the per-encoder intervals back out.

Usage:
  # record a trace (see scripts/xctrace_bench.sh for the full pipeline):
  xctrace record --template 'Metal System Trace' --output bench.trace \
      --launch -- <python> benchmarks/bench_metal_gpu.py --kind fwd ...

  # then parse it (defaults to the python process; Compute = our kernel):
  python scripts/xctrace_gpu_intervals.py bench.trace
  python scripts/xctrace_gpu_intervals.py bench.trace --needle Compute --raw

The Metal System Trace's `metal-gpu-intervals` table lumps *every*
process's GPU work together (WindowServer compositing dominates the row
count), so we filter by `--process` (default `python`). Mojo does not
set Metal encoder labels (`metal-object-label` is empty), so encoders
are grouped by GPU channel + the command-buffer encoder label with the
per-iteration indices stripped — our forward kernel lands under
`Compute / Compute Command`, host<->device copies under `Blit Command`.

GPU hardware counters (ALU busy, bandwidth, occupancy) are NOT in
headless traces; that part of Instruments is GUI-only on Apple silicon —
open the .trace bundle to see them.
"""

import argparse
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def export_table(trace: str, schema: str = "metal-gpu-intervals") -> bytes:
    """Run `xctrace export` for the given trace table; return XML."""
    out = subprocess.run(
        [
            "xctrace",
            "export",
            "--input",
            trace,
            "--xpath",
            f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        ],
        capture_output=True,
    )
    if out.returncode != 0:
        sys.exit(f"xctrace export failed:\n{out.stderr.decode(errors='replace')}")
    return out.stdout


def _resolver(root):
    """id/ref value-dictionary resolver for an exported table (see
    parse_intervals). Returns a fn mapping a ref-element to its definition."""
    registry = {el.get("id"): el for el in root.iter() if el.get("id")}

    def resolve(el):
        ref = el.get("ref")
        return registry.get(ref, el) if ref else el

    return resolve


def load_clock_timeline(trace: str):
    """Return a sorted list of (start_ns, end_ns, state) GPU clock windows.

    Parsed from `gpu-performance-state-intervals`, which the Metal System
    Trace populates headlessly. `state` is e.g. "Maximum"/"Minimum". Empty
    if the table is absent. Lets us tag each GPU interval with the clock it
    ran at — Apple's DVFS drops the GPU to its minimum state between the
    synchronized per-call dispatches, so short kernels are often measured
    downclocked; filtering to "Maximum" recovers a stable steady-state time
    without needing the Instruments GUI.
    """
    try:
        root = ET.fromstring(export_table(trace, "gpu-performance-state-intervals"))
    except SystemExit:
        return []
    resolve = _resolver(root)
    windows = []
    for row in root.iter("row"):
        start = dur = None
        state = None
        for k in row:
            if k.tag == "start-time":
                start = int(resolve(k).text)
            elif k.tag == "duration":
                dur = int(resolve(k).text)
            elif k.tag == "gpu-performance-state":
                state = resolve(k).get("fmt", "")
        if start is not None and state:
            windows.append((start, start + (dur or 0), state))
    windows.sort()
    return windows


def clock_at(windows, t: int) -> str:
    """GPU clock state active at timestamp `t` (ns); '' if unknown."""
    for s, e, st in windows:
        if s <= t < e:
            return st
    return ""


def _normalize_label(lbl: str) -> str:
    """Collapse per-iteration encoder labels so they group across calls.

    `Command Buffer 7:Compute Command 0` -> `Compute Command`.
    """
    lbl = re.sub(r"Command Buffer \d+:", "", lbl)
    lbl = re.sub(r" \d+$", "", lbl)
    return lbl.strip() or "?"


def parse_intervals(xml: bytes, process: str):
    """Yield (channel, label, start_fmt, start_ns, duration_ns) per encoder.

    The export uses a global id/ref value dictionary (the first use of a
    value carries `id=` + the data; later uses are `<tag ref=.../>`), so
    we resolve refs against a document-wide registry. Each row's GPU
    duration is its *first* `<duration>` child — the second one is the
    "CPU to GPU Latency" column, also typed `duration`.
    """
    root = ET.fromstring(xml)
    resolve = _resolver(root)

    for row in root.iter("row"):
        kids = list(row)

        proc = ""
        durations = []
        channel = ""
        flabel = None
        for k in kids:
            if k.tag == "process" and not proc:
                proc = resolve(k).get("fmt", "")
            elif k.tag == "duration":
                durations.append(resolve(k))
            elif k.tag == "gpu-channel-name" and not channel:
                channel = resolve(k).get("fmt", "")
            elif k.tag == "formatted-label" and flabel is None:
                flabel = resolve(k)

        if process and process not in proc:
            continue
        if not durations or durations[0].text is None:
            continue

        label = "?"
        if flabel is not None:
            s = flabel.find("string")
            if s is not None:
                label = resolve(s).get("fmt", "") or "?"

        start = ""
        start_ns = 0
        for k in kids:
            if k.tag == "start-time":
                el = resolve(k)
                start = el.get("fmt", "")
                start_ns = int(el.text)
                break

        yield (
            channel,
            _normalize_label(label),
            start,
            start_ns,
            int(durations[0].text),
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize per-encoder GPU intervals from an xctrace trace."
    )
    p.add_argument("trace", help="path to the .trace bundle")
    p.add_argument(
        "needle",
        nargs="?",
        default="",
        help="case-insensitive substring filter on channel/label "
        "(e.g. 'Compute' to isolate the conv kernel)",
    )
    p.add_argument(
        "--process",
        default="python",
        help="only rows from processes whose name contains this "
        "(default 'python'; '' for every process)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="also print every matching interval, not just the summary",
    )
    p.add_argument(
        "--clock",
        action="store_true",
        help="tag each interval with the GPU clock state it ran at "
        "(from gpu-performance-state-intervals) and split the "
        "summary by state. Apple's DVFS downclocks between the "
        "per-call syncs, so 'Maximum' rows are the trustworthy "
        "steady-state time and the rest is DVFS noise.",
    )
    args = p.parse_args()

    needle = args.needle.lower()
    windows = load_clock_timeline(args.trace) if args.clock else []
    # Key is (channel, label) or (channel, label, clock_state) with --clock.
    groups: dict[tuple, list[int]] = defaultdict(list)
    n = 0
    for channel, label, start, start_ns, ns in parse_intervals(
        export_table(args.trace), args.process
    ):
        if needle and needle not in channel.lower() and needle not in label.lower():
            continue
        n += 1
        clock = clock_at(windows, start_ns) or "?" if args.clock else None
        key = (channel, label, clock) if args.clock else (channel, label)
        groups[key].append(ns)
        if args.raw:
            tag = f"  {clock:>8}" if args.clock else ""
            print(f"{start:>14}  {ns / 1e3:>10.2f}us  {channel:>8}{tag}  {label}")

    if not groups:
        sys.exit(
            f"no GPU intervals matched (process={args.process!r}, "
            f"needle={args.needle!r})"
        )

    if args.clock and not windows:
        print("(no gpu-performance-state-intervals in trace; clock unknown)\n")

    if args.raw:
        print()
    clock_col = f"  {'clock':>8}" if args.clock else ""
    head = (
        f"{'channel':>8}{clock_col}  {'count':>5}  {'median':>10}  "
        f"{'min':>10}  {'max':>10}  encoder"
    )
    print(head)
    print("-" * len(head))
    for key, durs in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        channel, label = key[0], key[1]
        clock_cell = f"  {key[2]:>8}" if args.clock else ""
        # Median, not mean: with DVFS the distribution is bimodal, and the
        # median of a single-clock group is a robust steady-state estimate.
        print(
            f"{channel:>8}{clock_cell}  {len(durs):>5}  "
            f"{statistics.median(durs) / 1e3:>8.2f}us  "
            f"{min(durs) / 1e3:>8.2f}us  {max(durs) / 1e3:>8.2f}us  {label}"
        )

    grand = sum(sum(d) for d in groups.values())
    print(
        f"\ntotal GPU time {grand / 1e6:.3f} ms across {n} encoder intervals "
        f"(process filter: {args.process!r})"
    )
    if args.clock and windows:
        print("Trust the 'Maximum'-clock rows; other states are DVFS-throttled.")


if __name__ == "__main__":
    main()
