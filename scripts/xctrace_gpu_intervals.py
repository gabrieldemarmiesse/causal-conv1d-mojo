#!/usr/bin/env python3
"""Extract per-encoder GPU execution intervals from an xctrace trace.

This is the Apple-silicon analog of `bench_gpu_kernel_time.py`'s
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


def export_table(trace: str) -> bytes:
    """Run `xctrace export` for the metal-gpu-intervals table; return XML."""
    out = subprocess.run(
        [
            "xctrace", "export", "--input", trace,
            "--xpath",
            '/trace-toc/run[@number="1"]/data/'
            'table[@schema="metal-gpu-intervals"]',
        ],
        capture_output=True,
    )
    if out.returncode != 0:
        sys.exit(f"xctrace export failed:\n{out.stderr.decode(errors='replace')}")
    return out.stdout


def _normalize_label(lbl: str) -> str:
    """Collapse per-iteration encoder labels so they group across calls.

    `Command Buffer 7:Compute Command 0` -> `Compute Command`.
    """
    lbl = re.sub(r"Command Buffer \d+:", "", lbl)
    lbl = re.sub(r" \d+$", "", lbl)
    return lbl.strip() or "?"


def parse_intervals(xml: bytes, process: str):
    """Yield (channel, label, start_fmt, duration_ns) per matching encoder.

    The export uses a global id/ref value dictionary (the first use of a
    value carries `id=` + the data; later uses are `<tag ref=.../>`), so
    we resolve refs against a document-wide registry. Each row's GPU
    duration is its *first* `<duration>` child — the second one is the
    "CPU to GPU Latency" column, also typed `duration`.
    """
    root = ET.fromstring(xml)
    registry = {el.get("id"): el for el in root.iter() if el.get("id")}

    def resolve(el):
        ref = el.get("ref")
        return registry.get(ref, el) if ref else el

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
        for k in kids:
            if k.tag == "start-time":
                start = resolve(k).get("fmt", "")
                break

        yield channel, _normalize_label(label), start, int(durations[0].text)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize per-encoder GPU intervals from an xctrace trace.")
    p.add_argument("trace", help="path to the .trace bundle")
    p.add_argument("needle", nargs="?", default="",
                   help="case-insensitive substring filter on channel/label "
                        "(e.g. 'Compute' to isolate the conv kernel)")
    p.add_argument("--process", default="python",
                   help="only rows from processes whose name contains this "
                        "(default 'python'; '' for every process)")
    p.add_argument("--raw", action="store_true",
                   help="also print every matching interval, not just the summary")
    args = p.parse_args()

    needle = args.needle.lower()
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    n = 0
    for channel, label, start, ns in parse_intervals(
        export_table(args.trace), args.process
    ):
        if needle and needle not in channel.lower() and needle not in label.lower():
            continue
        n += 1
        groups[(channel, label)].append(ns)
        if args.raw:
            print(f"{start:>14}  {ns / 1e3:>10.2f}us  {channel:>8}  {label}")

    if not groups:
        sys.exit(f"no GPU intervals matched (process={args.process!r}, "
                 f"needle={args.needle!r})")

    if args.raw:
        print()
    head = (f"{'channel':>8}  {'count':>5}  {'mean':>10}  {'min':>10}  "
            f"{'max':>10}  encoder")
    print(head)
    print("-" * len(head))
    for (channel, label), durs in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        print(f"{channel:>8}  {len(durs):>5}  {statistics.mean(durs) / 1e3:>8.2f}us  "
              f"{min(durs) / 1e3:>8.2f}us  {max(durs) / 1e3:>8.2f}us  {label}")

    grand = sum(sum(d) for d in groups.values())
    print(f"\ntotal GPU time {grand / 1e6:.3f} ms across {n} encoder intervals "
          f"(process filter: {args.process!r})")


if __name__ == "__main__":
    main()
