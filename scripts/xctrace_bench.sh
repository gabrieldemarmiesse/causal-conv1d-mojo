#!/usr/bin/env bash
# xctrace_bench.sh — precise Apple-GPU kernel timing for causal_conv1d.
#
# The Apple analog of running `bench_gpu_kernel_time.py` under CUPTI:
# Metal has no torch device-time hook, so we record an Instruments
# "Metal System Trace" around the Mojo Metal kernel and read the
# per-encoder GPU intervals back out with xctrace_gpu_intervals.py.
#
# Usage:
#   scripts/xctrace_bench.sh [--kind fwd|bwd|update] [--shape B,D,L,W]
#                            [--dtype fp16|bf16|fp32] [--iters N]
#                            [--activation silu|identity] [--needle STR]
#                            [--output trace.trace]
#
# Examples:
#   scripts/xctrace_bench.sh --kind fwd
#   scripts/xctrace_bench.sh --kind fwd --shape 1,1024,2048,4 --needle fwd
#   scripts/xctrace_bench.sh --kind update --dtype bf16
#
# The trace bundle is kept so you can open it in Instruments for the
# GUI-only HW counters (ALU busy, bandwidth, occupancy) that headless
# xctrace export does not expose.

set -euo pipefail
cd "$(dirname "$0")/.."

KIND=fwd
SHAPE=""
DTYPE=fp16
ITERS=50
ACT=silu
NEEDLE=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind)       KIND="$2"; shift ;;
    --shape)      SHAPE="$2"; shift ;;
    --dtype)      DTYPE="$2"; shift ;;
    --iters)      ITERS="$2"; shift ;;
    --activation) ACT="$2"; shift ;;
    --needle)     NEEDLE="$2"; shift ;;
    --output)     OUTPUT="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Mojo doesn't label its Metal encoders, so the parser groups our kernel
# under `Compute / Compute Command`; leave the needle empty to also see
# the host<->device `Blit` copies, or pass --needle Compute to isolate
# just the conv kernel.
[[ -z "$OUTPUT" ]] && OUTPUT="$(mktemp -d)/ccv_${KIND}.trace"

BENCH_ARGS=(--kind "$KIND" --dtype "$DTYPE" --iters "$ITERS" --activation "$ACT")
[[ -n "$SHAPE" ]] && BENCH_ARGS+=(--shape "$SHAPE")

# 1) Pre-warm: run once in the full `uv` env so the JIT `mojo build` lands
#    in the on-disk cache. The traced run below then only dlopen()s the
#    cached .so, keeping `mojo build` out of the trace.
echo "=== pre-warm (fills JIT cache; not traced) ==="
uv run python benchmarks/bench_metal_gpu.py "${BENCH_ARGS[@]}" --warmup 5

# 2) Resolve the venv interpreter so we can hand xctrace a concrete
#    executable to --launch (it cannot launch the `uv run` wrapper cleanly).
PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

echo
echo "=== xctrace record (Metal System Trace) -> $OUTPUT ==="
# xctrace intermittently crashes (Bus/Segfault) while finalizing the
# trace bundle, leaving an unexportable .trace. It's flaky, not
# deterministic, so retry until `xctrace export` can actually read the
# result back. `|| true` keeps `set -e` from aborting on the crash.
ATTEMPTS=6
ok=0
for attempt in $(seq 1 "$ATTEMPTS"); do
  rm -rf "$OUTPUT"
  xctrace record --template "Metal System Trace" --output "$OUTPUT" \
    --launch -- "$PYTHON" benchmarks/bench_metal_gpu.py \
      "${BENCH_ARGS[@]}" --warmup 1 || true
  # Valid bundle iff the intervals table exports without error.
  if xctrace export --input "$OUTPUT" \
       --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
       >/dev/null 2>&1; then
    ok=1
    break
  fi
  echo "  (attempt $attempt/$ATTEMPTS: xctrace produced an unreadable trace, retrying)"
done
if [[ "$ok" -ne 1 ]]; then
  echo "xctrace failed to produce a readable trace after $ATTEMPTS attempts." >&2
  exit 1
fi

echo
echo "=== per-encoder GPU intervals (Compute Command = the conv kernel) ==="
# --clock: split by GPU clock state. Apple DVFS drops the clock between the
# per-call syncs, so the 'Maximum'-clock row is the trustworthy steady-state
# kernel time; lower-clock rows are throttled noise.
uv run python scripts/xctrace_gpu_intervals.py "$OUTPUT" $NEEDLE --clock || true

echo
echo "trace kept at: $OUTPUT (open in Instruments for HW counters)"
