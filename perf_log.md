# Update kernel perf log

## Baseline (already at-target across grid)

Bench: `bench_gpu_kernel_time.py`, fp16, silu, bias, width=4, state_len=3, 500 iters
GPU: NVIDIA H100 80GB HBM3

```
   shape (B,D) |  mojo (us/call) |  upstream (us/call) |   ratio
----------------------------------------------------------------
      (1, 256) |            1.73 |                2.06 |   0.84x
      (1, 512) |            1.83 |                2.09 |   0.88x
     (1, 1024) |            1.93 |                2.20 |   0.88x
     (1, 2048) |            1.99 |                2.16 |   0.92x
     (1, 4096) |            2.04 |                2.22 |   0.92x
     (4, 1024) |            2.02 |                2.20 |   0.92x
     (4, 2048) |            2.11 |                2.30 |   0.92x
     (4, 4096) |            2.32 |                2.40 |   0.97x
    (16, 2048) |            2.45 |                2.50 |   0.98x
    (32, 4096) |            3.39 |                3.44 |   0.99x
```

Already <=1.0x upstream across the entire grid (best 0.84x, worst 0.99x).

## What we have

The kernel already:
- 1 thread per channel, 64 channels per block (matches upstream's `kNThreads=64`).
- Skips the state-shift loop trivially when `state_len == width-1` (the typical
  Mamba decode config) via a single PTX predicate `setp.lt.s64`.
- Uses `ld.global.nc.b16` (read-only with cache hint) for weight/state reads.
- Uses comptime branching on `is_circular`, `has_state_indices`, `apply_silu`,
  `has_bias` to keep each path tight (16 variants per dtype × 3 widths × 3
  dtypes = 144 total cubins; mutually-exclusive combos are filtered).
- Phase 2 (history load) is a `comptime for` loop, fully unrolled.
- Inner conv is 4 FMAs, ptxas-vectorized cleanly.

PTX of the canonical dtype=fp16, width=4, has_bias=True, apply_silu=True,
has_state_indices=False, is_circular=False variant:

- 4 `ld.global.nc.b16` (weight) + 1 `ld.global.b16` (bias).
- 3 `ld.global.b16` (history, conditional on width=4).
- Up to 3 `st.global.b16` (state writeback in phase 2, predicated).
- Inner loop body: 1 ld + 4 fma + silu (exp+div) + 1 st.

## Notes on attempted optimizations not pursued

- **Vectorize the 3 history reads** at state_len=3, width=4: would need a
  64-bit + 16-bit pair (6 bytes is awkward) and add a comptime fast-path
  gated on `state_len == width-1`, doubling the dispatch tree. Not worth
  it when ratio is already 0.84x-0.99x.
- **Reciprocal-multiply silu** (`x * (1/(1+exp(-x)))` via `rcp.approx.f32`)
  would save a `div.rn.f32` per output. With only 1 output per call (the
  common decode case), that's ~5 cycles per kernel of ~2us — sub-1%.
- **Comptime-specialize on `seqlen==1`**: would shave the loop-header
  predicate but adds a dispatcher branch + doubles compiled variant
  count. Diminishing returns.
- **Vector loads for x and output**: only 1 element per thread; nothing
  to vectorize.

## Conclusion

The Mojo update kernel beats upstream on every shape tested (10/10).
Worst-case margin (32, 4096) is 0.99x — a 1% improvement over upstream's
CUDA kernel. Best margin (1, 256) is 0.84x. The kernel is structurally
near-identical to upstream's; the Mojo/LLVM-NVPTX codegen produces
slightly tighter PTX, primarily benefiting small-grid launch latency.
No code changes to the update kernel were made.
