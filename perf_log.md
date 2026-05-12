# Backward kernel perf log

## Baseline (after merging knelts=8 + vec-smem from claude-agent-best-perf)

GPU: NVIDIA H100 80GB HBM3 | dtype=fp16 | activation=silu | bias=True | iters=100

|       shape (B,D,L,W) |  mojo us | upstream us | ratio |
|----------------------:|---------:|------------:|------:|
|     (1, 1024,  512, 4) |     4.5 |        4.3 | 1.05x |
|    (1, 1024, 2048, 4) |     7.1 |        5.9 | 1.21x |
|    (1, 1024, 8192, 4) |    22.5 |       19.8 | 1.14x |
|    (1, 2048, 2048, 4) |    12.9 |       10.5 | 1.23x |
|    (1, 4096, 2048, 4) |    26.5 |       22.0 | 1.20x |
|    (4, 2048, 2048, 4) |    47.5 |       40.9 | 1.16x |
|    (4, 4096, 2048, 4) |    92.0 |       75.7 | 1.21x |
|    (8, 2048, 4096, 4) |   159.2 |      139.8 | 1.14x |

Worst: (1, 2048, 2048, 4) at 1.23x.

## Iterations

### Opt1: pack width dweight + dbias block-sums into single vectorised reduce

Replace `width` sequential calls to `_block_sum_f32` (one barrier each) +
the optional dbias call with one call to `_block_sum_f32_vec[n=width+1]`
that packs everything into a single barrier. Saves `width` barriers in
the post-loop reduce phase (4-5 fewer block-wide syncthreads).

|       shape (B,D,L,W) |  mojo us | upstream us | ratio |  Δratio |
|----------------------:|---------:|------------:|------:|--------:|
|     (1, 1024,  512, 4) |     3.8 |        4.3 | 0.87x |  -0.18  |
|    (1, 1024, 2048, 4) |     6.3 |        5.9 | 1.07x |  -0.14  |
|    (1, 1024, 8192, 4) |    21.7 |       19.9 | 1.10x |  -0.04  |
|    (1, 2048, 2048, 4) |    11.3 |       10.5 | 1.08x |  -0.15  |
|    (1, 4096, 2048, 4) |    23.5 |       22.0 | 1.07x |  -0.13  |
|    (4, 2048, 2048, 4) |    42.8 |       40.9 | 1.05x |  -0.11  |
|    (4, 4096, 2048, 4) |    82.9 |       75.8 | 1.09x |  -0.12  |
|    (8, 2048, 4096, 4) |   152.0 |      139.7 | 1.09x |  -0.05  |

### Opt2: drop redundant pre-write `barrier()` in dout-exchange dance

The first `barrier()` before the `tidx>0` writes to `smem_dout` (was
labelled "all reads of smem_x done; safe to reuse smem_dout") wasn't
load-bearing: `smem_x` and `smem_dout` are *separate* stack-allocated
shared buffers (no overlap), and every `smem_dout[tidx*kNElts..]` slot
for `tidx>0` was last *read* in the previous chunk iter and protected
by that iter's "all halo reads done" barrier — which the start-of-iter
`barrier()` (`smem_x writes visible`) covers transitively. One fewer
block-wide sync per chunk iter.

|       shape (B,D,L,W) |  mojo us | upstream us | ratio |  Δratio |
|----------------------:|---------:|------------:|------:|--------:|
|     (1, 1024,  512, 4) |     3.7 |        4.3 | 0.87x |  -0.00  |
|    (1, 1024, 2048, 4) |     6.3 |        5.9 | 1.06x |  -0.01  |
|    (1, 1024, 8192, 4) |    21.5 |       19.8 | 1.08x |  -0.02  |
|    (1, 2048, 2048, 4) |    11.2 |       10.5 | 1.06x |  -0.02  |
|    (1, 4096, 2048, 4) |    23.1 |       22.1 | 1.05x |  -0.02  |
|    (4, 2048, 2048, 4) |    42.2 |       40.9 | 1.03x |  -0.02  |
|    (4, 4096, 2048, 4) |    82.0 |       75.7 | 1.08x |  -0.01  |
|    (8, 2048, 4096, 4) |   150.5 |      139.7 | 1.08x |  -0.01  |

### Opt10: vec-load tidx-0's x_prev from global (16 bytes/thread, 1 LDG.E.128)

Replace the per-element bounds-checked scalar loop for tidx==0's
x_prev (which read 8 fp16 elements one at a time) with a single
16-byte vec load when `contig_inner`. The chunk_start is always a
multiple of `kChunkSize = kNThreads * kNElts`, so `chunk_start -
kNElts` is kNElts-aligned, giving us the LDG.E.128 promise.

|       shape (B,D,L,W) |  mojo us | upstream us | ratio |  Δratio |
|----------------------:|---------:|------------:|------:|--------:|
|     (1, 1024,  512, 4) |     3.6 |        4.3 | 0.83x |  -0.02  |
|    (1, 1024, 2048, 4) |     5.4 |        5.9 | 0.91x |   0.00  |
|    (1, 1024, 8192, 4) |    18.6 |       19.8 | 0.94x |  -0.07  |
|    (1, 2048, 2048, 4) |     9.3 |       10.5 | 0.89x |  -0.02  |
|    (1, 4096, 2048, 4) |    20.4 |       22.0 | 0.93x |  -0.02  |
|    (4, 2048, 2048, 4) |    37.8 |       41.0 | 0.92x |  -0.03  |
|    (4, 4096, 2048, 4) |    73.3 |       75.7 | 0.97x |  -0.02  |
|    (8, 2048, 4096, 4) |   136.2 |      139.7 | 0.97x |  -0.03  |

**Every shape now beats upstream.** Worst 0.97x, best 0.83x.

### Opt9: use `rcp.approx.ftz.f32` instead of `div.rn.f32` in silu sigmoid

Replace `1.0 / (1.0 + exp(-pre))` (which lowers to `div.rn.f32`,
~30 cycles latency on H100 sm_90a) with `rcp.approx.ftz.f32`
(`llvm.nvvm.rcp.approx.ftz.f`, single-cycle approximate reciprocal).
Same trick the fwd-perf agent identified as their biggest single
win — silu's sigmoid backward is computed once per kNElts per chunk
per thread.

|       shape (B,D,L,W) |  mojo us | upstream us | ratio |  Δratio |
|----------------------:|---------:|------------:|------:|--------:|
|     (1, 1024,  512, 4) |     3.7 |        4.3 | 0.85x |  -0.02  |
|    (1, 1024, 2048, 4) |     5.4 |        5.9 | 0.91x |  -0.15  |
|    (1, 1024, 8192, 4) |    20.1 |       19.8 | 1.01x |  -0.07  |
|    (1, 2048, 2048, 4) |     9.5 |       10.5 | 0.91x |  -0.15  |
|    (1, 4096, 2048, 4) |    20.8 |       22.0 | 0.95x |  -0.10  |
|    (4, 2048, 2048, 4) |    39.0 |       40.9 | 0.95x |  -0.08  |
|    (4, 4096, 2048, 4) |    75.3 |       75.8 | 0.99x |  -0.09  |
|    (8, 2048, 4096, 4) |   140.4 |      139.8 | 1.00x |  -0.08  |

6 of 8 shapes now BEAT upstream. Worst-case shrunk to 1.01x.

### Opt8 (reverted): kNThreads=256

Doubling the block size to halve chunk-loop trips regressed every
shape (1.20-1.43x). The smaller shapes don't have enough work per
block to keep 256 threads busy, and the larger smem footprint
(kChunkSize doubles) cuts SM occupancy.

### Opt5 (reverted): SIMD-vectorise dx / dweight inner loops via `slice`

Replace the per-(i, k) scalar nested loops in P5 (dx) and P6 (dweight)
with one explicit `SIMD[fp32, kNElts + W - 1] combined` register
(built from `dpre` + the leading `W-1` of `dout_halo`) followed by
`width` SIMD-wide FMAs over `combined.slice[kNElts, offset=W-1-k]`.

Result: same or marginally worse on every shape (≤ +0.03 ratio
shift). The mojo compiler was apparently already vectorising the
nested scalar loops well — making it explicit added a few `vector.
extract` ops without compressing the inner loop, and burned a couple
of registers on the temporary `combined`. **Reverted.**

### Opt4 (reverted): spread post-reduce atomics across lanes 0..n-1 of warp 0

Idea: every lane of warp 0 holds the full `SIMD[fp32, n]` block-reduce
result (broadcast by the `_warp_sum_f32` butterfly). Spread the `n`
atomic-adds across lanes 0..n-1 instead of serialising them through
`tidx == 0`, so they issue in parallel.

Result: same or slightly worse on every shape (≤ +0.01 ratio shift).
The atomic-add throughput at the L2 atomic unit was apparently not the
bottleneck — they were already overlapping in flight. The extra warp
divergence (different lanes following different `if lane < width`
branches) cost slightly more than the parallel issue saved.
**Reverted.**

### Opt3 (reverted): ping-pong smem_carry to drop 4th barrier

Idea: instead of in-place carry-write to `smem_dout[0]` *after* the
halo reads (which forced a 4th `barrier()` to separate "stomp" from
"read"), use a separate ping-pong `smem_carry[2*kNElts]` keyed on
`chunk_rev & 1`. Thread 0 writes the carry in the *same* batch as the
`tidx>0` smem_dout writes; thread kNThreads-1 reads the *other* slot.
No 4th barrier needed.

Result: marginally worse on every shape (1.06x → 1.07x worst-case).
The extra `parity = chunk_rev & 1` computation + branchy stores/reads
+ the diverging warp paths (tidx==0 doing a different store, tidx==
kNThreads-1 a different load) cost slightly more than the single
saved barrier. **Reverted.**

