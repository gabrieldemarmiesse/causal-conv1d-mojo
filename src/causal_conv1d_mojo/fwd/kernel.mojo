"""GPU forward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_fwd.cu`. The launcher lives in
`dispatch.mojo`; this file holds only the kernel itself.

Design (matches upstream's tri-dao kernel):
- Grid: `(dim, batch)` — one block per (B, D). Each block walks the
  full seqlen via an inner chunk loop. Mirrors upstream's
  `dim3 grid(params.batch, params.dim)`.
- Block size: `kNThreads` (=128). Per-thread element count:
  `kNElts = 16 / sizeof(dtype)` (=8 for fp16/bf16, =4 for fp32) so
  every per-chunk load is a 16-byte vector (`ld.global.nc.v4.b32`).
- Chunk size: `kNThreads * kNElts` (=1024 for fp16, =512 for fp32).
- Halo (W-1 values from previous chunk) shared via `smem_exchange`:
  slot `kNThreads-1` carries the previous chunk's tail across chunks.
  Three-barrier dance keeps thread 0 and thread kNThreads-1 from
  trampling each other's reads/writes (mirrors upstream).

This replaces the old design that had grid = (chunks, dim, batch): each
chunk-block re-loaded weights/bias and re-read its left-halo from
global. The new design loads weights/bias once and shares boundary
x values via smem.
"""

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from layout import TileTensor, TensorLayout, Idx, Coord

from common import _silu_f32, kNThreads, kNEltsFwd


def fwd_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    OLayoutType: TensorLayout,
](
    seqlen: Int,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[dtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output: TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin],
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
) where (
    TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
    and TileTensor[dtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
    and TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin].flat_rank == 3
):
    """Causal conv1d forward, GPU. One block per (B, D); walks seqlen.

    `has_seq_idx`: `seq_idx_ptr` is a `(B, L)` int32 tensor of sequence
    ids. For each output position `t`, historical reads from `src_t < t`
    are masked to 0 unless `seq_idx[b, src_t] == seq_idx[b, t]`, so
    packed mini-batches don't bleed across sequence boundaries.
    `seq_idx[b, t] < 0` marks a padding position — its output is forced
    to 0.

    `has_initial_states`: `initial_states_ptr` is a `(B, D, W-1)` tensor
    that supplies the historical context before `t = 0`. For
    `src_t in [-(W-1), 0)`, we read `initial_states[b, c, src_t + W - 1]`
    instead of treating the out-of-range position as zero. Mutually
    exclusive with `has_seq_idx`.

    `contig_inner`: the innermost stride of x/output is 1. Encoded by
    the dispatcher in the Layout types (`Idx[1]()` in the inner-stride
    slot), so the inner-stride multiply folds out at comptime.

    `aligned_seq`: seqlen is a multiple of `kNThreads * kNElts`. When
    True, the per-chunk bounds-checked tail path is dropped — halves
    the kernel's compiled code and eliminates predicated stores.
    """
    comptime accum_t = DType.float32
    comptime kNElts: Int = kNEltsFwd[dtype]()
    comptime kChunkSize: Int = kNThreads * kNElts

    var tidx: Int = thread_idx.x
    var channel_id: Int = block_idx.x
    var batch_id: Int = block_idx.y

    # ---- Load weights once per block (fp32 registers) ----
    var weights = SIMD[accum_t, width](0)

    comptime for k in range(width):
        weights[k] = weight[channel_id, k].cast[accum_t]()

    # ---- Load bias once per block ----
    var cur_bias: Scalar[accum_t] = 0

    comptime if has_bias:
        cur_bias = bias_ptr[channel_id].cast[accum_t]()

    # ---- Smem exchange buffer for (W-1) halo across chunks ----
    # Slot i holds thread i's last kNElts x values; we read slot
    # (tidx-1) % kNThreads to obtain "previous kNElts elements".
    # Slot kNThreads-1 doubles as the inter-chunk carry (last kNElts
    # of the previous chunk, initialised to 0 on the first chunk).
    var smem_exchange = stack_allocation[
        kNThreads * kNElts, dtype, address_space=AddressSpace.SHARED
    ]()

    # Thread 0 zero-init slot kNThreads-1 — it serves as the "previous
    # chunk's tail" before the first chunk's halo barrier dance.
    # Single 16-byte aligned store: kNElts*sizeof(dtype) == 16.
    if tidx == 0:
        (smem_exchange + (kNThreads - 1) * kNElts).store[alignment=16](
            SIMD[dtype, kNElts](0)
        )

    var seq_idx_base: Int = batch_id * seq_idx_b_stride
    var initial_states_base: Int = (
        batch_id * initial_states_b_stride
        + channel_id * initial_states_c_stride
    )

    # Note: no pre-loop barrier needed for the init write — the loop's
    # first `barrier()` (at the top of each iteration) already serves as
    # the visibility barrier for chunk 0's first read of slot N-1.

    var n_chunks: Int = ceildiv(seqlen, kChunkSize)

    for chunk in range(n_chunks):
        var chunk_start: Int = chunk * kChunkSize
        var seq_start: Int = chunk_start + tidx * kNElts

        # ---- [P1] Load this thread's kNElts of x from global ----
        # x_curr: this thread's slice. x_prev: the previous thread's
        # tail (obtained from smem below).
        # 16-byte LDG fast path: `aligned_seq` + `contig_inner` ⇒ no
        # bounds check, single vec load.
        var x_curr = SIMD[dtype, kNElts](0)

        comptime if contig_inner and aligned_seq:
            x_curr = x.load[width=kNElts, alignment=16](
                Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
            )
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                x_curr = x.load[width=kNElts, alignment=16](
                    Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start))
                )
            else:

                comptime for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        x_curr[i] = x[batch_id, channel_id, t]
        else:

            comptime for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    x_curr[i] = x[batch_id, channel_id, t]

        # ---- [P2] Halo dance ----
        # Slot kNThreads-1 holds the prev chunk's tail (initialised to
        # 0 before the loop, then re-written by tidx==kNThreads-1 at
        # the end of each iter). We need to:
        #   1. let tidx==0 read slot kNThreads-1 (prev chunk's tail)
        #   2. then all tidx>0 read slot tidx-1 (own neighbour's data)
        # Upstream's pattern (matched here):
        #   sync; if(tidx<N-1) smem[tidx]=x_curr; sync;
        #   x_prev = smem[(tidx-1) % N]; sync;
        #   if(tidx==N-1) smem[tidx]=x_curr;
        barrier()  # complete prev iter / pre-loop init

        # Vectorized smem store: tidx*kNElts*sizeof(dtype) is 16-byte
        # aligned (kNElts*sizeof(dtype) = 16), so a single st.shared.v4.b32
        # instead of kNElts scalar st.shared.b16/b32. Matches the bwd
        # kernel's smem-vec optimisation.
        if tidx < kNThreads - 1:
            (smem_exchange + tidx * kNElts).store[alignment=16](x_curr)

        barrier()  # writes from tidx<N-1 visible; slot N-1 still holds
                  # prev chunk's tail (or 0 on first chunk)

        var prev_tidx = tidx - 1 if tidx > 0 else (kNThreads - 1)
        # Vectorized smem load: same alignment argument as the store path.
        var x_prev = (smem_exchange + prev_tidx * kNElts).load[
            width=kNElts, alignment=16
        ]()

        # On chunk 0, tidx 0 needs x_prev populated with initial_states
        # (for the W-1 trailing slots). Otherwise x_prev stays at 0.
        comptime if has_initial_states:
            if tidx == 0 and chunk == 0:

                comptime for i in range(width - 1):
                    x_prev[kNElts - (width - 1) + i] = initial_states_ptr[
                        initial_states_base + i * initial_states_l_stride
                    ]

        barrier()  # all halo reads done; tidx==N-1 may now stomp slot N-1

        # Same vectorized store as above for the inter-chunk carry write.
        if tidx == kNThreads - 1:
            (smem_exchange + tidx * kNElts).store[alignment=16](x_curr)

        # ---- [P3] Build x_window = [x_prev || x_curr] in fp32 ----
        # We only need the last (W-1) of x_prev plus all of x_curr;
        # the compiler will dead-code-eliminate the unused slots.
        var x_vals = SIMD[accum_t, 2 * kNElts](0)

        comptime for i in range(kNElts):
            x_vals[i] = x_prev[i].cast[accum_t]()
            x_vals[kNElts + i] = x_curr[i].cast[accum_t]()

        # ---- [P4] seq_idx window (only when has_seq_idx) ----
        # Needed at positions [seq_start - (W-1) .. seq_start + kNElts - 1].
        # Out-of-range positions get -1 so the gate naturally fails.
        comptime kSeqWindow: Int = (width - 1) + kNElts
        var seq_window = InlineArray[Int32, kSeqWindow](uninitialized=True)

        comptime if has_seq_idx:

            comptime for j in range(kSeqWindow):
                var t_j = seq_start + j - (width - 1)
                if 0 <= t_j and t_j < seqlen:
                    seq_window[j] = seq_idx_ptr[
                        seq_idx_base + t_j * seq_idx_l_stride
                    ]
                else:
                    seq_window[j] = -1

        # ---- [P5] Compute out[i] = bias + sum_w weights[w] * x_vals[kNElts + i - (W-1-w)] ----
        var out_vals = SIMD[accum_t, kNElts](0)

        comptime for i in range(kNElts):
            var acc: Scalar[accum_t] = cur_bias
            var cur_id: Int32 = 0

            comptime if has_seq_idx:
                cur_id = seq_window[(width - 1) + i]

            comptime for w in range(width):
                comptime x_idx: Int = kNElts + i - (width - 1 - w)
                var include: Bool = True

                comptime if has_seq_idx:
                    include = seq_window[i + w] == cur_id
                if include:
                    acc += weights[w] * x_vals[x_idx]

            comptime if apply_silu:
                acc = _silu_f32(Float32(acc))

            # Padding tokens (seq_idx < 0) → out = 0 (after silu/bias).
            comptime if has_seq_idx:
                if cur_id < 0:
                    acc = 0

            out_vals[i] = acc

        # ---- [P6] Store out_vals to global ----
        comptime if contig_inner and aligned_seq:
            output.store[alignment=16](
                Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start)),
                out_vals.cast[dtype](),
            )
        elif contig_inner:
            if chunk_start + kChunkSize <= seqlen:
                output.store[alignment=16](
                    Coord(Idx(batch_id), Idx(channel_id), Idx(seq_start)),
                    out_vals.cast[dtype](),
                )
            else:

                comptime for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        output[batch_id, channel_id, t] = out_vals[i].cast[
                            dtype
                        ]()
        else:

            comptime for i in range(kNElts):
                var t = seq_start + i
                if t < seqlen:
                    output[batch_id, channel_id, t] = out_vals[i].cast[dtype]()
