"""GPU forward kernel for causal_conv1d.

Mirrors upstream's `causal_conv1d_fwd.cu`. The launcher lives in
`launch.mojo`; this file holds only the kernel itself.

Design (matches upstream's tri-dao kernel):
- Input/state/output use `dtype`; weight/bias independently use
  `wdtype`. Parameter loads cast to fp32 before accumulation.
- Grid: `(dim, batch)` — one block per (B, D). Each block walks the
  full seqlen via an inner chunk loop. Mirrors upstream's
  `dim3 grid(params.batch, params.dim)`.
- Block size: `kNThreads` (=128). Per-thread element count:
  `kNElts = 16 / sizeof(dtype)` (=8 for fp16/bf16, =4 for fp32) so
  aligned rows use a 16-byte vector (`ld.global.nc.v4.b32`). Rows whose
  base pointer or outer strides do not preserve 16-byte alignment use
  element-aligned accesses, matching upstream's non-vector load path.
- Chunk size: `kNThreads * kNElts` (=1024 for fp16, =512 for fp32).
- Halo (W-1 values from previous chunk) shared via `smem_exchange`:
  slot `kNThreads-1` carries the previous chunk's tail across chunks.
  Three-barrier dance keeps thread 0 and thread kNThreads-1 from
  trampling each other's reads/writes (mirrors upstream).
- With packed sequences plus initial states, virtual positions before
  t=0 carry `seq_idx[b, 0]`; the ordinary id gate therefore exposes the
  initial state only to the first packed sequence in each batch row.
- The channel-last kernel carries the same (W-1)-row seq_idx halo beside
  its register x halo. Row-id loads are warp broadcasts (every lane asks
  for the same address), so packed inputs keep the coalesced channel
  vectors and four-row walk of the non-packed fast path.

This replaces the old design that had grid = (chunks, dim, batch): each
chunk-block re-loaded weights/bias and re-read its left-halo from
global. The new design loads weights/bias once and shares boundary
x values via smem.
"""

from std.bit import next_power_of_two
from std.gpu import block_idx, thread_idx
from max.gpu import barrier
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.math import ceildiv
from std.memory import AddressSpace, stack_allocation
from std.sys import size_of
from std.utils.index import StaticTuple
from layout import TileTensor, TensorLayout, Idx, Coord

from common import kNThreads, kNThreadsCL, kNEltsFwd
from _silu import _silu_f32


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
def fwd_kernel[
    dtype: DType,
    wdtype: DType,
    kWidth: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
    contig_inner: Bool,
    aligned_seq: Bool,
    vec_aligned: Bool,
    XLayoutType: TensorLayout,
    WLayoutType: TensorLayout,
    OLayoutType: TensorLayout,
    SLayoutType: TensorLayout,
    ILayoutType: TensorLayout,
](
    seqlen_arg: Int32,
    x: TileTensor[dtype, XLayoutType, ImmutAnyOrigin],
    weight: TileTensor[wdtype, WLayoutType, ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    seq_idx: TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin],
    initial_states: TileTensor[dtype, ILayoutType, ImmutAnyOrigin],
    output: TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin],
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
    instead of treating the out-of-range position as zero. When
    `has_seq_idx` is also set, those virtual positions carry
    `seq_idx[b, 0]`, so later packed sequences cannot read the state.

    `contig_inner`: the innermost stride of x/output is 1. Encoded by
    the dispatcher in the Layout types (`Idx[1]` in the inner-stride
    slot), so the inner-stride multiply folds out at comptime.

    `aligned_seq`: seqlen is a multiple of `kNThreads * kNElts` and every
    x/output row is 16-byte aligned. When True, the per-chunk
    bounds-checked tail path is dropped and global loads/stores use the
    16-byte vector path.

    `vec_aligned`: weaker — seqlen is a multiple of `kNElts` and every
    x/output row is 16-byte aligned. Each thread's `seq_start` is also a
    multiple of `kNElts`, so its slice is wholly in or out of bounds;
    the partial-element scalar fallback path becomes statically dead.
    This is the case for aligned shapes like `(B, D, 512, W)` where
    chunk-alignment fails (kChunkSize=1024 for fp16) but vector alignment
    still holds. The length condition mirrors upstream's `kIsVecLoad`
    switch; the row-alignment checks additionally make sliced and odd-row-
    stride tensors safe.
    """

    # Mojo 1.0 rejects `Int`/`UInt` as device kernel arguments (not
    # `DevicePassable`); take them as Int32 and widen here so the body
    # keeps its `Int` arithmetic.
    var seqlen = Int(seqlen_arg)
    comptime assert (
        TileTensor[dtype, XLayoutType, ImmutAnyOrigin].flat_rank == 3
        and TileTensor[wdtype, WLayoutType, ImmutAnyOrigin].flat_rank == 2
        and TileTensor[mut=True, dtype, OLayoutType, MutAnyOrigin].flat_rank
        == 3
        and TileTensor[DType.int32, SLayoutType, ImmutAnyOrigin].flat_rank == 2
        and TileTensor[dtype, ILayoutType, ImmutAnyOrigin].flat_rank == 3
    ), "fwd_kernel: unexpected tensor ranks (expected x/o/init=3, w/seq_idx=2)"
    comptime accum_t = DType.float32
    comptime kNElts: Int = kNEltsFwd[dtype]()
    comptime kChunkSize: Int = kNThreads * kNElts

    var tidx: Int = thread_idx.x
    var channel_id: Int = block_idx.x
    var batch_id: Int = block_idx.y

    # ---- Load weight_vals once per block (fp32 registers) ----
    var weight_vals = SIMD[accum_t, next_power_of_two(kWidth)](0)

    comptime for k in range(kWidth):
        weight_vals[k] = weight[channel_id, k].cast[accum_t]()

    # ---- Load bias once per block ----
    var bias_val: Scalar[accum_t] = 0

    comptime if has_bias:
        bias_val = bias_ptr[channel_id].cast[accum_t]()

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
                Coord(batch_id, channel_id, seq_start)
            )
        elif contig_inner and vec_aligned:
            # vec_aligned ⇒ each thread's kNElts slice is either fully
            # in-bounds or starts past `seqlen`. Either a single 16-byte
            # vec load or nothing — never partial. The else arm of the
            # `seq_start + kNElts <= seqlen` test below would be the
            # scalar fallback, but vec_aligned makes it statically dead.
            if seq_start + kNElts <= seqlen:
                x_curr = x.load[width=kNElts, alignment=16](
                    Coord(batch_id, channel_id, seq_start)
                )
        elif contig_inner:
            # Upstream's non-kIsVecLoad path uses scalar global accesses
            # for the whole row. Preserve the per-thread slice here but
            # promise only dtype alignment, so a misaligned row can never
            # lower to a faulting 16-byte transaction.
            if seq_start + kNElts <= seqlen:
                x_curr = x.load[width=kNElts, alignment=size_of[dtype]()](
                    Coord(batch_id, channel_id, seq_start)
                )
            elif seq_start < seqlen:

                comptime for i in range(kNElts):
                    var t = seq_start + i
                    if t < seqlen:
                        x_curr[i] = x[batch_id, channel_id, t]
        else:
            if seq_start < seqlen:

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

                comptime for i in range(kWidth - 1):
                    x_prev[kNElts - (kWidth - 1) + i] = initial_states[
                        batch_id, channel_id, i
                    ]

        barrier()  # all halo reads done; tidx==N-1 may now stomp slot N-1

        # Same vectorized store as above for the inter-chunk carry write.
        if tidx == kNThreads - 1:
            (smem_exchange + tidx * kNElts).store[alignment=16](x_curr)

        # Threads whose entire slice is past the seqlen (typical on partial
        # chunks like seqlen=512 with kChunkSize=1024 where threads 64..127
        # have seq_start >= 512) can't produce any valid output. They still
        # had to participate in the smem dance — but they can skip the
        # cast → conv → silu → store work below. The `continue` jumps back
        # to the loop top where the next iter's first `barrier()` syncs
        # them with the other threads. Comptime-gated on `aligned_seq=False`
        # so the (very common) aligned-fast-path stays a straight line.
        comptime if not aligned_seq:
            if seq_start >= seqlen:
                continue

        # ---- [P3] Build x_window = [x_prev || x_curr] in fp32 ----
        # We only need the last (W-1) of x_prev plus all of x_curr;
        # the compiler will dead-code-eliminate the unused slots.
        var x_vals = SIMD[accum_t, 2 * kNElts](0)

        comptime for i in range(kNElts):
            x_vals[i] = x_prev[i].cast[accum_t]()
            x_vals[kNElts + i] = x_curr[i].cast[accum_t]()

        # ---- [P4] seq_idx window (only when has_seq_idx) ----
        # Needed at positions [seq_start - (W-1) .. seq_start + kNElts - 1].
        # Out-of-range positions get -1 so the gate naturally fails,
        # except that the dual seq_idx + initial_states variant assigns
        # pre-t=0 positions the first packed sequence's id (upstream
        # v1.7.0 semantics).
        comptime kSeqWindow: Int = (kWidth - 1) + kNElts
        var seq_window = InlineArray[Int32, kSeqWindow](uninitialized=True)

        comptime if has_seq_idx:

            comptime for j in range(kSeqWindow):
                var t_j = seq_start + j - (kWidth - 1)
                if 0 <= t_j and t_j < seqlen:
                    seq_window[j] = seq_idx[batch_id, t_j]
                else:

                    comptime if has_initial_states:
                        if t_j < 0:
                            seq_window[j] = seq_idx[batch_id, 0]
                        else:
                            seq_window[j] = -1
                    else:
                        seq_window[j] = -1

        # ---- [P5] Compute out[i] = bias + sum_w weight_vals[w] * x_vals[kNElts + i - (W-1-w)] ----
        var out_vals = SIMD[accum_t, kNElts](0)

        comptime for i in range(kNElts):
            var acc: Scalar[accum_t] = bias_val
            var cur_id: Int32 = 0

            comptime if has_seq_idx:
                cur_id = seq_window[(kWidth - 1) + i]

            comptime for w in range(kWidth):
                comptime x_idx: Int = kNElts + i - (kWidth - 1 - w)
                var include: Bool = True

                comptime if has_seq_idx:
                    include = seq_window[i + w] == cur_id
                if include:
                    acc += weight_vals[w] * x_vals[x_idx]

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
                Coord(batch_id, channel_id, seq_start),
                out_vals.cast[dtype](),
            )
        elif contig_inner and vec_aligned:
            # vec_aligned ⇒ no boundary partial. Single vec store guarded
            # by `seq_start < seqlen` (or equivalently +kNElts ≤ seqlen);
            # OOB threads were already gated off by the `continue` above.
            output.store[alignment=16](
                Coord(batch_id, channel_id, seq_start),
                out_vals.cast[dtype](),
            )
        elif contig_inner:
            # Same alignment-agnostic policy as the load above.
            if seq_start + kNElts <= seqlen:
                output.store[alignment=size_of[dtype]()](
                    Coord(batch_id, channel_id, seq_start),
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreadsCL))
)
def fwd_channellast_kernel[
    dtype: DType,
    wdtype: DType,
    kWidth: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
](
    seqlen_arg: Int32,
    dim_arg: Int32,
    rows_per_block_arg: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_b_stride: UInt32,
    x_l_stride: UInt32,
    w_c_stride: UInt32,
    o_b_stride: UInt32,
    o_l_stride: UInt32,
    seq_idx_b_stride: UInt32,
    seq_idx_l_stride: UInt32,
    is_b_stride: UInt32,
    is_c_stride: UInt32,
    is_l_stride: UInt32,
):
    """Causal conv1d forward for channel-last x/out (dim contiguous,
    `x.stride(1) == 1`, seqlen strided).

    Unlike upstream's CUDA channel-last kernel (128 threads, a
    [W-1+64][64+kNElts] smem tile, two thread->work re-mappings and two
    barriers), this exploits the layout directly with no shared memory
    and no barriers: because dim is the contiguous axis, one thread owns
    `kNElts` consecutive channels (a single 16-byte vector) and walks its
    block's `kChunkLCL` seqlen rows sequentially, carrying the (W-1)-row
    halo in registers. Loads and stores are fully coalesced along dim
    (adjacent threads touch adjacent 16-byte slices of the same row) and
    x/out are each touched exactly once per element — the roofline
    minimum. The (W-1) halo rows before the chunk are re-read from
    global (or `initial_states` / zero before t=0), matching upstream's
    halo-by-reload approach. With packed sequences, the matching row-id
    halo travels beside x: a tap contributes only when its id equals the
    output row's id, padding ids produce zero, and negative-time rows use
    `seq_idx[b, 0]` only when initial states are present.

    Dispatch preconditions (gated in `_jit.py`): dim, x/out batch and
    seqlen strides are all multiples of kNElts (so every input/output
    vector access is 16-byte aligned), weight is width-contiguous, and
    the bias base is aligned to `kNElts * sizeof(wdtype)`. seq_idx is
    made contiguous by `_fn.py`; its explicit strides are still honoured.

    `rows_per_block` is runtime, chosen by the launcher: 64 rows when
    the grid is already wide enough, halving to a backend-specific floor
    (4 on discrete GPUs, 8 on Metal) for small shapes where (batch x
    C-chunks x L-chunks) would otherwise leave the GPU latency-starved.
    The trade is halo re-reads ((W-1)/rows extra x traffic) against
    occupancy — at small shapes the kernel is latency-bound, not
    bandwidth-bound, so more blocks win.

    Raw pointers + UInt32 strides instead of TileTensor: same rationale
    as `update/kernel.mojo` — with every stride dynamic there is nothing
    for TileTensor's comptime layout machinery to fold away, and the
    explicit address math keeps the prologue flat.
    """

    # Mojo 1.0 rejects `Int`/`UInt` as device kernel arguments (not
    # `DevicePassable`); take them as Int32 and widen here so the body
    # keeps its `Int` arithmetic.
    var seqlen = Int(seqlen_arg)
    var dim = Int(dim_arg)
    var rows_per_block = Int(rows_per_block_arg)
    comptime accum_t = DType.float32
    comptime kNElts: Int = kNEltsFwd[dtype]()
    # This thread still owns kNElts(dtype) channels, but the staged
    # weight/bias vector spans their wdtype representation: 8, 16, or
    # 32 bytes across the supported mixed-dtype pairs. Same-dtype stays
    # exactly the existing 16-byte vector.
    comptime kWeightVecBytes: Int = kNElts * size_of[wdtype]()
    # Channels covered by one block (one smem weight-tile row per tap).
    comptime kChunkC: Int = kNThreadsCL * kNElts

    # Grid is (L-chunks, batch, C-chunks): the seqlen-chunk count is the
    # only axis that can get huge (seqlen / rows_per_block), so it rides
    # grid.x (2^31-1 cap) — grid.y/z cap at 65535 on CUDA. Consecutive
    # blocks along x are consecutive L-chunks of the same (batch, C)
    # slice, so a block's (W-1)-row halo re-read hits rows its x-1
    # neighbour just pulled into L2.
    var tidx: Int = thread_idx.x
    var chunk_l: Int = block_idx.x
    var batch_id: Int = block_idx.y
    var chunk_c: Int = block_idx.z

    # This thread's kNElts consecutive channels.
    var c0: Int = (chunk_c * kNThreadsCL + tidx) * kNElts

    # ---- Stage the block's weight tile through smem (coalesced) ----
    # A naive per-thread read of its own taps
    # (`w_ptr[(c0+i)*w_c_stride + j]`, 32 scalar loads with a 64-byte
    # warp stride) shreds L1 sectors — ncu showed it costing ~8x
    # upstream's total load-sector count and dominating kernel time. So
    # the whole block cooperatively copies its (kChunkC x kWidth) weight
    # tile with *adjacent* threads touching *adjacent* elements, and
    # each thread then reads its taps back from smem. Layout is
    # transposed (tap-major, `[j][c_local]`) so the read-back below is a
    # conflict-free kWeightVecBytes smem vector per tap.
    var w_smem = stack_allocation[
        kWidth * kChunkC,
        Scalar[wdtype],
        address_space = AddressSpace.SHARED,
        alignment=kWeightVecBytes,
    ]()

    comptime for it in range(kNElts * kWidth):
        var e: Int = it * kNThreadsCL + tidx
        var c_local: Int = e % kChunkC
        var j: Int = e // kChunkC
        var c: Int = chunk_c * kChunkC + c_local
        w_smem[j * kChunkC + c_local] = (
            w_ptr[c * Int(w_c_stride) + j] if c < dim else Scalar[wdtype](0)
        )
    barrier()

    # No thread may exit before the staging barrier above, so the
    # out-of-range check lives here rather than at the top.
    if c0 >= dim:
        return

    var x_base = x_ptr + batch_id * Int(x_b_stride) + c0
    var o_base = o_ptr + batch_id * Int(o_b_stride) + c0

    # ---- This thread's weights: one kWeightVecBytes smem vector per tap ----
    var w_vecs = InlineArray[SIMD[accum_t, kNElts], kWidth](
        fill=SIMD[accum_t, kNElts](0)
    )

    comptime for j in range(kWidth):
        w_vecs[j] = (
            (w_smem + j * kChunkC + tidx * kNElts)
            .load[width=kNElts, alignment=kWeightVecBytes]()
            .cast[accum_t]()
        )

    # ---- Bias: one kWeightVecBytes vector (unit stride; c0 is a
    # multiple of kNElts, and c0 < dim with dim % kNElts == 0 keeps
    # the vector in bounds) ----
    var bias_vec = SIMD[accum_t, kNElts](0)

    comptime if has_bias:
        bias_vec = (
            (bias_ptr + c0)
            .load[width=kNElts, alignment=kWeightVecBytes]()
            .cast[accum_t]()
        )

    var l0: Int = chunk_l * rows_per_block
    var l_end: Int = min(l0 + rows_per_block, seqlen)

    # ---- Halo: window[j] = x row (l0 - (W-1) + j), in fp32 ----
    # Rows before t=0 come from initial_states (b, c, t + W - 1) or zero.
    # The x-row reads are aligned vector loads; the initial_states reads
    # stay scalar so arbitrary strides are fine (one-time cost of at
    # most (W-1)*kNElts scalar loads per thread).
    var window = InlineArray[SIMD[accum_t, kNElts], kWidth - 1](
        fill=SIMD[accum_t, kNElts](0)
    )
    var seq_window = InlineArray[Int32, kWidth - 1](fill=-1)

    comptime for j in range(kWidth - 1):
        var l_h: Int = l0 - (kWidth - 1) + j
        if l_h >= 0:
            window[j] = (
                (x_base + l_h * Int(x_l_stride))
                .load[width=kNElts, alignment=16]()
                .cast[accum_t]()
            )
        else:

            comptime if has_initial_states:

                comptime for i in range(kNElts):
                    window[j][i] = initial_states_ptr[
                        batch_id * Int(is_b_stride)
                        + (c0 + i) * Int(is_c_stride)
                        + (l_h + kWidth - 1) * Int(is_l_stride)
                    ].cast[accum_t]()

        comptime if has_seq_idx:
            if l_h >= 0:
                # Every lane in a warp loads the same row id, which the
                # GPU services as a broadcast transaction. It then stays
                # in registers for the same lifetime as `window[j]`.
                seq_window[j] = seq_idx_ptr[
                    batch_id * Int(seq_idx_b_stride)
                    + l_h * Int(seq_idx_l_stride)
                ]
            else:

                comptime if has_initial_states:
                    seq_window[j] = seq_idx_ptr[
                        batch_id * Int(seq_idx_b_stride)
                    ]

    # ---- Walk the chunk's rows; slide the halo in registers ----
    # Unrolled by kUnroll: each iteration issues kUnroll *independent*
    # x-row vector loads before any of them is consumed. With the
    # register-capped occupancy of this kernel (~20 warps/SM), a
    # one-row-at-a-time walk leaves each warp stalled on its single
    # outstanding load (ncu: long_scoreboard dominant); 4 loads in
    # flight per warp quarters that. The window shift also folds into
    # register renaming instead of per-row vector moves.
    # The epilogue's window shift below indexes xv[kUnroll-kWidth+1+j],
    # so this kernel requires kWidth-1 <= kUnroll (a negative comptime
    # index fails the InlineArray bounds constraint at mojo-build time).
    # The dispatcher (`channel_last` in fwd/_jit.py) therefore routes
    # width > kUnroll+1 (= 5) to the generic kernel — keep the two in
    # sync if kUnroll changes.
    comptime kUnroll: Int = 4

    var l: Int = l0
    while l + kUnroll <= l_end:
        var xv = InlineArray[SIMD[accum_t, kNElts], kUnroll](
            fill=SIMD[accum_t, kNElts](0)
        )
        var seqv = InlineArray[Int32, kUnroll](fill=-1)

        comptime for u in range(kUnroll):
            xv[u] = (
                (x_base + (l + u) * Int(x_l_stride))
                .load[width=kNElts, alignment=16]()
                .cast[accum_t]()
            )

            comptime if has_seq_idx:
                seqv[u] = seq_idx_ptr[
                    batch_id * Int(seq_idx_b_stride)
                    + (l + u) * Int(seq_idx_l_stride)
                ]

        comptime for u in range(kUnroll):
            var acc = bias_vec
            var cur_id: Int32 = 0

            comptime if has_seq_idx:
                cur_id = seqv[u]

            comptime for j in range(kWidth):
                # Tap j of output row (l+u) reads input row
                # (l+u) - (kWidth-1) + j: still in the carried window
                # when that offset is negative, else one of this
                # iteration's fresh rows.
                comptime s = u - (kWidth - 1) + j
                comptime if s < 0:

                    comptime if has_seq_idx:
                        if seq_window[kWidth - 1 + s] == cur_id:
                            acc = w_vecs[j].fma(
                                window[kWidth - 1 + s], acc
                            )
                    else:
                        acc = w_vecs[j].fma(window[kWidth - 1 + s], acc)
                else:

                    comptime if has_seq_idx:
                        if seqv[s] == cur_id:
                            acc = w_vecs[j].fma(xv[s], acc)
                    else:
                        acc = w_vecs[j].fma(xv[s], acc)

            comptime if apply_silu:

                comptime for i in range(kNElts):
                    acc[i] = _silu_f32(Float32(acc[i]))

            comptime if has_seq_idx:
                if cur_id < 0:
                    acc = 0

            (o_base + (l + u) * Int(o_l_stride)).store[alignment=16](
                acc.cast[dtype]()
            )

        comptime for j in range(kWidth - 1):
            window[j] = xv[kUnroll - (kWidth - 1) + j]

            comptime if has_seq_idx:
                seq_window[j] = seqv[kUnroll - (kWidth - 1) + j]
        l += kUnroll

    # Remainder rows (< kUnroll of them), one at a time.
    while l < l_end:
        var x_now = (
            (x_base + l * Int(x_l_stride))
            .load[width=kNElts, alignment=16]()
            .cast[accum_t]()
        )

        var acc = bias_vec
        var cur_id: Int32 = 0

        comptime if has_seq_idx:
            cur_id = seq_idx_ptr[
                batch_id * Int(seq_idx_b_stride)
                + l * Int(seq_idx_l_stride)
            ]

        comptime for j in range(kWidth - 1):

            comptime if has_seq_idx:
                if seq_window[j] == cur_id:
                    acc = w_vecs[j].fma(window[j], acc)
            else:
                acc = w_vecs[j].fma(window[j], acc)

        # x_now and the output row are the same position, so their ids
        # necessarily match (including padding, which is zeroed after
        # bias/silu below).
        acc = w_vecs[kWidth - 1].fma(x_now, acc)

        comptime if apply_silu:

            comptime for i in range(kNElts):
                acc[i] = _silu_f32(Float32(acc[i]))

        comptime if has_seq_idx:
            if cur_id < 0:
                acc = 0

        (o_base + l * Int(o_l_stride)).store[alignment=16](
            acc.cast[dtype]()
        )

        comptime for j in range(kWidth - 2):
            window[j] = window[j + 1]

            comptime if has_seq_idx:
                seq_window[j] = seq_window[j + 1]
        window[kWidth - 2] = x_now

        comptime if has_seq_idx:
            seq_window[kWidth - 2] = cur_id
        l += 1
