"""Dedicated channel-last causal_conv1d backward GPU kernel.

This mirrors upstream's `causal_conv1d_channellast_bwd_kernel`: global
loads and stores are 16-byte vectors along the contiguous channel axis,
then shared memory transposes the `(L, C)` tile so one thread can own one
channel over a run of rows. That keeps only one channel's weights and
dweight partials in registers while preserving coalesced global traffic.

The dweight/dbias flush is intentionally isolated in
`_reduce_channellast_grads`; a future deterministic variant can replace
that one atomic-add site with per-(batch, L-chunk) workspace stores.
"""

from std.atomic import Atomic, Ordering
from std.bit import log2_floor, next_power_of_two
from std.gpu import block_idx, thread_idx
from max.gpu import barrier
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.math import exp
from std.memory import AddressSpace, stack_allocation
from std.sys import inlined_assembly, size_of
from std.sys.info import is_nvidia_gpu
from std.utils.index import StaticTuple

from common import kChunkCBwdCL_for, kNThreads
from kernel import _rcp_approx_f32, _shfl_xor_f32, kAtomicScope


@always_inline
def _atomic_add_f32(
    ptr: UnsafePointer[Float32, MutAnyOrigin], value: Float32
):
    """Native relaxed FP32 atomic add on NVIDIA; portable elsewhere.

    `Atomic[Float32].fetch_add` expands to a compare-and-swap retry loop
    in this Mojo toolchain. Inline PTX selects CUDA's one-transaction
    relaxed device-scope atomic, which matters here because every L chunk
    flushes one partial per channel. HIP/Metal keep the portable version.
    """
    comptime if is_nvidia_gpu():
        _ = inlined_assembly[
            "atom.relaxed.gpu.global.add.f32 $0, [$1], $2;",
            Float32,
            constraints="=f,l,f,~{memory}",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), value)
    else:
        _ = Atomic[DType.float32, scope=kAtomicScope].fetch_add[
            ordering=Ordering.RELAXED
        ](ptr, value)


@always_inline
def _subgroup_sum_f32[group_size: Int](value: Float32) -> Float32:
    """Sum adjacent power-of-two lane groups (2 for fp16, 4 for fp32)."""
    var result = value
    comptime for i in range(log2_floor(group_size)):
        result += _shfl_xor_f32(result, UInt32(1 << i))
    return result


@always_inline
def _reduce_channellast_grads[
    width: Int,
    has_bias: Bool,
    threads_per_channel: Int,
](
    channel: Int,
    dim: Int,
    channel_lane: Int,
    local_dweight: SIMD[DType.float32, next_power_of_two(width)],
    local_dbias: Float32,
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Reduce one L-chunk's channel gradients and flush them globally.

    This is the only dweight/dbias output site in the channel-last kernel.
    Non-deterministic variants use the same relaxed device-scope atomics as
    upstream CUDA `atomicAdd` and the existing generic Mojo kernel.
    """
    var dweight = SIMD[DType.float32, next_power_of_two(width)](0)

    comptime for w in range(width):
        dweight[w] = _subgroup_sum_f32[threads_per_channel](
            local_dweight[w]
        )

    var dbias = _subgroup_sum_f32[threads_per_channel](local_dbias)
    if channel_lane == 0 and channel < dim:

        comptime for w in range(width):
            _atomic_add_f32(
                dweight_acc_ptr + channel * width + w,
                dweight[w],
            )

        comptime if has_bias:
            _atomic_add_f32(dbias_acc_ptr + channel, dbias)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
def bwd_channellast_kernel[
    dtype: DType,
    wdtype: DType,
    kChunkL: Int,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
](
    seqlen_arg: Int32,
    dim_arg: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[wdtype], MutAnyOrigin],
    dout_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dx_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dweight_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dbias_acc_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dinitial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_b_stride: UInt32,
    x_l_stride: UInt32,
    weight_c_stride: UInt32,
    weight_w_stride: UInt32,
    dout_b_stride: UInt32,
    dout_l_stride: UInt32,
    dx_b_stride: UInt32,
    dx_l_stride: UInt32,
    seq_idx_b_stride: UInt32,
    seq_idx_l_stride: UInt32,
    initial_states_b_stride: UInt32,
    initial_states_c_stride: UInt32,
    initial_states_l_stride: UInt32,
    dinitial_states_b_stride: UInt32,
    dinitial_states_c_stride: UInt32,
    dinitial_states_l_stride: UInt32,
):
    """Fused backward for channel-last x/dout/dx (channel stride one).

    A 128-thread block first issues 16-byte vectors along C into padded
    shared-memory rows for x and dout. Threads then remap from load
    coordinates `(L-row, C-vector)` to compute coordinates `(channel,
    L-run)`: fp16/bf16 use two threads per channel and fp32 uses four.
    Each compute thread recomputes dpre, accumulates dweight/dbias, and
    forms dx for its one channel. A second shared-memory transpose makes
    dx stores 16-byte channel vectors again.

    Tiles carry W-1 x rows on both sides and W-1 dout rows on the right.
    The left edge reads initial_states (or zero), with virtual negative
    positions assigned `seq_idx[b, 0]`; the right halo supplies the
    anti-causal dx terms. Padding ids (<0) force dpre to zero. All global
    strides are raw UInt32 element strides so CUDA/HIP/Metal compile the
    same kernel without TileTensor address-packing overhead.

    Dispatch preconditions live in `_jit.py`: width <= 5, channel stride
    one, dim/outer strides compatible with 16-byte vectors, and aligned
    x/dout/dx/bias base pointers. Unsupported layouts use bwd_full_kernel.
    """

    # Mojo 1.0 rejects `Int`/`UInt` as device kernel arguments (not
    # `DevicePassable`); take them as Int32 and widen here so the body
    # keeps its `Int` arithmetic.
    var seqlen = Int(seqlen_arg)
    var dim = Int(dim_arg)
    comptime kNElts: Int = 16 // size_of[dtype]()
    comptime kChunkC: Int = kChunkCBwdCL_for[dtype]()
    comptime kLoadThreadsPerRow: Int = kChunkC // kNElts
    comptime kLoadRows: Int = kNThreads // kLoadThreadsPerRow
    comptime kNLoads: Int = kChunkL // kLoadRows
    comptime kLPerThread: Int = kChunkL * kChunkC // kNThreads
    comptime kThreadsPerChannel: Int = kChunkL // kLPerThread
    comptime kSmemStride: Int = kChunkC + kNElts
    comptime kXRows: Int = kChunkL + 2 * (width - 1)
    comptime kDoutRows: Int = kChunkL + width - 1
    comptime kSeqWindow: Int = kLPerThread + 2 * (width - 1)

    var tid: Int = thread_idx.x
    var chunk_l: Int = block_idx.x
    var batch_id: Int = block_idx.y
    var chunk_c: Int = block_idx.z
    var chunk_l_start: Int = chunk_l * kChunkL
    var chunk_c_start: Int = chunk_c * kChunkC

    # Load mapping: eight adjacent threads cover one 128-byte C span.
    var load_l: Int = tid // kLoadThreadsPerRow
    var load_c_vec: Int = tid % kLoadThreadsPerRow
    var c_vec: Int = chunk_c_start + load_c_vec * kNElts

    var x_smem = stack_allocation[
        kXRows * kSmemStride,
        dtype,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()
    var dout_smem = stack_allocation[
        kDoutRows * kSmemStride,
        dtype,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    # Main L tile. Every shared slot is initialized, including the tail
    # chunk and the final partial C tile, so compute threads need no bounds
    # branches while reading the transpose.
    comptime for load in range(kNLoads):
        var row: Int = load * kLoadRows + load_l
        var global_l: Int = chunk_l_start + row
        var x_vec = SIMD[dtype, kNElts](0)
        var dout_vec = SIMD[dtype, kNElts](0)
        if global_l < seqlen and c_vec < dim:
            x_vec = (
                x_ptr
                + batch_id * Int(x_b_stride)
                + global_l * Int(x_l_stride)
                + c_vec
            ).load[width=kNElts, alignment=16]()
            dout_vec = (
                dout_ptr
                + batch_id * Int(dout_b_stride)
                + global_l * Int(dout_l_stride)
                + c_vec
            ).load[width=kNElts, alignment=16]()
        (x_smem + (width - 1 + row) * kSmemStride + load_c_vec * kNElts).store[
            alignment=16
        ](x_vec)
        (dout_smem + row * kSmemStride + load_c_vec * kNElts).store[
            alignment=16
        ](dout_vec)

    # Left x halo and right dout halo. The optional right x halo is only
    # needed to recompute silu' for the dout rows consumed by dx.
    if load_l < width - 1:
        var left_l: Int = chunk_l_start + load_l - (width - 1)
        var right_l: Int = chunk_l_start + kChunkL + load_l
        var x_left = SIMD[dtype, kNElts](0)
        var dout_right = SIMD[dtype, kNElts](0)

        if left_l >= 0 and left_l < seqlen and c_vec < dim:
            x_left = (
                x_ptr
                + batch_id * Int(x_b_stride)
                + left_l * Int(x_l_stride)
                + c_vec
            ).load[width=kNElts, alignment=16]()
        elif left_l < 0 and chunk_l == 0 and c_vec < dim:

            comptime if has_initial_states:

                comptime for i in range(kNElts):
                    x_left[i] = initial_states_ptr[
                        batch_id * Int(initial_states_b_stride)
                        + (c_vec + i) * Int(initial_states_c_stride)
                        + load_l * Int(initial_states_l_stride)
                    ]

        if right_l < seqlen and c_vec < dim:
            dout_right = (
                dout_ptr
                + batch_id * Int(dout_b_stride)
                + right_l * Int(dout_l_stride)
                + c_vec
            ).load[width=kNElts, alignment=16]()

        (x_smem + load_l * kSmemStride + load_c_vec * kNElts).store[
            alignment=16
        ](x_left)
        (dout_smem + (kChunkL + load_l) * kSmemStride + load_c_vec * kNElts).store[
            alignment=16
        ](dout_right)

        comptime if apply_silu:
            var x_right = SIMD[dtype, kNElts](0)
            if right_l < seqlen and c_vec < dim:
                x_right = (
                    x_ptr
                    + batch_id * Int(x_b_stride)
                    + right_l * Int(x_l_stride)
                    + c_vec
                ).load[width=kNElts, alignment=16]()
            (
                x_smem
                + (width - 1 + kChunkL + load_l) * kSmemStride
                + load_c_vec * kNElts
            ).store[alignment=16](x_right)

    barrier()

    # Compute mapping: adjacent lane groups own consecutive runs of one
    # channel. This is the register-pressure-saving transpose at the heart
    # of upstream's channel-last backward design.
    var channel: Int = chunk_c_start + tid // kThreadsPerChannel
    var channel_lane: Int = tid % kThreadsPerChannel
    var row_start: Int = channel_lane * kLPerThread
    var bias: Float32 = 0
    var weights = SIMD[DType.float32, next_power_of_two(width)](0)
    if channel < dim:

        comptime if has_bias:
            bias = bias_ptr[channel].cast[DType.float32]()

        comptime for w in range(width):
            weights[w] = weight_ptr[
                channel * Int(weight_c_stride) + w * Int(weight_w_stride)
            ].cast[DType.float32]()

    var x_vals = InlineArray[Float32, kLPerThread + 2 * (width - 1)](
        fill=0
    )
    var dpre = InlineArray[Float32, kLPerThread + width - 1](fill=0)

    comptime for i in range(kLPerThread + width - 1):
        x_vals[i] = x_smem[
            (row_start + i) * kSmemStride + (channel - chunk_c_start)
        ].cast[DType.float32]()
        dpre[i] = dout_smem[
            (row_start + i) * kSmemStride + (channel - chunk_c_start)
        ].cast[DType.float32]()

    comptime if apply_silu:

        comptime for i in range(
            kLPerThread + width - 1,
            kLPerThread + 2 * (width - 1),
        ):
            x_vals[i] = x_smem[
                (row_start + i) * kSmemStride + (channel - chunk_c_start)
            ].cast[DType.float32]()

    var seq_ids = InlineArray[Int32, kSeqWindow](fill=-1)

    comptime if has_seq_idx:

        comptime for i in range(kSeqWindow):
            var global_l: Int = (
                chunk_l_start + row_start + i - (width - 1)
            )
            if global_l >= 0 and global_l < seqlen:
                seq_ids[i] = seq_idx_ptr[
                    batch_id * Int(seq_idx_b_stride)
                    + global_l * Int(seq_idx_l_stride)
                ]
            elif global_l < 0:

                comptime if has_initial_states:
                    seq_ids[i] = seq_idx_ptr[
                        batch_id * Int(seq_idx_b_stride)
                    ]

    # Recompute dpre for this run and its W-1 right halo. Out-of-range
    # dout was staged as zero. Packed padding additionally forces dpre=0.
    comptime for i in range(kLPerThread + width - 1):
        var cur_id: Int32 = 0

        comptime if has_seq_idx:
            cur_id = seq_ids[i + width - 1]

        comptime if apply_silu:
            var pre: Float32 = bias

            comptime for w in range(width):
                var include: Bool = True

                comptime if has_seq_idx:
                    include = seq_ids[i + w] == cur_id
                if include:
                    pre += weights[w] * x_vals[i + w]

            var sigmoid = _rcp_approx_f32(1.0 + exp(-pre))
            dpre[i] *= sigmoid * (1.0 + pre * (1.0 - sigmoid))

        comptime if has_seq_idx:
            if cur_id < 0:
                dpre[i] = 0

    var local_dweight = SIMD[DType.float32, next_power_of_two(width)](0)
    var local_dbias: Float32 = 0

    comptime for i in range(kLPerThread):
        local_dbias += dpre[i]

    comptime for w in range(width):

        comptime for i in range(kLPerThread):
            var include: Bool = True

            comptime if has_seq_idx:
                include = seq_ids[i + w] == seq_ids[i + width - 1]
            if include:
                local_dweight[w] += x_vals[i + w] * dpre[i]

    var dx_vals = InlineArray[Float32, kLPerThread](fill=0)

    comptime for i in range(kLPerThread):
        var cur_id: Int32 = 0

        comptime if has_seq_idx:
            cur_id = seq_ids[i + width - 1]

        comptime for w in range(width):
            var include: Bool = True

            comptime if has_seq_idx:
                include = seq_ids[i + width - 1 + w] == cur_id
            if include:
                dx_vals[i] += weights[width - 1 - w] * dpre[i + w]

    # Initial-state gradients use only chunk zero's first L-run. Honor
    # arbitrary outer/inner strides; these W-1 scalar stores are negligible
    # next to the vectorized main dx tile.
    comptime if has_initial_states:
        if chunk_l == 0 and channel_lane == 0 and channel < dim:

            comptime for i in range(width - 1):
                var dinit: Float32 = 0

                comptime for w in range(width):

                    comptime if w <= i:
                        var include: Bool = True

                        comptime if has_seq_idx:
                            include = seq_ids[i] == seq_ids[
                                width - 1 + i - w
                            ]
                        if include:
                            dinit += weights[w] * dpre[i - w]
                dinitial_states_ptr[
                    batch_id * Int(dinitial_states_b_stride)
                    + channel * Int(dinitial_states_c_stride)
                    + i * Int(dinitial_states_l_stride)
                ] = dinit.cast[dtype]()

    _reduce_channellast_grads[
        width,
        has_bias,
        kThreadsPerChannel,
    ](
        channel,
        dim,
        channel_lane,
        local_dweight,
        local_dbias,
        dweight_acc_ptr,
        dbias_acc_ptr,
    )

    # Do not overwrite x_smem until every compute thread has finished its
    # read-only x window; then transpose dx back to 16-byte channel vectors.
    barrier()

    comptime for i in range(kLPerThread):
        x_smem[
            (width - 1 + row_start + i) * kSmemStride
            + (channel - chunk_c_start)
        ] = dx_vals[i].cast[dtype]()

    barrier()

    comptime for load in range(kNLoads):
        var row: Int = load * kLoadRows + load_l
        var global_l: Int = chunk_l_start + row
        if global_l < seqlen and c_vec < dim:
            var dx_vec = (
                x_smem
                + (width - 1 + row) * kSmemStride
                + load_c_vec * kNElts
            ).load[width=kNElts, alignment=16]()
            (
                dx_ptr
                + batch_id * Int(dx_b_stride)
                + global_l * Int(dx_l_stride)
                + c_vec
            ).store[alignment=16](dx_vec)
