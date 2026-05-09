"""Pure-mojo CPU forward for causal_conv1d.

No upstream analogue — this exists so the package works on a machine
without a GPU without forcing users to `pip install causal-conv1d`
(which needs a C++ toolchain to source-build). The GPU kernels in
`causal_conv1d_fwd.mojo` / `causal_conv1d_bwd.mojo` are the real
product; the CPU paths are the slow fallback.

Pattern follows max/kernels/src/state_space/causal_conv1d.mojo:
parallelise over (batch, channel) work items via `sync_parallelize`.
Each worker pre-loads its row of weights into a register, then walks
seqlen.
"""

from std.algorithm import sync_parallelize

from causal_conv1d_common import _silu_f32


fn fwd_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    has_seq_idx: Bool,
    has_initial_states: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    initial_states_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    seq_idx_b_stride: Int,
    seq_idx_l_stride: Int,
    initial_states_b_stride: Int,
    initial_states_c_stride: Int,
    initial_states_l_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    """Causal conv1d forward, CPU path.

    Comptime params:
        has_bias: load `bias_ptr[d]` per channel, or skip and use 0.
        has_seq_idx: gate historical reads on `seq_idx[b, src_t] ==
            seq_idx[b, t]`; force output to 0 when `seq_idx[b, t] < 0`
            (padding).
        apply_silu: apply silu (= swish) on the output, or skip.
    When the gate is False, the corresponding pointer is never
    dereferenced — caller may pass null from the Python wrapper.
    """
    alias accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var bias_v: Scalar[accum_t] = 0

        @parameter
        if has_bias:
            bias_v = bias_ptr[d].cast[accum_t]()

        var weights = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            weights[k] = weight_ptr[
                d * weight_c_stride + k * weight_w_stride
            ].cast[accum_t]()

        var x_base = b * x_batch_stride + d * x_c_stride
        var out_base = b * out_batch_stride + d * out_c_stride
        var seq_idx_base: Int = b * seq_idx_b_stride
        var initial_states_base: Int = (
            b * initial_states_b_stride + d * initial_states_c_stride
        )

        for t in range(seqlen):
            var pre: Scalar[accum_t] = bias_v

            var cur_id: Int32 = 0

            @parameter
            if has_seq_idx:
                cur_id = seq_idx_ptr[seq_idx_base + t * seq_idx_l_stride]

            @parameter
            for k in range(width):
                var src_t = t + k - (width - 1)
                if src_t >= 0:
                    var include: Bool = True

                    @parameter
                    if has_seq_idx:
                        var src_id: Int32 = seq_idx_ptr[
                            seq_idx_base + src_t * seq_idx_l_stride
                        ]
                        include = src_id == cur_id
                    if include:
                        pre += (
                            weights[k]
                            * x_ptr[x_base + src_t * x_l_stride].cast[accum_t]()
                        )
                else:

                    @parameter
                    if has_initial_states:
                        # src_t in [-(W-1), 0); index 0..W-2 of initial_states.
                        var is_idx: Int = src_t + (width - 1)
                        pre += (
                            weights[k]
                            * initial_states_ptr[
                                initial_states_base
                                + is_idx * initial_states_l_stride
                            ].cast[accum_t]()
                        )

            var out_v: Scalar[accum_t]

            @parameter
            if apply_silu:
                out_v = _silu_f32(pre)
            else:
                out_v = pre

            @parameter
            if has_seq_idx:
                if cur_id < 0:
                    out_v = 0

            output_ptr[out_base + t * out_l_stride] = out_v.cast[dtype]()

    sync_parallelize[process_bc](batch * dim)
