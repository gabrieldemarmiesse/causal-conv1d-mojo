"""Pure-mojo CPU single-step update kernel for causal_conv1d.

No upstream analogue — this exists so the package's autoregressive-
decode path works on a machine without a GPU. The real product is the
GPU kernel in `causal_conv1d_update.mojo`.

Algorithm matches the GPU kernel: shift `conv_state` left by `seqlen`,
read the trailing `width-1` historical values, walk the new x and emit
outputs while writing the new x into the state's tail.
"""

from std.algorithm import sync_parallelize

from causal_conv1d_common import _silu_f32


fn update_kernel_cpu[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
](
    batch: Int,
    dim: Int,
    seqlen: Int,
    state_len: Int,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    weight_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    conv_state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_batch_stride: Int,
    x_c_stride: Int,
    x_l_stride: Int,
    weight_c_stride: Int,
    weight_w_stride: Int,
    state_batch_stride: Int,
    state_c_stride: Int,
    state_l_stride: Int,
    out_batch_stride: Int,
    out_c_stride: Int,
    out_l_stride: Int,
):
    alias accum_t = DType.float32

    @parameter
    fn process_bc(bc_idx: Int):
        var b = bc_idx // dim
        var d = bc_idx % dim

        var x_base = b * x_batch_stride + d * x_c_stride
        var out_base = b * out_batch_stride + d * out_c_stride
        var state_base = b * state_batch_stride + d * state_c_stride

        var weights = SIMD[accum_t, width](0)

        @parameter
        for k in range(width):
            weights[k] = weight_ptr[
                d * weight_c_stride + k * weight_w_stride
            ].cast[accum_t]()

        var bias_v: Scalar[accum_t] = 0

        @parameter
        if has_bias:
            bias_v = bias_ptr[d].cast[accum_t]()

        var advance_len = seqlen

        # Phase 1: shift state left by `seqlen`.
        for i in range(state_len - advance_len - (width - 1)):
            conv_state_ptr[state_base + i * state_l_stride] = conv_state_ptr[
                state_base + (i + advance_len) * state_l_stride
            ]

        # Phase 2: read trailing W-1 history into x_vals (with writeback
        # for the small-state_len edge case).
        var x_vals = SIMD[accum_t, width](0)

        @parameter
        for i in range(width - 1):
            var read_idx: Int = state_len - (width - 1) + i
            var state_val = conv_state_ptr[
                state_base + read_idx * state_l_stride
            ]
            var write_idx: Int = state_len - advance_len - (width - 1) + i
            if i < advance_len + (width - 1) and write_idx >= 0:
                conv_state_ptr[
                    state_base + write_idx * state_l_stride
                ] = state_val
            x_vals[i] = state_val.cast[accum_t]()

        # Phase 3: walk new x.
        for i in range(seqlen):
            var x_val = x_ptr[x_base + i * x_l_stride]
            var write_idx: Int = state_len - advance_len + i
            if i < advance_len and write_idx >= 0:
                conv_state_ptr[state_base + write_idx * state_l_stride] = x_val
            x_vals[width - 1] = x_val.cast[accum_t]()

            var out_val: Scalar[accum_t] = bias_v

            @parameter
            for k in range(width):
                out_val += weights[k] * x_vals[k]

            @parameter
            if apply_silu:
                out_val = _silu_f32(Float32(out_val))

            output_ptr[out_base + i * out_l_stride] = out_val.cast[dtype]()

            @parameter
            for k in range(width - 1):
                x_vals[k] = x_vals[k + 1]

    sync_parallelize[process_bc](batch * dim)
