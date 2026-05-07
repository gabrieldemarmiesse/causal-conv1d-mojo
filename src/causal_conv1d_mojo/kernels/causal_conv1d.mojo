import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from std.utils.index import IndexList
from std.math import exp


@compiler.register("causal_conv1d_fn")
struct CausalConv1dFn[
    width: Int,
    has_bias: Bool,
    has_initial_states: Bool,
    activation: StaticString,
]:
    """Fused depthwise causal 1D convolution + carry-state.

    out[b, d, t] = silu_opt( sum_k src[b, d, t + k - (W-1)] * weight[d, k] + bias[d] )
    final_states[b, d, s] = src[b, d, L + s] for s in [0, W-1)
    where src = (initial_states or zeros) ++ x along the sequence dim.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        outp: OutputTensor[rank=3, ...],
        final_states: OutputTensor[dtype=outp.dtype, rank=3, ...],
        x: InputTensor[dtype=outp.dtype, rank=3, ...],
        weight: InputTensor[dtype=outp.dtype, rank=2, ...],
        bias: InputTensor[dtype=outp.dtype, rank=1, ...],
        initial_states: InputTensor[dtype=outp.dtype, rank=3, ...],
        ctx: DeviceContextPtr,
    ) raises:
        alias W: Int = Self.width
        alias dt = outp.dtype

        comptime assert dt.is_floating_point()

        @parameter
        @always_inline
        def compute_out[
            simd_width: Int
        ](idx: IndexList[3]) -> SIMD[dt, simd_width]:
            var b = idx[0]
            var d = idx[1]
            var t = idx[2]
            var acc: Scalar[dt] = 0

            @parameter
            for k in range(W):
                var src_t = t + k - (W - 1)
                var val: Scalar[dt]
                if src_t < 0:
                    comptime if Self.has_initial_states:
                        val = initial_states.load[1](
                            IndexList[3](b, d, t + k)
                        )[0]
                    else:
                        val = 0
                else:
                    val = x.load[1](IndexList[3](b, d, src_t))[0]
                acc += val * weight.load[1](IndexList[2](d, k))[0]

            comptime if Self.has_bias:
                acc += bias.load[1](IndexList[1](d))[0]

            comptime if Self.activation == "silu":
                acc = acc / (Scalar[dt](1) + exp(-acc))

            return SIMD[dt, simd_width](acc)

        foreach[compute_out, target=target, simd_width=1](outp, ctx)

        @parameter
        @always_inline
        def compute_final[
            simd_width: Int
        ](idx: IndexList[3]) -> SIMD[dt, simd_width]:
            # NOTE: read x.dim_size(2) inside the closure rather than
            # capturing it from the enclosing scope -- a captured runtime
            # var triggers a segfault in MAX 26.2's Model.execute.
            # See /tmp/max_multi_output_repro for a minimal reproducer.
            var L = x.dim_size(2)
            var b = idx[0]
            var d = idx[1]
            var s = idx[2]
            var combined_idx = L + s
            var val: Scalar[dt]
            if combined_idx < (W - 1):
                comptime if Self.has_initial_states:
                    val = initial_states.load[1](
                        IndexList[3](b, d, combined_idx)
                    )[0]
                else:
                    val = 0
            else:
                val = x.load[1](
                    IndexList[3](b, d, combined_idx - (W - 1))
                )[0]
            return SIMD[dt, simd_width](val)

        foreach[compute_final, target=target, simd_width=1](
            final_states, ctx
        )
