from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns


def noop_kernel(unused: Int):
    pass


def main() raises:
    var ctx = DeviceContext()
    print("Device:", ctx.name())
    var compiled = ctx.compile_function[noop_kernel, noop_kernel]()

    # warmup
    for _ in range(500):
        ctx.enqueue_function(compiled, 0, grid_dim=(1,), block_dim=(1,))
    ctx.synchronize()


    comptime ITERS: Int = 5000
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function(compiled, 0, grid_dim=(1,), block_dim=(1,))
    var t1 = perf_counter_ns()
    var us_enqueue_only = Float64(t1 - t0) / 1000.0 / Float64(ITERS)
    print("enqueue_function:", us_enqueue_only, " μs/call")
