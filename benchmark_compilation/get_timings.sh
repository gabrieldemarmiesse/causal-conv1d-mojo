set -ex

echo "=== GPU forward (JIT per variant) ==="
time uv run --frozen --no-sync python benchmark_compilation/simple_forward.py

echo "=== CPU forward (AOT comptime sweep) ==="
time uv run --frozen --no-sync python benchmark_compilation/simple_forward_cpu.py
