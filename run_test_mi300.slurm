#!/bin/bash
#SBATCH --job-name=ccm_test_mi300
#SBATCH --account=cad17897
#SBATCH --constraint=MI300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# mojo requires GLIBCXX_3.4.30+ which the login/compute nodes don't ship in
# the default /lib64/libstdc++.so.6 (tops out at 3.4.29).  GCC 13.2 bundles
# a newer libstdc++ that covers it.
export LD_LIBRARY_PATH=/opt/software/gcc/13.2.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

REPO=/lus/scratch/CT10/cad17897/gdemarmiesse/causal-conv1d-mojo
VENV="$REPO/.venv-rocm"

echo "=== Node: $(hostname) ==="
echo "=== ROCm info ==="
rocm-smi --showallinfo 2>/dev/null | head -30 || true

cd "$REPO"

# Create and sync a ROCm venv on first run.  The compute nodes have no
# outbound internet, so we skip the sync when the venv already contains
# the three packages we actually need (torch, mojo, pytest).
if [ ! -d "$VENV" ] || ! "$VENV/bin/python" -c "import torch, mojo, pytest" 2>/dev/null; then
    echo "=== Creating ROCm venv ==="
    uv venv "$VENV" --python 3.13
    echo "=== Syncing ROCm deps ==="
    UV_PROJECT_ENVIRONMENT="$VENV" uv sync --extra rocm --python 3.13
else
    echo "=== Venv already complete, skipping sync ==="
fi

echo "=== Python / torch version ==="
"$VENV/bin/python" -c "
import torch
print('torch:', torch.__version__)
print('CUDA (HIP) available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print('GPU', i, ':', torch.cuda.get_device_name(i))
"

echo "=== Running single test ==="
"$VENV/bin/python" -m pytest "tests/test_fwd.py::test_contiguous[cuda-fp16-None-with_bias-shape0]" -v --no-header -s 2>&1

echo "=== Done ==="
