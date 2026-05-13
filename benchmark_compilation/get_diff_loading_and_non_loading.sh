set -ex

# get the time to import torch
time uv run --frozen --no-sync python -c "import torch"

# get the time to import causal-conv1d-mojo
time uv run --frozen --no-sync python -c "import causal_conv1d_mojo"

# get the time to execute a simple forward pass
time uv run --frozen --no-sync ...

# and now redo it again to see if it's faster the second time around (it should be)
time uv run --frozen --no-sync ...
