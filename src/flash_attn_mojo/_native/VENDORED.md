# Vendored from `modular/modular`

Files under `src/flash_attn_mojo/_native/attention/`,
`src/flash_attn_mojo/_native/kv_cache.mojo`, and
`src/flash_attn_mojo/_native/kv_cache_ragged.mojo` are copied verbatim
from the [modular/modular](https://github.com/modular/modular)
repository, paths `max/kernels/src/nn/attention/...`,
`max/kernels/src/nn/kv_cache.mojo`,
`max/kernels/src/nn/kv_cache_ragged.mojo`.

Provenance for the initial vendor:

- Upstream commit: `56b5e1aef2b7dc57ee04e1bd377de3f5c18b546e`
- Source paths under that commit:
  - `max/kernels/src/nn/attention/` (entire subtree)
  - `max/kernels/src/nn/kv_cache.mojo`
  - `max/kernels/src/nn/kv_cache_ragged.mojo`

The original Apache 2.0 license headers (with LLVM Exception) are
preserved verbatim in every vendored file. The license text itself is
reproduced at the upstream repository's `LICENSE` file; we don't
duplicate it here, but vendored files remain governed by it.

Local modifications — none yet at the time of the initial vendor.
Subsequent commits in this repo will modify these files; check
`git log -- src/flash_attn_mojo/_native/` to see what diverged.

To upgrade:

1. `cd /path/to/modular && git pull` to fetch new upstream changes.
2. `cd /path/to/causal-conv1d-mojo`. Diff against the upstream commit
   we last vendored from (recorded above) and merge changes file-by-file
   into `src/flash_attn_mojo/_native/`. Update the "Upstream commit"
   field in this file when done.
