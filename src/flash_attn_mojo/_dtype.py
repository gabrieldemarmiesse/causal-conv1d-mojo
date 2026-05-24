"""Shared helpers used by every per-function subpackage wrapper."""

from __future__ import annotations

import torch


# `bias` and `dbias_acc` may be None when the user omits bias; in that
# case we pass 0 for the data pointer. The Mojo kernels never
# dereference these pointers when the comptime `has_bias=False`.
def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


# Must match the dispatch in the Mojo entry points.
_DTYPE_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
