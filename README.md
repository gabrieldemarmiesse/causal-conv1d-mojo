# flash-attn-mojo

A Mojo port of Tri Dao's CUDA
[flash-attn](https://github.com/Dao-AILab/flash-attention) library,
packaged so it works everywhere torch works (no CUDA toolchain
required at install time, ships pre-built wheels for the common
configs).

**Status: scaffolding only.** The Python infrastructure is in place;
the Mojo kernels themselves are not yet implemented. CUDA calls to
`flash_attn_func` currently raise `NotImplementedError`. CPU calls
fall through to a pure-PyTorch SDPA reference. See `CLAUDE.md` for
the design notes.

## Install

```bash
pip install flash-attn-mojo
```

For drop-in `import flash_attn` compatibility with code originally
written against upstream Tri Dao:

```bash
pip install flash-attn-mojo-compatibility
```

(See `compat/README.md` for caveats — that package conflicts with
the upstream `flash-attn` package on the `flash_attn` import name.)

## Usage

```python
import flash_attn_mojo
out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
```

Or, with the compat package installed:

```python
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)
```

## Layout

- `src/flash_attn_mojo/` — main package
- `compat/` — separate distribution providing the `flash_attn` alias
- `tests/` — pytest suite
- `CLAUDE.md` — design notes and contributor guide
