# CLAUDE.md

Guidance for working on this repo: a Mojo port of Tri Dao's CUDA
`flash-attn`. The benchmark we care about is "GPU kernel time vs
upstream flash-attn 2 CUDA", with upstream as the moving target.

**Status: scaffolding only.** The Python infrastructure (JIT-on-first-
use cache, env-signature keying, autograd + `torch.library.custom_op`
plumbing, compat package for `import flash_attn`, beartype-in-tests)
is in place. The Mojo kernels themselves (`fwd/kernel.mojo`,
`bwd/kernel.mojo`) are not yet implemented — calls to
`flash_attn_func` on CUDA tensors raise `NotImplementedError` and
point at the missing piece.

## Repository layout

- `src/flash_attn_mojo/`
  - `fwd/`, `bwd/`: GPU kernels (one subpackage each, mirroring
    causal-conv1d-mojo's structure). Pure JIT-on-first-use — there is
    no `dispatch.mojo` and no AOT comptime sweep. Once implemented,
    each subpackage will have:
    - `kernel.mojo` (the device function — comptime-parameterized
      over dtype, head_dim, causal flag, …).
    - `common.mojo` (shared constants/helpers).
    - `launch.mojo` (configures `DeviceContext`, builds `TileTensor`
      layouts, calls `compile_function` + `enqueue_function`).
    - `variant.mojo` (the static per-subpackage entry point. Reads
      its comptime params via `std.sys.get_defined_*` so a single
      source file covers every config. Exports `PyInit_variant`).
    - `_jit.py` (Python: extracts the config tuple from the call's
      runtime args, formats a readable mod name, materialises the
      config as `-D KEY=VALUE` pairs, and delegates to the shared
      cache+compile+load helper).
    - `__init__.py` (Python wrapper that builds the args tuple and
      calls `_jit.call_<sub>(args)`).
    The shared `mojo build` → `dlopen` plumbing lives in
    `_jit_common.py` at the package root (`compile_and_load`).
    Per-variant artefacts cache under
    `$XDG_CACHE_HOME/flash_attn_mojo/<sub>/<backend>/<arch>/<cpu_tag>/<mod_name>.hash-<h>.so`.
  - `_jit_common.py`: shared variant cache + compile + load helper.
    Also owns the env-signature → cache-hash logic.
  - `_fn.py`: the public `flash_attn_func` autograd op.
  - `reference.py`: pure-PyTorch `flash_attn_ref` (SDPA-based).
- `tests/`: pytest suite. Run with `uv run --extra nvidia pytest`
  (the `nvidia` extra brings in upstream flash-attn for cross-
  validation against the Mojo kernels, once implemented).
- `compat/`: drop-in `import flash_attn` shim package
  (`flash-attn-mojo-compatibility`).
- `flash-attention/`: vendored Tri Dao CUDA source (read-only
  reference for kernel patterns). Cloned via
  `git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git`;
  gitignored, not part of the repo. The relevant subdirs:
  - `csrc/flash_attn/src/` — the sm80 (Ampere/Ada) FA2 kernels.
    `flash_fwd_hdim{32,64,96,128,...}_{fp16,bf16}_{,causal_}sm80.cu`
    are the per-head-dim instantiations; the algorithm lives in
    `flash_fwd_kernel.h` and `flash_bwd_kernel.h`.
  - `hopper/` — the sm90 (Hopper) FA3 kernels (separate codebase
    that uses TMA + WGMMA).
  - `flash_attn/` — the Python wrapper (`flash_attn_interface.py`)
    that mirrors the API surface we expose.

## Running the benches

Always use `uv run --extra nvidia …` — the `nvidia` extra pulls in
the upstream flash-attn wheel that the benches diff against.

```bash
# Once kernels exist:
# Kernel-only GPU time per shape (uses torch.profiler CUPTI hooks)
uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py
```

## Cache invalidation

In most cases you should never need to manually clear the cache —
the env signature (Python ABI, mojo version, modular SDK path, CPU
brand, ptxas version) automatically invalidates on env shifts. If
you suspect something stale anyway:

```bash
rm -rf ~/.cache/flash_attn_mojo/
```

### Cache-key contents

Identical to causal-conv1d-mojo's. See `_jit_common.py::_env_signature`
for the authoritative list:

- **`soabi`**: Python C-extension ABI tag.
- **`mojo_version`**: `mojo --version` output (includes git hash).
- **`modular_root`**: path to the modular SDK install (baked into the
  `.so`'s `RUNPATH`).
- **`cpu_brand`**: full host-CPU brand string. Mojo's `-march=native`
  bakes host SIMD into every `.so`; sharing the cache across CPUs
  with fewer ISA extensions SIGILLs.
- **`jit_common_hash`**: this file's contents (defensive).
- **`ptxas`** (CUDA only): bundled / vendored / external.

### Production: pre-warmed cache + `FLASH_ATTN_MOJO_USE_CACHE_ONLY`

Same pattern as causal-conv1d-mojo's `CAUSAL_CONV1D_USE_CACHE_ONLY`.
Pre-warm a staging host matching production, bundle
`~/.cache/flash_attn_mojo/` into the image, set the flag in prod to
turn any cache miss into a loud `RuntimeError` instead of a silent
~1.2 s JIT compile in the request hot path.

## Kernel-design patterns to mirror

(From the causal-conv1d-mojo work — most of these will apply here
too once we start writing kernels.)

1. **One block per (B, H_q)** with the seqlen walked in a chunk
   loop — keeps the K/V tile reuse high.
2. **16-byte LDG** per thread (`kNElts = 16 // size_of[dtype]()`).
3. **Smem ring-buffer** for K/V tiles across the seqlen iteration.
4. **`aligned_seq` comptime gate** to drop the bounds-checked
   tail-chunk path when seqlen is aligned to the block size.
5. **One cubin per (dtype × head_dim × causal × …) leaf**, compiled
   JIT on first use, cached.
6. **`Atomic[dtype, scope="device"].fetch_add[ordering=RELAXED]`**
   for the dq/dk/dv reduce in the backward.

## Where to look first when perf regresses

1. Run kernel-only timing benches and compare ratios per shape.
2. If small shapes regress but large shapes are fine → launch
   overhead or low-occupancy regime.
3. If all shapes regress → check PTX. Add a temporary
   `dump_asm=StaticString("/tmp/mojo_<sub>_%.ptx")` to the
   `compile_function[...]` call in `<sub>/launch.mojo`, trigger
   once, and remove. Don't commit it.
4. The vendored Tri Dao flash-attn source (clone into a sibling
   directory or symlink) is the reference for every algorithmic
   choice.

## Mojo gotchas hit while porting (carried over from causal-conv1d-mojo)

- `DType` has no `.size_of()` method; use `from std.sys import size_of`
  and call `size_of[dtype]()`.
- `stack_allocation[..., address_space=AddressSpace.SHARED]()` returns
  an `UnsafePointer` with no `.offset()` — use `ptr + i`.
- `comptime for x, y, ... in product(...)` only handles up to 4
  iterables. Nest loops.
- The `mojo build` cache bakes the *build env's* modular-lib path into
  each `.so`'s `RUNPATH`. The env signature handles env switches
  automatically; you only need to nuke the cache for disk hygiene.
- `dump_asm` paths must be `StaticString(...)`-wrapped.
- `TileTensor` has two costs at very small kernel runtimes (a few
  μs total):
  1. `linear_idx_type` defaults to `DType.int64` for global-memory
     tensors with dynamic dims → `mul.lo.s64` everywhere. Force
     `linear_idx_type=DType.int32` if the addressable range fits.
  2. The layout is a packed kernarg struct, accessed via offsetted
     `ld.param.b32` loads rather than direct register loads. ~5-10
     extra cycles in the prologue. Worth keeping the kernels that
     run for tens-to-hundreds of μs (fwd, bwd) on TileTensor; for
     anything microsecond-scale (like a decode-step kernel), raw
     pointers + Int32 strides are still the right call.
