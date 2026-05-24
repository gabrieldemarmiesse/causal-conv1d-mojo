# causal-conv1d-mojo-compatibility

Drop-in `import causal_conv1d` shim for
[causal-conv1d-mojo](https://pypi.org/project/causal-conv1d-mojo/).

```bash
pip install causal-conv1d-mojo-compatibility
```

Installs the main `causal-conv1d-mojo` package as a dependency, plus a
tiny top-level `causal_conv1d` re-export so existing code originally
written against the upstream Tri Dao `causal-conv1d` library works
unchanged:

```python
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
```

## Coexistence with upstream `causal-conv1d`

Both this package and the upstream `causal-conv1d` package own the
top-level `causal_conv1d/` import name. Pip will *let* you install
both, but the second-installed one overwrites parts of the first on
disk — you'll get a Frankenstein namespace.

To avoid this, importing `causal_conv1d` from this package will raise
an explicit `ImportError` if it detects upstream is also installed.
Pick one:

```bash
pip uninstall causal-conv1d                  # keep this package (Mojo kernels)
# or
pip uninstall causal-conv1d-mojo-compatibility   # keep upstream
```

## Why a separate package?

The main `causal-conv1d-mojo` package keeps its own `causal_conv1d_mojo`
import name and never touches the `causal_conv1d` namespace. Users who
just want fast Mojo kernels (`import causal_conv1d_mojo`) pay zero risk
of clashing with upstream. Users who explicitly want the drop-in
behaviour opt in by installing this compatibility package.

Same pattern used by [pycryptodome /
pycryptodomex](https://www.pycryptodome.org/): the main package keeps
a private namespace; the compat distribution takes over the well-known
name and accepts the conflict.

## Versioning

This package is version-locked to `causal-conv1d-mojo==X.Y.Z` exactly.
Release both together.
