# flash-attn-mojo-compatibility

Drop-in `import flash_attn` shim for
[flash-attn-mojo](https://pypi.org/project/flash-attn-mojo/).

```bash
pip install flash-attn-mojo-compatibility
```

Installs the main `flash-attn-mojo` package as a dependency plus a
tiny top-level `flash_attn` re-export so existing code originally
written against the upstream Tri Dao `flash-attn` library works
unchanged:

```python
from flash_attn import flash_attn_func
```

## Coexistence with upstream `flash-attn`

Both this package and the upstream `flash-attn` package own the
top-level `flash_attn/` import name. Pip will *let* you install
both, but the second-installed one overwrites parts of the first on
disk — you'll get a Frankenstein namespace. Importing `flash_attn`
from this package raises an explicit `ImportError` if it detects
upstream is also installed. Pick one:

```bash
pip uninstall flash-attn                       # keep this package
# or
pip uninstall flash-attn-mojo-compatibility    # keep upstream
```

## Versioning

Version-locked to `flash-attn-mojo==X.Y.Z` exactly. Released together.
