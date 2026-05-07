import ast
import inspect
from pathlib import Path

import pytest


UPSTREAM_INTERFACE = (
    Path(__file__).resolve().parent.parent
    / "causal-conv1d"
    / "causal_conv1d"
    / "causal_conv1d_interface.py"
)


def test_package_importable():
    import causal_conv1d_mojo  # noqa: F401


def test_causal_conv1d_fn_exported():
    import causal_conv1d_mojo

    assert callable(causal_conv1d_mojo.causal_conv1d_fn)
    assert isinstance(causal_conv1d_mojo.__version__, str)


def _signature_from_source(path: Path, fn_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.unparse(node.args)
    raise AssertionError(f"{fn_name} not found in {path}")


@pytest.mark.skipif(
    not UPSTREAM_INTERFACE.exists(),
    reason="upstream causal-conv1d source tree not cloned alongside the project",
)
def test_causal_conv1d_fn_signature_matches_upstream():
    from causal_conv1d_mojo import causal_conv1d_fn

    mojo_sig = ast.unparse(
        ast.parse(inspect.getsource(causal_conv1d_fn)).body[0].args
    )
    upstream_sig = _signature_from_source(UPSTREAM_INTERFACE, "causal_conv1d_fn")
    assert mojo_sig == upstream_sig, (
        f"signature drift: mojo={mojo_sig!r} upstream={upstream_sig!r}"
    )
