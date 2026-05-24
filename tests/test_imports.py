import ast
import copy
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


def _strip_annotations(args: ast.arguments) -> ast.arguments:
    """Drop param annotations so we compare the API surface (param names,
    defaults, positional/keyword split) without caring whether each side
    bothers to type-hint. Deep-copied so we don't mutate the caller's tree."""
    args = copy.deepcopy(args)
    for a in args.posonlyargs + args.args + args.kwonlyargs:
        a.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None
    return args


def _signature_from_source(path: Path, fn_name: str) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.unparse(_strip_annotations(node.args))
    raise AssertionError(f"{fn_name} not found in {path}")


@pytest.mark.skipif(
    not UPSTREAM_INTERFACE.exists(),
    reason="upstream causal-conv1d source tree not cloned alongside the project",
)
def test_causal_conv1d_fn_signature_matches_upstream():
    from causal_conv1d_mojo import causal_conv1d_fn

    # Compare bare API surface — param names, defaults, positional/keyword
    # split. Annotations are a per-package detail and differ legitimately
    # (we have them, upstream doesn't).
    mojo_args = ast.parse(inspect.getsource(causal_conv1d_fn)).body[0].args
    mojo_sig = ast.unparse(_strip_annotations(mojo_args))
    upstream_sig = _signature_from_source(UPSTREAM_INTERFACE, "causal_conv1d_fn")
    assert mojo_sig == upstream_sig, (
        f"signature drift: mojo={mojo_sig!r} upstream={upstream_sig!r}"
    )
