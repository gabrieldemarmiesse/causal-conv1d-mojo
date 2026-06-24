"""Assembly/PTX tooling for the NVIDIA master bench (steps e/f/g).

Our Mojo kernels carry no device code in the compiled ``.so`` (the GPU
kernel is lowered at runtime), so we dump PTX via the
``CAUSAL_CONV1D_DUMP_ASM`` env hook (see ``_jit_common._maybe_add_asm_dump``)
and derive everything else here with a portable ``ptxas`` / ``nvdisasm`` /
``cuobjdump`` — the ones shipped inside the ``triton`` wheel — rather than
relying on a system CUDA toolkit.

Subcommands
-----------
    sass    <in.ptx> <out.sass> [--arch sm_90a]
        PTX -> cubin (ptxas) -> SASS (nvdisasm).

    spill   <in.ptx> [--arch sm_90a] [--max-spill N]
        ptxas -v register/spill report. Exit 1 if spill bytes exceed
        --max-spill (the spill/regalloc canary; default 0).

    upstream-sass <fn> <out.sass> [--arch sm_90]
        Extract upstream Tri Dao's <fn> kernel SASS from its compiled
        .so via cuobjdump. fn in {fwd,bwd,update}.

    histogram <ours> <theirs> [--top N] [--format auto|sass|ptx]
        Side-by-side instruction-mix histogram (base-opcode counts) of
        two PTX or SASS files. The trailing column is ours-minus-theirs.
"""

from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


# --------------------------------------------------------------------------
# Locate the NVIDIA binaries shipped with the triton wheel (no system CUDA
# toolkit assumed). They live under
# ``<venv>/lib/pythonX/site-packages/triton/backends/nvidia/bin/``.
# --------------------------------------------------------------------------


def _triton_bin(name: str) -> str:
    hits = glob.glob(
        str(
            Path(sys.prefix)
            / "lib"
            / "*"
            / "site-packages"
            / "triton"
            / "backends"
            / "nvidia"
            / "bin"
            / name
        )
    )
    if not hits:
        raise SystemExit(
            f"could not find `{name}` under the triton wheel "
            f"(looked in {sys.prefix}/lib/*/site-packages/triton/backends/"
            f"nvidia/bin/). Run via `uv run --extra nvidia ...`."
        )
    return hits[0]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


# --------------------------------------------------------------------------
# PTX -> SASS, and the ptxas -v spill canary.
# --------------------------------------------------------------------------


def cmd_sass(args) -> int:
    ptxas, nvdisasm = _triton_bin("ptxas"), _triton_bin("nvdisasm")
    out = Path(args.out)
    cubin = out.with_suffix(".cubin")
    r = _run([ptxas, f"-arch={args.arch}", "-O3", str(args.ptx), "-o", str(cubin)])
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ptxas failed on {args.ptx}")
    r = _run([nvdisasm, "-c", str(cubin)])
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"nvdisasm failed on {cubin}")
    out.write_text(r.stdout)
    cubin.unlink(missing_ok=True)
    print(f"wrote {out} ({len(r.stdout.splitlines())} SASS lines)")
    return 0


_SPILL_RE = re.compile(
    r"(\d+)\s+bytes\s+spill\s+stores.*?(\d+)\s+bytes\s+spill\s+loads", re.DOTALL
)
_REG_RE = re.compile(r"Used\s+(\d+)\s+registers")


def cmd_spill(args) -> int:
    ptxas = _triton_bin("ptxas")
    # ptxas needs a real output target; throw the cubin away.
    r = _run(
        [ptxas, "-v", f"-arch={args.arch}", "-O3", str(args.ptx), "-o", "/dev/null"]
    )
    info = r.stderr + r.stdout
    if r.returncode != 0:
        sys.stderr.write(info)
        raise SystemExit(f"ptxas -v failed on {args.ptx}")
    regs = [int(m) for m in _REG_RE.findall(info)]
    spills = _SPILL_RE.findall(info)
    spill_store = sum(int(a) for a, _ in spills)
    spill_load = sum(int(b) for _, b in spills)
    print(f"ptxas -v ({args.arch}) for {Path(args.ptx).name}:")
    print(f"  registers (max over entries): {max(regs) if regs else '?'}")
    print(f"  spill stores: {spill_store} bytes")
    print(f"  spill loads:  {spill_load} bytes")
    total = spill_store + spill_load
    if total > args.max_spill:
        print(
            f"  SPILL CANARY FAILED: {total} bytes spilled > "
            f"--max-spill {args.max_spill}"
        )
        return 1
    print(f"  spill canary OK ({total} <= {args.max_spill})")
    return 0


# --------------------------------------------------------------------------
# Upstream SASS extraction (cuobjdump on the Tri Dao .so).
# --------------------------------------------------------------------------


def _upstream_so() -> str:
    hits = glob.glob(
        str(Path(sys.prefix) / "lib" / "*" / "site-packages" / "causal_conv1d_cuda*.so")
    )
    if not hits:
        raise SystemExit(
            "could not find the upstream causal_conv1d_cuda .so; "
            "run via `uv run --extra nvidia ...`."
        )
    return hits[0]


def _kernel_blocks(sass: str) -> list[tuple[str, str]]:
    """Split cuobjdump --dump-sass output into (function_name, body) blocks."""
    blocks: list[tuple[str, list[str]]] = []
    for line in sass.splitlines():
        if "Function :" in line:
            name = line.split("Function :", 1)[1].strip()
            blocks.append((name, [line]))
        elif blocks:
            blocks[-1][1].append(line)
    return [(n, "\n".join(b)) for n, b in blocks]


def cmd_upstream_sass(args) -> int:
    cuobjdump = _triton_bin("cuobjdump")
    so = _upstream_so()
    r = _run([cuobjdump, "--dump-sass", f"--gpu-architecture={args.arch}", so])
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("cuobjdump --dump-sass failed")
    # The cubin holds every (width × dtype × silu × vecload) instantiation
    # (>1000 of them); match the base kernel name plus the user's terms to
    # isolate the one comparable to our variant. Note `causal_conv1d_fwd_kernel`
    # excludes `causal_conv1d_channellast_fwd_kernel` (the substring differs).
    needles = [f"causal_conv1d_{args.fn}_kernel", *args.match]
    matches = [
        (n, b) for n, b in _kernel_blocks(r.stdout) if all(s in n for s in needles)
    ]
    if not matches:
        raise SystemExit(
            f"no kernel matched {needles} in {Path(so).name}; "
            f"re-run with --list to see candidates"
        )
    if args.list:
        for i, (n, _) in enumerate(matches):
            print(f"[{i}] {n}")
        return 0
    if len(matches) > 1 and args.index is None:
        sys.stderr.write(
            f"{len(matches)} kernels matched {needles}; pass --index or tighten "
            f"--match (use --list to see them). Defaulting to [0].\n"
        )
    idx = args.index or 0
    if not (0 <= idx < len(matches)):
        raise SystemExit(
            f"--index {idx} out of range; {len(matches)} kernel(s) matched"
        )
    name, body = matches[idx]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Record the selected mangled name as a header comment for provenance.
    Path(args.out).write_text(
        f"// selected: {name}\n// from: {Path(so).name}\n{body}\n"
    )
    print(f"wrote {args.out} ({len(body.splitlines())} SASS lines)\n  kernel: {name}")
    return 0


# --------------------------------------------------------------------------
# Instruction-mix histogram.
# --------------------------------------------------------------------------

# Predicate guard covers @P0/@!P1 (per-thread), @UP0/@!UP1 (uniform, common
# on sm_90), and the always-true @PT/@UPT. The optional leading `{` handles
# dual-issue scheduling-brace lines (`{ OPCODE ... }`).
_SASS_LINE = re.compile(
    r"/\*[0-9a-fA-F]+\*/\s+\{?\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)"
)
_PTX_LINE = re.compile(r"^\s*(?:@!?%?\w+\s+)?([a-z][a-z0-9_]*)\b")


def _detect_format(text: str) -> str:
    return "sass" if re.search(r"/\*[0-9a-fA-F]{4,}\*/", text[:8000]) else "ptx"


def _opcode_counts(path: Path, fmt: str) -> Counter:
    text = path.read_text(errors="replace")
    if fmt == "auto":
        fmt = _detect_format(text)
    counts: Counter = Counter()
    if fmt == "sass":
        for line in text.splitlines():
            m = _SASS_LINE.search(line)
            if m:
                counts[m.group(1).split(".")[0]] += 1  # base opcode
    else:  # ptx
        for line in text.splitlines():
            s = line.strip()
            # Keep '@'-prefixed lines: those are predicated instructions
            # (e.g. `@%p1 bra ...`) — _PTX_LINE consumes the guard. '$' /
            # trailing ':' still drop labels like `$L__BB0_2:`.
            if not s or s.startswith(("//", ".", "{", "}", "$")) or s.endswith(":"):
                continue
            m = _PTX_LINE.match(line)
            if m:
                counts[m.group(1).split(".")[0]] += 1
    return counts


def cmd_histogram(args) -> int:
    ours = _opcode_counts(Path(args.ours), args.format)
    theirs = _opcode_counts(Path(args.theirs), args.format)
    keys = set(ours) | set(theirs)
    rows = sorted(keys, key=lambda k: max(ours[k], theirs[k]), reverse=True)
    if args.top:
        rows = rows[: args.top]

    def _lbl(p):
        q = Path(p)
        return f"{q.parent.name}/{q.name}"

    print(
        f"instruction-mix histogram  (ours={_lbl(args.ours)}  theirs={_lbl(args.theirs)})"
    )
    print(f"  {'opcode':>14} | {'ours':>6} | {'theirs':>6} | {'delta':>6}")
    print("  " + "-" * 44)
    hidden = 0
    for k in rows:
        o, t = ours[k], theirs[k]
        if o == t:  # identical count -> no signal; drop to keep the diff readable
            hidden += 1
            continue
        print(f"  {k:>14} | {o:>6} | {t:>6} | {o - t:>+6}")
    if hidden:
        print(f"  ({hidden} opcode(s) with delta 0 hidden)")
    print("  " + "-" * 44)
    print(
        f"  {'TOTAL':>14} | {sum(ours.values()):>6} | "
        f"{sum(theirs.values()):>6} | {sum(ours.values()) - sum(theirs.values()):>+6}"
    )
    return 0


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sass", help="PTX -> SASS via ptxas+nvdisasm")
    s.add_argument("ptx")
    s.add_argument("out")
    s.add_argument("--arch", default="sm_90a")
    s.set_defaults(func=cmd_sass)

    s = sub.add_parser("spill", help="ptxas -v spill/regalloc canary")
    s.add_argument("ptx")
    s.add_argument("--arch", default="sm_90a")
    s.add_argument(
        "--max-spill", type=int, default=0, help="max total spill bytes before failing"
    )
    s.set_defaults(func=cmd_spill)

    s = sub.add_parser("upstream-sass", help="extract upstream SASS via cuobjdump")
    s.add_argument("fn", choices=("fwd", "bwd", "update"))
    s.add_argument("out")
    s.add_argument("--arch", default="sm_90")
    s.add_argument(
        "--match",
        action="append",
        default=[],
        help="extra mangled-name substring (repeatable)",
    )
    s.add_argument("--index", type=int, default=None, help="pick the Nth match")
    s.add_argument("--list", action="store_true", help="list matching kernels and exit")
    s.set_defaults(func=cmd_upstream_sass)

    s = sub.add_parser("histogram", help="side-by-side instruction-mix histogram")
    s.add_argument("ours")
    s.add_argument("theirs")
    s.add_argument("--top", type=int, default=0)
    s.add_argument("--format", choices=("auto", "sass", "ptx"), default="auto")
    s.set_defaults(func=cmd_histogram)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
