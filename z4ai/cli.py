# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the z4ai codec.

Usage examples::

    # Compress a raw weight dump as BF16 (2-byte elements) -> model.z4ai
    z4ai compress model.bin -o model.z4ai --dtype bf16

    # Decompress back to the original bytes
    z4ai decompress model.z4ai -o model.bin

    # Inspect a blob without decompressing it
    z4ai info model.z4ai

    # Pipe through stdin/stdout (use "-" for either path)
    cat model.bin | z4ai compress - --dtype fp32 > model.z4ai

The CLI is a thin, dependency-light wrapper over the public byte-buffer API
(:func:`z4ai.compress` / :func:`z4ai.decompress`).  It deliberately avoids the
NumPy array path so it can compress arbitrary files (the element ``width`` is
all the float-plane transpose needs).
"""

from __future__ import annotations

import argparse
import sys
from typing import BinaryIO, List, Optional

from . import __version__
from . import codec, format

# Human-friendly dtype names -> byte-plane transpose width.  These mirror the
# dtype codes in API_CONTRACT.md; only the element *size* matters to the
# byte-plane transform, so several dtypes share a width.
DTYPE_WIDTHS = {
    "uint8": 1,
    "int8": 1,
    "raw": 1,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "fp32": 4,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "fp64": 8,
    "float64": 8,
    "int64": 8,
    "uint64": 8,
}

_METHOD_NAMES = {
    format_method: name
    for format_method, name in [(0, "store"), (1, "zstd")]
}


def _resolve_width(dtype: Optional[str], width: Optional[int]) -> int:
    """Turn the user's ``--dtype``/``--width`` choice into a transpose width."""
    if width is not None:
        if width < 1:
            raise SystemExit("error: --width must be >= 1")
        return width
    if dtype is None:
        return 1
    key = dtype.lower()
    if key not in DTYPE_WIDTHS:
        valid = ", ".join(sorted(set(DTYPE_WIDTHS)))
        raise SystemExit(f"error: unknown --dtype {dtype!r}; choose one of: {valid}")
    return DTYPE_WIDTHS[key]


def _open_in(path: str) -> BinaryIO:
    if path == "-":
        return sys.stdin.buffer
    return open(path, "rb")


def _open_out(path: str) -> BinaryIO:
    if path == "-":
        return sys.stdout.buffer
    return open(path, "wb")


def _read_all(path: str) -> bytes:
    fh = _open_in(path)
    try:
        return fh.read()
    finally:
        if path != "-":
            fh.close()


def _write_all(path: str, data: bytes) -> None:
    fh = _open_out(path)
    try:
        fh.write(data)
        fh.flush()
    finally:
        if path != "-":
            fh.close()


def _default_out(in_path: str, *, compressing: bool) -> str:
    """Pick a sensible default output path when ``-o`` is omitted."""
    if in_path == "-":
        return "-"
    if compressing:
        return in_path + ".z4ai"
    if in_path.endswith(".z4ai"):
        return in_path[: -len(".z4ai")]
    return in_path + ".out"


def _human(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}" if i else f"{int(n)} {units[i]}"


def _cmd_compress(args: argparse.Namespace) -> int:
    data = _read_all(args.input)
    escalate = None if args.no_escalate else args.escalate
    # Prefer the dtype-aware path (it routes bf16/fp16/fp32 through the
    # high-ratio bit-field split); fall back to an explicit --width for advanced
    # / opaque buffers.  --width always wins when given.
    if args.width is not None:
        width = _resolve_width(None, args.width)
        blob = codec.compress(
            data, width=width, level=args.level, escalate_level=escalate,
            threads=args.threads, plane_workers=args.plane_workers,
            auto=not args.fast,
        )
    else:
        width = _resolve_width(args.dtype, None)
        blob = codec.compress(
            data, dtype=args.dtype, level=args.level, escalate_level=escalate,
            threads=args.threads, plane_workers=args.plane_workers,
            auto=not args.fast,
        )
    out_path = args.output or _default_out(args.input, compressing=True)
    _write_all(out_path, blob)
    if not args.quiet and out_path != "-":
        # Ratio is reported as original/compressed (>1 == smaller output), to
        # match the README and benchmark convention.
        ratio = (len(data) / len(blob)) if blob else 1.0
        saved = (1.0 - len(blob) / len(data)) * 100.0 if data else 0.0
        sys.stderr.write(
            f"{args.input}: {_human(len(data))} -> {_human(len(blob))} "
            f"({ratio:.3f}x, {saved:+.1f}% saved) [width={width}]\n"
        )
    return 0


def _cmd_decompress(args: argparse.Namespace) -> int:
    blob = _read_all(args.input)
    if not format.peek_is_z4ai(blob):
        raise SystemExit(f"error: {args.input} is not a z4ai blob (bad magic)")
    data = codec.decompress(blob)
    out_path = args.output or _default_out(args.input, compressing=False)
    _write_all(out_path, data)
    if not args.quiet and out_path != "-":
        sys.stderr.write(
            f"{args.input}: {_human(len(blob))} -> {_human(len(data))}\n"
        )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    blob = _read_all(args.input)
    if not format.peek_is_z4ai(blob):
        raise SystemExit(f"error: {args.input} is not a z4ai blob (bad magic)")
    container = format.deserialize(blob)
    orig = sum(p.orig_len for p in container.planes) + len(container.tail)
    comp = len(blob)
    transform_name = {
        format.TRANSFORM_OPAQUE: "opaque",
        format.TRANSFORM_BYTEPLANE: "byte-plane",
        format.TRANSFORM_BITFIELD: "bit-field",
    }.get(container.transform, f"transform#{container.transform}")
    print(f"file:        {args.input}")
    print(f"magic:       Z4AI  (valid)")
    print(f"transform:   {transform_name}")
    if container.dtype_code:
        print(f"dtype:       {container.dtype_code}")
    print(f"numpy array: {container.is_numpy}")
    if container.is_numpy:
        print(f"np dtype:    {container.dtype}")
        print(f"shape:       {container.shape}")
    print(f"width:       {container.width}")
    print(f"streams:     {len(container.planes)}")
    print(f"tail bytes:  {len(container.tail)}")
    for i, p in enumerate(container.planes):
        method = _METHOD_NAMES.get(p.method, f"method#{p.method}")
        # original/compressed, matching the headline convention below.
        pr = (p.orig_len / len(p.data)) if len(p.data) else 1.0
        print(
            f"  plane {i}: {method:5s}  {_human(p.orig_len)} -> "
            f"{_human(len(p.data))}  ({pr:.3f}x)"
        )
    ratio = (orig / comp) if comp else 1.0
    saved = (1.0 - comp / orig) * 100.0 if orig else 0.0
    print(f"original:    {_human(orig)} ({orig} B)")
    print(f"compressed:  {_human(comp)} ({comp} B)")
    print(f"ratio:       {ratio:.4f}x  ({saved:+.1f}% saved)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="z4ai",
        description="z4ai: lossless compression tuned for neural-network weights.",
    )
    parser.add_argument(
        "--version", action="version", version=f"z4ai {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common_io = argparse.ArgumentParser(add_help=False)
    common_io.add_argument("input", help='input path ("-" for stdin)')
    common_io.add_argument(
        "-o", "--output", default=None, help='output path ("-" for stdout)'
    )
    common_io.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the summary line"
    )

    c = sub.add_parser(
        "compress", parents=[common_io], help="compress a file or stream"
    )
    c.add_argument(
        "--dtype",
        default=None,
        help="element dtype (bf16, fp16, fp32, fp64, uint8, ...); sets width",
    )
    c.add_argument(
        "--width",
        type=int,
        default=None,
        help="explicit byte-plane element width (overrides --dtype)",
    )
    c.add_argument(
        "--level",
        type=int,
        default=codec.DEFAULT_LEVEL,
        help=f"base Zstd level (default {codec.DEFAULT_LEVEL})",
    )
    c.add_argument(
        "--escalate",
        type=int,
        default=codec.DEFAULT_ESCALATE_LEVEL,
        help=f"Zstd level for compressible planes "
        f"(default {codec.DEFAULT_ESCALATE_LEVEL})",
    )
    c.add_argument(
        "--no-escalate",
        action="store_true",
        help="disable per-plane level escalation",
    )
    c.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Zstd worker threads per plane (0 = single threaded)",
    )
    c.add_argument(
        "--plane-workers",
        type=int,
        default=1,
        help="Python threads used to compress planes concurrently",
    )
    c.add_argument(
        "--fast",
        action="store_true",
        help="single-pass mode: skip the AUTO best-of-transform search "
        "(~3x faster compress, identical ratio on i.i.d. float weights; may "
        "give up some ratio on highly structured/redundant data)",
    )
    c.set_defaults(func=_cmd_compress)

    d = sub.add_parser(
        "decompress", parents=[common_io], help="decompress a z4ai blob"
    )
    d.set_defaults(func=_cmd_decompress)

    i = sub.add_parser("info", help="print header/plane info for a z4ai blob")
    i.add_argument("input", help='input path ("-" for stdin)')
    i.set_defaults(func=_cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
