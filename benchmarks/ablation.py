# SPDX-License-Identifier: Apache-2.0

"""Transform ablation: does bit-field split actually beat byte-grouping (zipnn)?

This is the *decision-driving* experiment for the team. The codec can apply one
of several reversible transforms before the Zstd entropy stage. They all round
to the same backend, so this isolates the transform's effect on ratio:

  * ``raw``        - no transform, Zstd over the interleaved buffer (lower bound)
  * ``byte-group`` - Zstd per byte-plane  == what ZipNN and ``z4ai.transforms`` do
  * ``bit-field``  - Zstd per sign/exponent/mantissa stream (``z4ai.bitfield``)
  * ``zipnn``      - the real ZipNN library, if installed (the target to beat)

All four are verified byte-exact. Run::

    PYTHONPATH=. .venv/bin/python benchmarks/ablation.py --mb 32

The printed verdict tells the core-codec owner which transform to ship as the
default high-ratio mode.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import bytes_per_element, make_weights  # noqa: E402

import zstandard as zstd  # noqa: E402

# Import z4ai internals directly so the ablation does not depend on the (still
# in-flux) public API surface.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from z4ai import backend, bitfield, transforms  # noqa: E402


def _best_time(fn, repeats: int = 3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _mbps(n: int, sec: float) -> float:
    return (n / 1e6) / sec if sec > 0 else float("inf")


# --- transforms expressed as (compress->parts, decompress->bytes) -------------


def t_raw(data: bytes, level: int):
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    comp = cctx.compress(data)
    return [comp], lambda parts: dctx.decompress(parts[0])


def _planes_compress(planes, base_level, escalate):
    return [backend.compress_plane(p, base_level, 0, escalate) for p in planes]


def t_byte_group(data: bytes, level: int, width: int):
    tail_n = transforms.tail_len(len(data), width)
    body = data[: len(data) - tail_n] if tail_n else data
    tail = data[len(data) - tail_n :] if tail_n else b""
    planes = transforms.split_planes(body, width)
    cps = _planes_compress(planes, 3, level)

    def decode(_):
        out = [backend.decompress_plane(c.method, c.data, c.orig_len) for c in cps]
        return transforms.join_planes(out, width, tail)

    return [c.data for c in cps], decode


def t_bit_field(data: bytes, level: int, dtype_name: str):
    sr = bitfield.split_fields(data, dtype=dtype_name)
    streams = sr.as_streams()
    # Sign is already 1bpp-packed & ~random; exponent is the prize (high level);
    # mantissa is near-noise so the backend's probe stores it raw.
    cps = {k: backend.compress_plane(v, 3, 0, level) for k, v in streams.items()}

    def decode(_):
        rebuilt = bitfield.SplitResult(
            spec_name=sr.spec_name,
            count=sr.count,
            sign=backend.decompress_plane(*_unpack(cps["sign"])),
            exponent=backend.decompress_plane(*_unpack(cps["exponent"])),
            mantissa=backend.decompress_plane(*_unpack(cps["mantissa"])),
        )
        return bitfield.join_fields(rebuilt)

    return [c.data for c in cps.values()], decode


def _unpack(cp):
    return cp.method, cp.data, cp.orig_len


def t_zipnn(data: bytes, dtype: str):
    # IMPORTANT: ZipNN's input_format="byte" path is NOT lossless on this build
    # (see benchmarks/zipnn_adapter.py and NOTES/codec-findings-and-zipnn-validity.md).
    # Use the verified-lossless numpy/torch adapter, or omit ZipNN entirely.
    from zipnn_adapter import make_zipnn_codec

    comp_fn, decomp_fn, available = make_zipnn_codec(dtype)
    if not available:
        return None
    obj = comp_fn(data)
    # Represent the opaque compressed object as a single "part" for sizing.
    blob = obj[0] if isinstance(obj, tuple) else obj
    size_blob = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    return [size_blob], lambda _parts, _o=obj: decomp_fn(_o)


def _total(parts) -> int:
    return sum(len(p) for p in parts)


def run_dtype(dtype: str, mb: float, level: int, seed: int, repeats: int):
    bpe = bytes_per_element(dtype)
    n = max(bpe, int(mb * 1e6) // bpe)
    data = make_weights(n, dtype=dtype, seed=seed)
    width = bpe
    spec_name = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype]
    orig = len(data)

    print(f"== {dtype}  ({orig/1e6:.1f} MB, {n:,} elems) ==")
    hdr = f"{'transform':<14}{'ratio':>8}{'comp MB/s':>12}{'decomp MB/s':>13}{'lossless':>10}"
    print(hdr)
    print("-" * len(hdr))

    cases = [
        ("raw-zstd", lambda: t_raw(data, level)),
        ("byte-group", lambda: t_byte_group(data, level, width)),
        ("bit-field", lambda: t_bit_field(data, level, spec_name)),
        ("zipnn", lambda: t_zipnn(data, dtype)),
    ]

    results = {}
    for name, build in cases:
        built = build()
        if built is None:
            continue
        parts, decode = built
        (parts2, _), c_sec = _best_time(lambda b=build: (b()[0], None), repeats)
        out, d_sec = _best_time(lambda d=decode, p=parts: d(p), repeats)
        comp_bytes = _total(parts)
        lossless = bytes(out) == data
        ratio = orig / comp_bytes if comp_bytes else float("inf")
        results[name] = dict(
            ratio=ratio, comp=_mbps(orig, c_sec), decomp=_mbps(orig, d_sec),
            lossless=lossless, bytes=comp_bytes,
        )
        flag = "yes" if lossless else "NO!"
        print(f"{name:<14}{ratio:>8.3f}{results[name]['comp']:>12.0f}"
              f"{results[name]['decomp']:>13.0f}{flag:>10}")

    _verdict(results)
    print()
    return results


def _verdict(r):
    if "bit-field" not in r:
        return
    bf = r["bit-field"]
    for base in ("byte-group", "zipnn", "raw-zstd"):
        if base in r:
            b = r[base]
            dr = (bf["ratio"] / b["ratio"] - 1) * 100
            print(f"  bit-field vs {base:<10}: ratio {dr:+5.1f}%  "
                  f"(smaller output by {(1 - bf['bytes']/b['bytes'])*100:+.1f}%)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=32.0)
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32"])
    p.add_argument("--level", type=int, default=19, help="zstd level for the prize streams")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)

    print(f"z4ai transform ablation - {args.mb:g} MB/dtype, level={args.level}, "
          f"repeats={args.repeats}\n")
    for dtype in args.dtypes:
        run_dtype(dtype, args.mb, args.level, args.seed, args.repeats)


if __name__ == "__main__":
    main()
