# SPDX-License-Identifier: Apache-2.0

"""Headline benchmark: where z4ai *beats* ZipNN, and where it ties.

ZipNN compresses 256 KB chunks independently. z4ai runs Zstd over the whole
tensor with long-distance matching, so any redundancy farther apart than
256 KB - **tied input/output embeddings, duplicated or low-rank layers, large
zero-padding** - is invisible to ZipNN but caught by z4ai.

This script quantifies that on three weight *scenarios* (see
``_data.make_scenario``) and compares against a **verified-lossless** ZipNN
(``zipnn_adapter``), so the numbers are honest:

    PYTHONPATH=. .venv/bin/python benchmarks/scenarios.py --mb 16

Summary of the expected story (fp32, dense i.i.d. excluded from the "win"):
  * i.i.d. dense  -> tie (both ~1.17x); ZipNN faster.
  * structured    -> z4ai wins by 30-40x (whole-tensor window vs 256 KB chunks).
  * sparse        -> tie (~1.85x).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from _data import make_scenario  # noqa: E402
from zipnn_adapter import make_zipnn_codec  # noqa: E402
import zstandard as zstd  # noqa: E402
from z4ai import transforms  # noqa: E402


def _best(fn, repeats: int):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _mbps(n, s):
    return (n / 1e6) / s if s > 0 else float("inf")


def z4ai_bytegroup(data: bytes, width: int, level: int, threads: int):
    """Byte-group transpose + Zstd with long-distance matching (large window).

    ``enable_ldm`` + a large window are what let a single Zstd pass exploit
    cross-256 KB-chunk redundancy that ZipNN's chunked design cannot see.
    """
    params = zstd.ZstdCompressionParameters.from_level(
        level, enable_ldm=True, window_log=27, threads=threads
    )
    cctx = zstd.ZstdCompressor(compression_params=params)
    dctx = zstd.ZstdDecompressor()
    planes = transforms.split_planes(data, width)

    def comp():
        return [cctx.compress(p) for p in planes]

    def decomp(parts):
        out = [dctx.decompress(p) for p in parts]
        return transforms.join_planes(out, width, b"")

    return comp, decomp


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=16.0)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--level", type=int, default=9)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--scenarios", nargs="+", default=["iid", "structured", "sparse"])
    args = p.parse_args(argv)

    width = {"fp32": 4, "fp16": 2}[args.dtype]
    n = max(width, int(args.mb * 1e6) // width)
    cfn, dfn, zok = make_zipnn_codec(args.dtype)

    print(
        f"z4ai vs ZipNN by weight scenario - {args.dtype}, {args.mb:g} MB, "
        f"level={args.level}, threads={args.threads}"
    )
    print(f"ZipNN lossless path available: {zok}\n")
    hdr = (
        f"{'scenario':<12}{'codec':<14}{'ratio':>10}{'comp MB/s':>11}"
        f"{'decomp MB/s':>12}{'lossless':>10}"
    )

    wins = []
    for sc in args.scenarios:
        data = make_scenario(sc, n, dtype=args.dtype, seed=0)
        orig = len(data)
        print(f"== {sc} ({orig/1e6:.1f} MB) ==")
        print(hdr)
        print("-" * len(hdr))

        z_ratio = None
        if zok:
            zc, z_cs = _best(lambda: cfn(data), args.repeats)
            zd, z_ds = _best(lambda c=zc: dfn(c), args.repeats)
            z_blob = zc[0] if isinstance(zc, tuple) else zc
            z_bytes = len(
                z_blob if isinstance(z_blob, (bytes, bytearray)) else bytes(z_blob)
            )
            z_ratio = orig / z_bytes
            print(
                f"{'':<12}{'zipnn':<14}{z_ratio:>10.3f}{_mbps(orig, z_cs):>11.0f}"
                f"{_mbps(orig, z_ds):>12.0f}{('yes' if zd == data else 'NO!'):>10}"
            )

        comp, decomp = z4ai_bytegroup(data, width, args.level, args.threads)
        parts, c_s = _best(comp, args.repeats)
        out, d_s = _best(lambda p=parts: decomp(p), args.repeats)
        z_b = sum(len(x) for x in parts)
        ratio = orig / z_b
        ll = out == data
        print(
            f"{'':<12}{'z4ai':<14}{ratio:>10.3f}{_mbps(orig, c_s):>11.0f}"
            f"{_mbps(orig, d_s):>12.0f}{('yes' if ll else 'NO!'):>10}"
        )

        if z_ratio:
            factor = ratio / z_ratio
            verdict = (
                f"WIN {factor:.1f}x"
                if factor > 1.05
                else ("tie" if factor > 0.95 else f"lose {1/factor:.2f}x")
            )
            print(f"  -> z4ai/zipnn ratio = {factor:.2f}  [{verdict}]")
            wins.append((sc, factor))
        print()

    if wins:
        print("SUMMARY (ratio factor, >1 = z4ai wins):")
        for sc, f in wins:
            print(f"  {sc:<12} {f:>8.2f}x")


if __name__ == "__main__":
    main()
