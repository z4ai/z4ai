# SPDX-License-Identifier: Apache-2.0

"""Head-to-head benchmark for the chunked, multi-threaded z4ai codec.

Compares ``z4ai.chunked`` against plain Zstd and (if importable) ZipNN on
realistic synthetic weights, reporting ratio + compress/decompress throughput
and verifying losslessness.

    python benchmarks/bench_chunked.py --mb 48 --dtypes bf16 fp16 fp32 fp64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import bytes_per_element, make_weights  # noqa: E402

import zstandard as zstd  # noqa: E402
from z4ai import chunked  # noqa: E402


def _best(fn, repeats):
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def _mbps(n, s):
    return (n / 1e6) / s if s > 0 else float("inf")


def _zipnn_pair(dtype):
    try:
        import zipnn
    except Exception:  # noqa: BLE001
        return None
    zd = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype)
    if zd is None:  # ZipNN's byte path has no fp64 etc. - a z4ai coverage win.
        return None
    z = zipnn.ZipNN(bytearray_dtype=zd, input_format="byte")
    return z.compress, z.decompress


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=48.0)
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32", "fp64"])
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    print(f"Chunked z4ai benchmark - {args.mb:g} MB/dtype, repeats={args.repeats}\n")
    for dtype in args.dtypes:
        bpe = bytes_per_element(dtype)
        n_elem = max(bpe, int(args.mb * 1e6) // bpe)
        data = make_weights(n_elem, dtype=dtype, seed=args.seed)
        n = len(data)
        print(f"== {dtype}  ({n/1e6:.1f} MB) ==")
        hdr = f"{'codec':<12}{'ratio':>8}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
        print(hdr)
        print("-" * len(hdr))

        rows = []
        cz = zstd.ZstdCompressor(level=3, threads=-1)
        dz = zstd.ZstdDecompressor()
        comp, cs = _best(lambda: cz.compress(data), args.repeats)
        out, ds = _best(lambda: dz.decompress(comp), args.repeats)
        rows.append(
            ("zstd-3", n / len(comp), _mbps(n, cs), _mbps(n, ds), bytes(out) == data)
        )

        comp, cs = _best(lambda: chunked.compress(data, dtype=dtype), args.repeats)
        out, ds = _best(lambda: chunked.decompress(comp), args.repeats)
        rows.append(
            (
                "z4ai-chunked",
                n / len(comp),
                _mbps(n, cs),
                _mbps(n, ds),
                bytes(out) == data,
            )
        )

        pair = _zipnn_pair(dtype)
        if pair is not None:
            comp, cs = _best(lambda: pair[0](bytearray(data)), args.repeats)
            out, ds = _best(lambda: pair[1](comp), args.repeats)
            rows.append(
                ("zipnn", n / len(comp), _mbps(n, cs), _mbps(n, ds), bytes(out) == data)
            )
        else:
            print(
                "  (zipnn unavailable for this dtype - z4ai covers it, zipnn does not)"
            )

        for name, ratio, cmb, dmb, ok in rows:
            print(
                f"{name:<12}{ratio:>8.4f}{cmb:>12.0f}{dmb:>14.0f}{('yes' if ok else 'NO!'):>10}"
            )

        by = {r[0]: r for r in rows}
        if "zipnn" in by:
            z, k = by["z4ai-chunked"], by["zipnn"]
            print(
                f"  z4ai vs zipnn:  ratio {(z[1]/k[1]-1)*100:+.2f}%, "
                f"comp {z[2]/k[2]:.2f}x, decomp {z[3]/k[3]:.2f}x"
            )
        z, b = by["z4ai-chunked"], by["zstd-3"]
        print(f"  z4ai vs zstd-3: ratio {(z[1]/b[1]-1)*100:+.2f}%\n")


if __name__ == "__main__":
    main()
