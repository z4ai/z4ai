# SPDX-License-Identifier: Apache-2.0

"""Benchmark z4ai's sparse zero-aware mode against ZipNN on pruned weights.

Sparse / pruned checkpoints are where z4ai's explicit nonzero-bitmap transform
pulls decisively ahead of byte-grouping codecs: ZipNN must spend bits on every
zero element's byte planes, while z4ai spends ~1 bit per zero (and at high
sparsity, far less after the bitmap itself compresses).

Usage:
    python benchmarks/bench_sparse.py
    python benchmarks/bench_sparse.py --mb 64 --dtypes bf16 fp32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from _data import f32_to_bf16_bytes  # noqa: E402

import z4ai  # noqa: E402


def _make_sparse(dtype: str, n: int, frac: float, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n).astype(np.float32) * np.float32(0.02)
    f[rng.random(n) < frac] = 0.0
    if dtype == "bf16":
        return f32_to_bf16_bytes(f)
    if dtype == "fp16":
        return f.astype(np.float16).astype("<f2").tobytes()
    if dtype == "fp32":
        return f.astype("<f4").tobytes()
    raise ValueError(dtype)


def _zipnn_codec(dtype: str):
    try:
        import zipnn
    except Exception:
        return None
    name = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype)
    if name is None:
        return None
    z = zipnn.ZipNN(bytearray_dtype=name, input_format="byte")
    return z.compress, z.decompress


def _best(fn, repeats=3):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=32.0)
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    p.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args(argv)

    bpe = {"bf16": 2, "fp16": 2, "fp32": 4}
    have_zipnn = _zipnn_codec("bf16") is not None
    print(
        f"z4ai sparse benchmark - {args.mb:g} MB/case, repeats={args.repeats}, "
        f"zipnn={'yes' if have_zipnn else 'not available'}\n"
    )
    header = (
        f"{'dtype':<6}{'sparsity':>9}{'z4ai ratio':>12}{'zipnn ratio':>13}"
        f"{'win':>9}{'comp MB/s':>11}{'lossless':>10}"
    )
    print(header)
    print("-" * len(header))

    wins = []
    for dtype in args.dtypes:
        for s in args.sparsities:
            n = max(1, int(args.mb * 1e6) // bpe[dtype])
            data = _make_sparse(dtype, n, s)
            mb = len(data) / 1e6

            blob, c_sec = _best(lambda: z4ai.compress(data, dtype=dtype), args.repeats)
            restored = z4ai.decompress(blob)
            lossless = bytes(restored) == data
            z4ai_ratio = len(data) / len(blob)

            zipnn_ratio = float("nan")
            pair = _zipnn_codec(dtype)
            if pair is not None:
                cb = pair[0](bytearray(data))
                zipnn_ratio = len(data) / len(cb)

            win = (
                (z4ai_ratio / zipnn_ratio - 1) * 100
                if zipnn_ratio == zipnn_ratio
                else float("nan")
            )
            if win == win:
                wins.append(win)
            print(
                f"{dtype:<6}{s:>9.2f}{z4ai_ratio:>12.3f}{zipnn_ratio:>13.3f}"
                f"{win:>+8.1f}%{mb / c_sec:>11.0f}{('yes' if lossless else 'NO!'):>10}"
            )

    if wins:
        print(
            f"\nz4ai beats ZipNN on ratio in {sum(w > 0 for w in wins)}/{len(wins)} "
            f"sparse cases (mean {sum(wins) / len(wins):+.1f}%)."
        )


if __name__ == "__main__":
    main()
