# SPDX-License-Identifier: Apache-2.0

"""Honest benchmark on a *real* trained checkpoint - no torch required.

``benchmark_real.py`` silently falls back to synthetic Gaussian data when torch
/ safetensors are missing, which is misleading (Gaussian noise is the entropy
floor where everything ties). This script instead downloads a real PyTorch
``.bin`` checkpoint (a plain ZIP of raw little-endian fp32 tensor blobs),
extracts the tensor bytes directly with the stdlib ``zipfile`` module, and
benchmarks the genuine trained weights.

Why this matters for the project's claims: z4ai's large win comes from
*long-range exact redundancy* (duplicated/repeated layers, tied embeddings
stored twice, multi-shard concatenation). A single dense checkpoint like
bert-tiny is dominated by one embedding table with no such duplication, so it
is close to the i.i.d. case - expect a *small loss* vs ZipNN here, and reserve
the big win for genuinely redundant artifacts. Reporting both keeps us honest.

Usage::

    python benchmarks/bench_real_checkpoint.py
    python benchmarks/bench_real_checkpoint.py --url <hf .bin resolve URL>
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import z4ai

_DEFAULT_URL = (
    "https://huggingface.co/prajjwal1/bert-tiny/resolve/main/pytorch_model.bin"
)


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _extract_fp32_bytes(blob: bytes) -> bytes:
    """Concatenate the raw fp32 tensor blobs stored inside a PyTorch .bin ZIP."""
    zf = zipfile.ZipFile(io.BytesIO(blob))
    parts = [
        zf.read(name)
        for name in zf.namelist()
        if "/data/" in name and zf.getinfo(name).file_size % 4 == 0
    ]
    raw = b"".join(parts)
    return raw[: len(raw) // 4 * 4]


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=_DEFAULT_URL, help="checkpoint .bin URL")
    args = ap.parse_args()

    try:
        blob = _download(args.url)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"download failed ({exc}); this benchmark needs network access.")
        return
    raw = _extract_fp32_bytes(blob)
    print(f"real checkpoint: {len(raw) / 1e6:.2f} MB of fp32 tensor bytes\n")

    import zstandard as zstd

    zb, zc = _timed(lambda: z4ai.compress(raw, dtype="fp32"))
    zo, zd = _timed(lambda: z4ai.decompress(zb))
    assert bytes(zo) == raw, "z4ai round-trip failed"
    z4ai_ratio = len(raw) / len(zb)

    sb = zstd.ZstdCompressor(level=3).compress(raw)
    zstd_ratio = len(raw) / len(sb)

    try:
        from zipnn import ZipNN

        nb, _ = _timed(
            lambda: ZipNN(bytearray_dtype="float32").compress(bytearray(raw))
        )
        zipnn_ratio = len(raw) / len(nb)
    except Exception:
        zipnn_ratio = float("nan")

    print(f"  zstd-3 : ratio {zstd_ratio:.3f}")
    if zipnn_ratio == zipnn_ratio:
        print(f"  zipnn  : ratio {zipnn_ratio:.3f}")
    print(
        f"  z4ai   : ratio {z4ai_ratio:.3f}   compress {zc:.0f}ms  decompress {zd:.0f}ms"
    )
    if zipnn_ratio == zipnn_ratio:
        delta = (z4ai_ratio / zipnn_ratio - 1) * 100
        verdict = "WIN" if delta > 0 else "loss"
        print(f"\n  vs ZipNN on real dense weights: {verdict} {delta:+.1f}% ratio")
        print("  (single dense checkpoint ~ i.i.d.; expect a small loss here. The")
        print("   large win is on duplicated/tied/multi-shard weights - see")
        print("   benchmarks/bench_structured.py.)")


if __name__ == "__main__":
    main()
