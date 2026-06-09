# SPDX-License-Identifier: Apache-2.0

"""Reproducible proof of z4ai's structured-weight win over ZipNN.

Real checkpoints are full of redundancy that repeats far apart: tied
input/output embeddings, duplicated/repeated transformer blocks, EMA and
optimizer-shadow copies, and multi-shard concatenations. ZipNN Huffman-codes
independent ~256 KiB chunks, so it cannot match redundancy beyond one chunk.
z4ai runs Zstd with long-distance matching over the whole plane, so it
deduplicates redundancy megabytes apart.

This script makes that concrete. It also reports the i.i.d. (pure-noise) case,
where ZipNN's tuned entropy coder is a few percent ahead - the deliberate
small-loss / large-win trade.

Run::

    .venv/bin/python benchmarks/bench_structured.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# Allow running as a plain script (`python benchmarks/bench_structured.py`)
# without installing the package: put the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import z4ai

try:
    from zipnn import ZipNN

    _HAVE_ZIPNN = True
except Exception:  # pragma: no cover - zipnn is an optional benchmark dep
    _HAVE_ZIPNN = False


def _to_bf16_u16(f32: np.ndarray) -> np.ndarray:
    """Truncate float32 to bf16, returned as the raw uint16 bit pattern."""
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def _zipnn_ratio(raw: bytes, znd: str) -> float:
    if not _HAVE_ZIPNN:
        return float("nan")
    return len(raw) / len(ZipNN(bytearray_dtype=znd).compress(raw))


def _row(name: str, arr: np.ndarray, znd: str) -> None:
    raw = arr.tobytes()
    t = time.perf_counter()
    blob = z4ai.compress(arr)
    ct = time.perf_counter() - t
    assert bytes(z4ai.decompress(blob)) == raw, f"{name}: round-trip failed"

    zr = len(raw) / len(blob)
    nr = _zipnn_ratio(raw, znd)
    if nr == nr:  # not NaN
        delta = f"{(zr / nr - 1) * 100:+.0f}%"
        verdict = "WIN" if zr > nr else "loss"
        zipnn_col = f"{nr:7.3f}"
    else:
        delta, verdict, zipnn_col = "n/a", "n/a", "   --- "
    print(
        f"  {name:26s} z4ai {zr:8.3f} @ {ct * 1e3:6.1f}ms | "
        f"zipnn {zipnn_col} | {verdict} {delta}"
    )


def main() -> None:
    rng = np.random.default_rng(0)
    print("z4ai vs ZipNN - i.i.d. (small loss) vs structured (large win)\n")

    print("i.i.d. gaussian (ZipNN's tuned coder edges ratio; we trade it):")
    g32 = rng.standard_normal(2000 * 2000).astype(np.float32)
    _row("fp32 i.i.d.", g32, "float32")
    _row("bf16 i.i.d.", _to_bf16_u16(g32), "bfloat16")

    print("\nduplicated layers (8x a 4 MB block - tied/repeated weights):")
    block = rng.standard_normal(1_000_000).astype(np.float32)
    tiled = np.tile(block, 8)
    _row("fp32 duplicated x8", tiled, "float32")
    _row("bf16 duplicated x8", _to_bf16_u16(tiled), "bfloat16")

    print("\ntied embedding (input == output embedding, concatenated):")
    emb = rng.standard_normal(2_000_000).astype(np.float32)
    tied = np.concatenate([emb, emb])
    _row("fp32 tied embedding", tied, "float32")

    if not _HAVE_ZIPNN:
        print("\n(zipnn not installed: `pip install zipnn` for the comparison)")


if __name__ == "__main__":
    main()
