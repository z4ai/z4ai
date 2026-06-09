# SPDX-License-Identifier: Apache-2.0

"""Benchmark z4ai's palette transform on dequantised low-precision weights.

Most *deployed* LLM weights today are quantised (INT4/INT8/FP8 via GPTQ / AWQ /
bitsandbytes / compressed-tensors) and frequently shipped *dequantised back into
a wide float container* (bf16/fp16/fp32).  Such a tensor holds only a small set
of distinct values, which z4ai captures with a palette (dictionary) transform -
a bijective, fully lossless relabelling to a dense index + codebook.

This harness builds per-tensor-quantised tensors at several bit widths and
reports z4ai's ratio (which auto-selects the palette) vs plain whole-buffer
zstd-19, and -- when installed and verified lossless -- ZipNN.

    python benchmarks/bench_palette.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import z4ai  # noqa: E402
from z4ai import palette  # noqa: E402

try:
    from zipnn import ZipNN

    _HAVE_ZIPNN = True
except Exception:  # noqa: BLE001
    _HAVE_ZIPNN = False


def _view(f32: np.ndarray, dtype: str) -> bytes:
    f32 = np.ascontiguousarray(f32.astype(np.float32))
    if dtype == "bf16":
        return (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if dtype == "fp16":
        return f32.astype(np.float16).tobytes()
    return f32.tobytes()


def _quantize(f32: np.ndarray, levels: int) -> np.ndarray:
    s = np.abs(f32).max() / levels
    return (np.round(f32 / s) * s).astype(np.float32)


def _best(fn, reps=3):
    best = 1e9
    out = None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def main() -> None:
    rng = np.random.default_rng(0)
    n = 8_000_000
    w = rng.standard_normal(n).astype(np.float32)
    zdt = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

    print(
        f"palette benchmark - {n/1e6:.0f}M elements per tensor, per-tensor quantised\n"
    )
    header = f"{'scheme':16s} {'dtype':5s} {'z4ai':>8s} {'zstd-19':>8s}"
    if _HAVE_ZIPNN:
        header += f" {'zipnn':>8s}"
    header += f" {'codec':>8s} {'c MB/s':>8s} {'d MB/s':>8s}"
    print(header)
    print("-" * len(header))

    for name, levels in [("INT4", 7), ("INT6", 31), ("INT8", 127)]:
        q = _quantize(w, levels)
        for dtype in ("bf16", "fp16", "fp32"):
            raw = _view(q, dtype)
            ct, blob = _best(lambda: z4ai.compress(raw, dtype=dtype))
            dt, back = _best(lambda: z4ai.decompress(blob))
            assert bytes(back) == raw, "z4ai lossless check failed"
            z4 = len(raw) / len(blob)
            used = "palette" if palette.peek_is_palette(blob) else "transform"
            z19 = len(raw) / len(zstd.ZstdCompressor(level=19).compress(raw))
            row = f"{name:16s} {dtype:5s} {z4:7.3f}x {z19:7.3f}x"
            if _HAVE_ZIPNN:
                try:
                    Z = ZipNN(bytearray_dtype=zdt[dtype])
                    cz = Z.compress(raw)
                    ll = bytes(Z.decompress(cz)) == raw
                    zr = len(raw) / len(cz)
                    row += f" {zr:6.3f}x{'!' if not ll else ' '}"
                except Exception:  # noqa: BLE001
                    row += f" {'n/a':>8s}"
            row += f" {used:>8s} {len(raw)/1e6/ct:7.0f} {len(raw)/1e6/dt:7.0f}"
            print(row)

    if _HAVE_ZIPNN:
        print("\n(! = ZipNN round-trip was NOT byte-exact on this build)")


if __name__ == "__main__":
    main()
