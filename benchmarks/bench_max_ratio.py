# SPDX-License-Identifier: Apache-2.0

"""Honest head-to-head: z4ai default vs z4ai ``effort="max"`` vs ZipNN vs zstd.

The ``effort="max"`` tier adds a context-modeling backend (brotli, chunk-parallel)
on top of z4ai's field split.  This is the tier for model *distribution* — compress
once (offline), download/decompress many — where the smallest file wins.  ZipNN
deliberately optimises for speed (order-0 Huffman in 256 KiB chunks); ``max``
trades offline compress time for a meaningfully smaller file.

Run::

    python benchmarks/bench_max_ratio.py
    python benchmarks/bench_max_ratio.py --models distilgpt2 EleutherAI/pythia-70m

Numbers are reproducible from the cached HF models; nothing is hand-edited into
the docs without coming from this script (see COORDINATION.md integrity rule).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make `import z4ai` and `from real_model import ...` work whether this is run as
# `python benchmarks/bench_max_ratio.py` (from the repo root) or as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import z4ai

try:
    import zipnn as _zipnn

    _HAVE_ZIPNN = True
except Exception:  # noqa: BLE001
    _HAVE_ZIPNN = False

import zstandard as _zstd


def _f32_to_bf16_bytes(f32: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    return (((u32 >> 16) & 0xFFFF).astype(np.uint16)).tobytes()


def _timed(fn, reps=1):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return out, (time.perf_counter() - t) / reps


def _bench_one(name: str, raw: bytes, dtype: str) -> None:
    n = len(raw)
    mb = n / 1e6
    print(f"\n=== {name}  ({mb:.1f} MB, {dtype}) ===")
    rows = []

    def add(label, blob, comp_s, dec_fn):
        out, dec_s = _timed(dec_fn, reps=3)
        assert bytes(out) == raw, f"{label} NOT LOSSLESS"
        rows.append((label, n / len(blob), mb / comp_s, mb / dec_s))

    # z4ai default (fast streaming codec)
    blob, cs = _timed(lambda: z4ai.compress(raw, dtype=dtype))
    add("z4ai default", blob, cs, lambda b=blob: z4ai.decompress(b))

    # z4ai max-ratio (brotli/context backend)
    blob, cs = _timed(lambda: z4ai.compress(raw, dtype=dtype, effort="max"))
    add("z4ai max    ", blob, cs, lambda b=blob: z4ai.decompress(b))

    # ZipNN
    if _HAVE_ZIPNN:
        zdt = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
            dtype, "float32"
        )
        comp = _zipnn.ZipNN(bytearray_dtype=zdt)
        blob, cs = _timed(lambda: comp.compress(bytearray(raw)))
        add("zipnn       ", blob, cs, lambda b=blob: comp.decompress(b))

    # plain zstd-3 (reference floor)
    blob, cs = _timed(lambda: _zstd.ZstdCompressor(level=3).compress(raw))
    add(
        "zstd-3      ",
        blob,
        cs,
        lambda b=blob: _zstd.ZstdDecompressor().decompress(b, max_output_size=n),
    )

    print(f"  {'codec':14s} {'ratio':>7s} {'compress':>12s} {'decompress':>12s}")
    base = next((r for r in rows if r[0].startswith("zipnn")), None)
    for label, ratio, c, d in rows:
        tag = ""
        if base is not None and label.startswith("z4ai"):
            tag = f"  ({100*(ratio/base[1]-1):+.1f}% vs zipnn)"
        print(f"  {label:14s} {ratio:6.3f}x {c:10.1f}MB/s {d:10.0f}MB/s{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="*", default=["distilgpt2", "EleutherAI/pythia-70m"]
    )
    ap.add_argument("--max-mb", type=int, default=48)
    ap.add_argument("--dtypes", nargs="*", default=["bf16", "fp32"])
    args = ap.parse_args()

    from real_model import _load_weights  # benchmarks/ on sys.path

    for model in args.models:
        w = _load_weights(model, args.max_mb * 1024 * 1024)
        f32 = np.ascontiguousarray(w, dtype=np.float32)
        for dtype in args.dtypes:
            raw = _f32_to_bf16_bytes(f32) if dtype == "bf16" else f32.tobytes()
            _bench_one(model, raw, dtype)

    # i.i.d. control: the entropy-floor case where NOTHING beats the floor much.
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(8_000_000).astype(np.float32) * 0.02
    _bench_one("i.i.d. gaussian (control)", _f32_to_bf16_bytes(iid), "bf16")


if __name__ == "__main__":
    main()
