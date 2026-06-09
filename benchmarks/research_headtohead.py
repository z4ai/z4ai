# SPDX-License-Identifier: Apache-2.0

"""Honest head-to-head: z4ai vs ZipNN vs zstd vs pcodec on a real checkpoint.

Downloads a real, dense, trained transformer (default ``EleutherAI/pythia-70m``,
166 MB safetensors) and reports lossless compression ratios for:

* **dense float weights only** (the information-theoretic wall — every codec ties
  here, ~1.22-1.25x), and
* the **whole checkpoint** (where the real production win lives — structure,
  embeddings, tied weights; z4ai beats ZipNN).

It also prints the per-field entropy decomposition that explains the wall. This
is the script behind ``NOTES/ratio-ceiling-and-techniques.md``.

Usage::

    python -m benchmarks.research_headtohead                 # pythia-70m
    python -m benchmarks.research_headtohead --model EleutherAI/pythia-160m

Optional comparators (`pip install zipnn pcodec`) are used when importable and
skipped otherwise — z4ai + zstd always run.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np
import zstandard as zstd

from z4ai import codec, safetensors as zst

_ZMAP = {
    "F32": ("fp32", "<f4"),
    "F16": ("fp16", "<f2"),
    "BF16": ("bf16", None),
    "F64": ("fp64", "<f8"),
}


def _download(model: str, dst: str) -> str:
    url = f"https://huggingface.co/{model}/resolve/main/model.safetensors"
    print(f"downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)
    return dst


def _parse(blob: bytes):
    (hlen,) = struct.unpack_from("<Q", blob, 0)
    hdr = json.loads(blob[8 : 8 + hlen])
    data = blob[8 + hlen :]
    bydt = defaultdict(list)
    for name, m in hdr.items():
        if name == "__metadata__":
            continue
        b, e = m["data_offsets"]
        bydt[m["dtype"]].append((name, data[b:e]))
    return bydt


def _zstd(raw: bytes, level: int) -> int:
    return len(zstd.ZstdCompressor(level=level).compress(raw))


def _zipnn(raw: bytes, code: str):
    try:
        from benchmarks.zipnn_adapter import make_zipnn_codec

        c, d, ok = make_zipnn_codec(code)
        if not ok:
            return None
        return len(bytes(c(raw)[0]))
    except Exception:
        return None


def _pco(raw: bytes, npdt, level: int = 12):
    try:
        from pcodec import standalone as pco, ChunkConfig

        arr = np.frombuffer(raw, dtype=npdt)
        return len(pco.simple_compress(arr, ChunkConfig(compression_level=level)))
    except Exception:
        return None


def _entropy_decomp(raw: bytes):
    """Per-field order-0/order-1 entropy of an fp16 buffer (bits/element)."""
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    n = len(u)
    sign, exp, mant = (u >> 15) & 1, (u >> 10) & 0x1F, u & 0x3FF

    def h0(x, nb):
        c = np.bincount(x, minlength=1 << nb).astype(float)
        p = c / c.sum()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    prev = np.empty(n, np.uint8)
    prev[0] = 0
    prev[1:] = exp[:-1].astype(np.uint8)
    tot = 0.0
    for ctx in range(32):
        s = exp[prev == ctx]
        if len(s):
            c = np.bincount(s, minlength=32).astype(float)
            p = c / c.sum()
            p = p[p > 0]
            tot += len(s) * float(-(p * np.log2(p)).sum())
    print(
        f"  entropy/elem: sign H0={h0(sign,1):.3f}  exp H0={h0(exp,5):.3f} "
        f"H1={tot/n:.3f}  mant H0={h0(mant,10):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--path", default="/tmp/z4ai_research_model.safetensors")
    args = ap.parse_args()

    try:
        blob = open(args.path, "rb").read()
    except FileNotFoundError:
        blob = open(_download(args.model, args.path), "rb").read()

    bydt = _parse(blob)
    mb_total = len(blob) / 1e6
    print(
        f"\nmodel {args.model}: {mb_total:.0f} MB, dtypes "
        f"{ {k: len(v) for k, v in bydt.items()} }"
    )

    # Per-dtype dense ratios.
    for dt, items in bydt.items():
        raw = b"".join(x[1] for x in items)
        mb = len(raw) / 1e6
        if mb < 1:
            continue
        code, npdt = _ZMAP.get(dt, (None, None))
        print(f"\n=== {dt}  {mb:.1f} MB ({len(items)} tensors) ===")
        if dt == "F16":
            _entropy_decomp(raw)
        rows = [
            ("zstd-3", _zstd(raw, 3)),
            ("z4ai default", len(codec.compress(raw, dtype=code))),
            (
                "z4ai escalate=19",
                len(codec.compress(raw, dtype=code, escalate_level=19)),
            ),
        ]
        zn = _zipnn(raw, code) if code else None
        if zn:
            rows.insert(1, ("ZipNN", zn))
        if npdt:
            pc = _pco(raw, npdt)
            if pc:
                rows.append(("pcodec L12", pc))
        for name, clen in rows:
            print(f"  {name:20} {mb/(clen/1e6):7.3f}x")

    # Whole-checkpoint (the production number).
    print(f"\n=== WHOLE CHECKPOINT {mb_total:.0f} MB (byte-exact) ===")
    t = time.perf_counter()
    frame = zst.compress_bytes(blob)
    dt = time.perf_counter() - t
    assert zst.decompress_bytes(frame) == blob, "ROUND-TRIP NOT LOSSLESS"
    print(
        f"  z4ai safetensors   {mb_total/(len(frame)/1e6):7.3f}x  "
        f"({mb_total/dt:.0f} MB/s compress, lossless verified)"
    )
    print(f"  zstd-3 whole file  {mb_total/(_zstd(blob,3)/1e6):7.3f}x")
    try:
        import zipnn

        z = zipnn.ZipNN(input_format="byte")
        c = z.compress(bytearray(blob))
        print(f"  ZipNN whole file   {mb_total/(len(bytes(c))/1e6):7.3f}x")
    except Exception as e:  # noqa: BLE001
        print(f"  ZipNN whole file   (unavailable: {e})")


if __name__ == "__main__":
    sys.exit(main())
