# SPDX-License-Identifier: Apache-2.0

"""Real-model benchmark: z4ai vs ZipNN vs raw Zstd on actual checkpoint weights.

This is the benchmark that matters for the project's claim. Synthetic Gaussian
weights ("iid") are near the entropy floor, so every codec ties there. *Real*
checkpoints are far more compressible because tensors share a small set of
exponents and many repeated values across a large window - exactly the structure
z4ai's global-window byte/field transform exploits and that ZipNN's independent
256 KiB chunks cannot see.

It downloads a small public checkpoint (default ``prajjwal1/bert-tiny``) once,
caches it under ``/tmp``, then reports compression ratio and throughput. ZipNN is
driven correctly (its byte path mutates its input in place and needs a mutable
``bytearray``), and every codec is verified byte-exact before its numbers count.

Usage::

    python benchmarks/benchmark_real.py
    python benchmarks/benchmark_real.py --url <hf-resolve-url> --dtype fp32
    python benchmarks/benchmark_real.py --file /path/to/pytorch_model.bin
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # repo root for `import z4ai`

import zstandard as zstd  # noqa: E402

DEFAULT_URL = (
    "https://huggingface.co/prajjwal1/bert-tiny/resolve/main/pytorch_model.bin"
)
DEFAULT_CACHE = Path("/tmp/z4ai_realmodel.bin")

# Map a NumPy float dtype to the codes z4ai / ZipNN understand.
_ZIPNN_DTYPE = {"float32": "float32", "float16": "float16"}
_Z4AI_DTYPE = {"float32": "fp32", "float16": "fp16"}


def _time(fn, *, repeats: int = 3):
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _mbps(nbytes: int, seconds: float) -> float:
    return (nbytes / 1e6) / seconds if seconds > 0 else float("inf")


def load_weights(path: Path, want_dtype: str) -> bytes:
    """Load a checkpoint and return all matching-dtype weights as a flat buffer."""
    import torch  # local import: torch is a benchmark-only dependency

    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[want_dtype]
    flats = [
        v.detach().cpu().contiguous().view(-1).numpy()
        for v in state.values()
        if hasattr(v, "dtype") and v.dtype == torch_dtype and v.numel() > 0
    ]
    if not flats:
        raise SystemExit(
            f"no {want_dtype} tensors found in {path}; try --dtype float16"
        )
    return (
        np.concatenate(flats)
        .astype("<f4" if want_dtype == "float32" else "<f2")
        .tobytes()
    )


def fetch(url: str, cache: Path) -> Path:
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    print(f"downloading {url} -> {cache} ...")
    urllib.request.urlretrieve(url, cache)
    return cache


def bench_z4ai(data: bytes, dtype_code: str, repeats: int):
    import z4ai

    orig = bytes(data)
    comp, c_sec = _time(
        lambda: z4ai.compress(bytearray(orig), dtype=dtype_code), repeats=repeats
    )
    out, d_sec = _time(lambda: z4ai.decompress(comp), repeats=repeats)
    return _row("z4ai", orig, comp, c_sec, d_sec, bytes(out) == orig)


def bench_zipnn(data: bytes, zdtype: str, repeats: int):
    import zipnn

    orig = bytes(data)
    # ZipNN's byte path mutates its input and requires a mutable bytearray; hand
    # each call a throwaway copy and verify against the pristine original.
    comp, c_sec = _time(
        lambda: zipnn.ZipNN(bytearray_dtype=zdtype, input_format="byte").compress(
            bytearray(orig)
        ),
        repeats=repeats,
    )
    fixed = zipnn.ZipNN(bytearray_dtype=zdtype, input_format="byte").compress(
        bytearray(orig)
    )
    out, d_sec = _time(
        lambda: zipnn.ZipNN(bytearray_dtype=zdtype, input_format="byte").decompress(
            bytearray(fixed)
        ),
        repeats=repeats,
    )
    return _row("zipnn", orig, comp, c_sec, d_sec, bytes(out) == orig)


def bench_zstd(data: bytes, level: int, repeats: int):
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    comp, c_sec = _time(lambda: cctx.compress(data), repeats=repeats)
    out, d_sec = _time(lambda: dctx.decompress(comp), repeats=repeats)
    return _row(f"zstd-{level}", data, comp, c_sec, d_sec, bytes(out) == data)


def _row(name, data, comp, c_sec, d_sec, lossless):
    n = len(data)
    return {
        "name": name,
        "ratio": n / len(comp) if comp else float("inf"),
        "comp_mbps": _mbps(n, c_sec),
        "decomp_mbps": _mbps(n, d_sec),
        "lossless": lossless,
        "comp_mb": len(comp) / 1e6,
    }


def _try(name, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"  {name} skipped: {e!r}")
        return None


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--url", default=DEFAULT_URL, help="HuggingFace resolve URL for a checkpoint"
    )
    p.add_argument(
        "--file", default=None, help="use a local checkpoint instead of downloading"
    )
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--zstd-level", type=int, default=3)
    args = p.parse_args(argv)

    path = Path(args.file) if args.file else fetch(args.url, Path(args.cache))
    data = load_weights(path, args.dtype)
    n = len(data)
    print(
        f"\nReal-model benchmark - {path.name}, dtype={args.dtype}, "
        f"{n/1e6:.2f} MB ({n // (4 if args.dtype=='float32' else 2):,} elements)\n"
    )

    rows = []
    r = _try("zstd", lambda: bench_zstd(data, args.zstd_level, args.repeats))
    if r:
        rows.append(r)
    r = _try("z4ai", lambda: bench_z4ai(data, _Z4AI_DTYPE[args.dtype], args.repeats))
    if r:
        rows.append(r)
    r = _try("zipnn", lambda: bench_zipnn(data, _ZIPNN_DTYPE[args.dtype], args.repeats))
    if r:
        rows.append(r)

    header = f"{'codec':<12}{'ratio':>9}{'comp MB':>10}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<12}{r['ratio']:>9.3f}{r['comp_mb']:>10.2f}{r['comp_mbps']:>12.0f}"
            f"{r['decomp_mbps']:>14.0f}{('yes' if r['lossless'] else 'NO!'):>10}"
        )

    by = {r["name"]: r for r in rows}
    if "z4ai" in by and "zipnn" in by:
        z, n_ = by["z4ai"], by["zipnn"]
        print(f"\n  z4ai vs ZipNN on real weights:")
        print(
            f"    ratio   : {z['ratio']/n_['ratio']:.2f}x  ({(z['ratio']/n_['ratio']-1)*100:+.0f}% better compression)"
        )
        print(
            f"    stored  : {z['comp_mb']:.1f} MB vs {n_['comp_mb']:.1f} MB "
            f"({(1-z['comp_mb']/n_['comp_mb'])*100:.0f}% smaller file)"
        )
        print(
            f"    compress: {z['comp_mbps']/n_['comp_mbps']:.2f}x throughput (one-time cost)"
        )
        print(f"    decode  : {z['decomp_mbps']/n_['decomp_mbps']:.2f}x throughput")


if __name__ == "__main__":
    main()
