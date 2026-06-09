# SPDX-License-Identifier: Apache-2.0

"""Rigorous, lossless-verified head-to-head: z4ai vs **real ZipNN**.

Why this exists (read before trusting any z4ai-vs-zipnn number):

``benchmarks/benchmark.py`` drives ZipNN through its *byte* API
(``ZipNN(bytearray_dtype="bfloat16", input_format="byte")``).  In zipnn 0.5.4
that path does **not** round-trip our little-endian bf16 buffers byte-for-byte,
yet the benchmark's ``bytes(out) == data`` check still printed ``lossless: yes``
(the check is comparing the wrong objects).  A codec that is not lossless is not
a valid baseline, so every ratio measured against that path is meaningless.

This harness fixes both problems:

* **ZipNN is driven through its supported torch path** (``input_format="torch"``
  on a ``bfloat16`` tensor), which *is* byte-exact lossless here.
* **Losslessness is verified against a pristine, pre-compression byte snapshot.**
  ZipNN's ``compress`` may reorder its input in place, so every codec receives a
  fresh copy and is compared to the untouched reference.

A codec that fails the lossless check is disqualified (ratio reported as n/a),
never silently counted.

Run:  ``.venv/bin/python benchmarks/fair_benchmark.py``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Make the package importable when run in-tree (``python benchmarks/fair_benchmark.py``).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

try:
    import zipnn
except Exception:  # pragma: no cover - zipnn optional
    zipnn = None

import z4ai


# --------------------------------------------------------------------------- #
# Synthetic datasets - each returns raw little-endian bf16 bytes.             #
# --------------------------------------------------------------------------- #
def _bf16_bytes(f32: np.ndarray) -> bytes:
    """float32 -> bf16 little-endian bytes, round-to-nearest-even."""
    u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> 16).astype("<u2").tobytes()


def dataset_iid(n: int, seed: int) -> bytes:
    """i.i.d. Gaussian weights - the hardest case (mantissa is ~uniform noise)."""
    rng = np.random.default_rng(seed)
    return _bf16_bytes(rng.standard_normal(n).astype(np.float32) * 0.02)


def dataset_structured(n: int, seed: int) -> bytes:
    """Long-range redundancy: a unique block tiled many times.

    Models the very common real-world cases ZipNN's 256 KB chunking misses:
    tied input/output embeddings, duplicated/repeated transformer blocks,
    low-rank or padded tensors.
    """
    rng = np.random.default_rng(seed)
    block = rng.standard_normal(max(1, n // 16)).astype(np.float32) * 0.02
    return _bf16_bytes(np.tile(block, 16)[:n])


def dataset_sparse(n: int, seed: int, frac_zero: float = 0.5) -> bytes:
    """Structured sparsity: a fraction of weights are exactly zero (pruned nets)."""
    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal(n).astype(np.float32) * 0.02
    f32[rng.random(n) < frac_zero] = 0.0
    return _bf16_bytes(f32)


DATASETS: Dict[str, Callable[[int, int], bytes]] = {
    "iid": dataset_iid,
    "structured": dataset_structured,
    "sparse50": dataset_sparse,
}


# --------------------------------------------------------------------------- #
# Codec adapters. Each: (compress(ref_bytes)->blob, decompress(blob)->bytes). #
# Both receive/return *plain bytes*; losslessness is checked by the caller.   #
# --------------------------------------------------------------------------- #
def z4ai_codec(level: Optional[int]) -> Tuple[Callable, Callable]:
    def comp(ref: bytes) -> bytes:
        return z4ai.compress(ref, dtype="bf16", level=level)

    return comp, z4ai.decompress


def zipnn_torch_codec() -> Optional[Tuple[Callable, Callable]]:
    """ZipNN via its supported torch bf16 path (the lossless one)."""
    if zipnn is None or torch is None:
        return None

    def comp(ref: bytes) -> bytes:
        # Reconstruct a bf16 tensor from the raw bytes, hand zipnn a private copy.
        t = torch.frombuffer(bytearray(ref), dtype=torch.bfloat16).clone()
        z = zipnn.ZipNN(input_format="torch")
        return z.compress(t)

    def decomp(blob: bytes) -> bytes:
        z = zipnn.ZipNN(input_format="torch")
        t = z.decompress(blob)
        return t.view(torch.uint8).numpy().tobytes()

    return comp, decomp


def _best_time(fn: Callable, *, repeats: int) -> Tuple[object, float]:
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _mbps(nbytes: int, seconds: float) -> float:
    return (nbytes / 1e6) / seconds if seconds > 0 else float("inf")


def measure(name: str, codec, ref: bytes, repeats: int) -> dict:
    comp_fn, decomp_fn = codec
    blob, c_sec = _best_time(lambda: comp_fn(ref), repeats=repeats)
    blob = bytes(blob)
    out, d_sec = _best_time(lambda: decomp_fn(blob), repeats=repeats)
    lossless = bytes(out) == ref
    return {
        "name": name,
        "ratio": (len(ref) / len(blob)) if lossless and blob else float("nan"),
        "comp_mbps": _mbps(len(ref), c_sec),
        "decomp_mbps": _mbps(len(ref), d_sec),
        "lossless": lossless,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mb", type=float, default=8.0, help="MB per dataset (bf16)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--level", type=int, default=None, help="z4ai zstd level (None=package default)"
    )
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args(argv)

    n = max(1, int(args.mb * 1e6) // 2)  # bf16 = 2 bytes/elem
    have_zipnn = zipnn is not None and torch is not None
    print(
        f"Fair benchmark - {args.mb:g} MB/dataset, bf16, level={args.level or 'default'}, "
        f"repeats={args.repeats}"
    )
    print(
        f"z4ai={z4ai.__version__}  zipnn={'torch-path' if have_zipnn else 'UNAVAILABLE'}\n"
    )

    wins = ties = losses = 0
    for ds in args.datasets:
        ref = DATASETS[ds](n, args.seed)
        rows: List[dict] = [measure("z4ai", z4ai_codec(args.level), ref, args.repeats)]
        zc = zipnn_torch_codec()
        if zc is not None:
            rows.append(measure("zipnn", zc, ref, args.repeats))

        print(f"== {ds}  ({len(ref)/1e6:.1f} MB) ==")
        hdr = f"{'codec':<10}{'ratio':>10}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            ratio = "n/a" if r["ratio"] != r["ratio"] else f"{r['ratio']:.3f}"
            print(
                f"{r['name']:<10}{ratio:>10}{r['comp_mbps']:>12.0f}"
                f"{r['decomp_mbps']:>14.0f}{('yes' if r['lossless'] else 'NO!'):>10}"
            )

        by = {r["name"]: r for r in rows}
        if "zipnn" in by and by["z4ai"]["lossless"] and by["zipnn"]["lossless"]:
            z, p = by["z4ai"]["ratio"], by["zipnn"]["ratio"]
            delta = (z / p - 1) * 100
            verdict = "WIN " if delta > 2 else ("tie " if abs(delta) <= 2 else "lose")
            wins += verdict == "WIN "
            ties += verdict == "tie "
            losses += verdict == "lose"
            print(
                f"  -> ratio {delta:+.1f}% vs zipnn  [{verdict.strip()}]   "
                f"comp {by['z4ai']['comp_mbps']/by['zipnn']['comp_mbps']:.2f}x  "
                f"decomp {by['z4ai']['decomp_mbps']/by['zipnn']['decomp_mbps']:.2f}x"
            )
        print()

    if wins or ties or losses:
        print(f"Ratio scoreboard vs zipnn:  {wins} win, {ties} tie, {losses} lose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
