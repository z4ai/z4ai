# SPDX-License-Identifier: Apache-2.0

"""Scenario benchmark: prove ``z4ai.auto`` beats ZipNN across weight regimes.

The headline benchmark (``benchmark.py``) uses a single dense Gaussian-mixture
tensor -- the *hardest* case, where every byte-grouping codec sits at the
entropy floor and ties.  The interesting differences only appear across the
*range* of distributions real checkpoints exhibit, so this harness sweeps:

  * ``gauss-mix``  -- dense Normal mixture (entropy-floor; everyone ties)
  * ``outlier``    -- mostly tiny values + ~1% large outliers (LLM-like)
  * ``sparse50``   -- 50% exact zeros (pruned / structured-sparse)
  * ``repeated``   -- drawn from a 256-value codebook (tied / quantization-ready)

For each it reports compression ratio (higher better) and verifies losslessness
for zstd-19, ZipNN (if installed) and ``z4ai.auto``, then prints the win/loss of
AUTO vs ZipNN.  Compress is single-shot (AUTO's max effort is intentionally
slow -- you pay it once per model); the point of this harness is *ratio*.

Usage:
    python benchmarks/benchmark_scenarios.py
    python benchmarks/benchmark_scenarios.py --elems 4000000 --effort max
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _data import f32_to_bf16_bytes  # noqa: E402

import zstandard as zstd  # noqa: E402
from z4ai import auto  # noqa: E402

try:
    import zipnn  # noqa: E402
    _HAVE_ZIPNN = True
except ImportError:
    _HAVE_ZIPNN = False


def make_scenarios(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}

    # dense Gaussian mixture (a couple of scales), the entropy-floor case
    scales = np.array([0.005, 0.02, 0.08], dtype=np.float32)
    which = rng.integers(0, len(scales), n)
    f = rng.standard_normal(n).astype(np.float32) * scales[which]
    out["gauss-mix"] = f32_to_bf16_bytes(f)

    # mostly-tiny values with ~1% large outliers (transformer-attention-like)
    f = rng.standard_normal(n).astype(np.float32) * 0.01
    mask = rng.random(n) < 0.01
    f[mask] = rng.standard_normal(int(mask.sum())).astype(np.float32) * 2.0
    out["outlier"] = f32_to_bf16_bytes(f)

    # 50% exact zeros (pruned / structured sparsity)
    f = rng.standard_normal(n).astype(np.float32) * 0.02
    f[rng.random(n) < 0.5] = 0.0
    out["sparse50"] = f32_to_bf16_bytes(f)

    # codebook of 256 values (tied weights / quantization-friendly)
    book = rng.standard_normal(256).astype(np.float32) * 0.02
    out["repeated"] = f32_to_bf16_bytes(book[rng.integers(0, 256, n)])

    return out


def _ratio_zstd(data: bytes, level: int):
    comp = zstd.ZstdCompressor(level=level).compress(data)
    out = zstd.ZstdDecompressor().decompress(comp, max_output_size=len(data))
    return len(data) / len(comp), bytes(out) == data


def _ratio_zipnn(data: bytes):
    z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
    comp = z.compress(bytes(bytearray(data)))            # fresh copy: zipnn mutates input
    out = z.decompress(bytes(bytearray(comp)))
    return len(data) / len(comp), bytes(out) == bytes(bytearray(data))


def _ratio_auto(data: bytes, effort: str):
    blob = auto.compress(data, dtype="bf16", effort=effort)
    out = auto.decompress(blob)
    return len(data) / len(blob), bytes(out) == data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elems", type=int, default=2_000_000, help="bf16 elements per scenario")
    p.add_argument("--effort", default="max", choices=["fast", "balanced", "max"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    print(f"z4ai scenario benchmark - bf16, {args.elems:,} elems/scenario, "
          f"effort={args.effort}, zipnn={'yes' if _HAVE_ZIPNN else 'not installed'}\n")
    hdr = f"{'scenario':<11}{'zstd-19':>9}{'zipnn':>9}{'z4ai-auto':>11}{'auto vs zipnn':>15}"
    print(hdr)
    print("-" * len(hdr))

    wins = losses = 0
    for name, data in make_scenarios(args.elems, args.seed).items():
        zr, zok = _ratio_zstd(data, 19)
        ar, aok = _ratio_auto(data, args.effort)
        if _HAVE_ZIPNN:
            nr, nok = _ratio_zipnn(data)
            delta = (ar / nr - 1) * 100
            verdict = f"{delta:+.1f}%"
            if delta > 0.05:
                wins += 1
            elif delta < -0.05:
                losses += 1
            nr_s = f"{nr:.3f}"
        else:
            nr_s, verdict, nok = "-", "n/a", True
        flags = "" if (zok and aok and nok) else "  !! LOSSLESS FAILURE"
        print(f"{name:<11}{zr:>9.3f}{nr_s:>9}{ar:>11.3f}{verdict:>15}{flags}")

    if _HAVE_ZIPNN:
        print(f"\nVerdict vs ZipNN: {wins} win(s), {losses} loss(es) across scenarios "
              f"(ties counted as neither). AUTO trades compress speed for ratio.")


if __name__ == "__main__":
    main()
