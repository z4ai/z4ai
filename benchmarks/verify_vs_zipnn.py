# SPDX-License-Identifier: Apache-2.0

"""Authoritative head-to-head ratio matrix: z4ai vs ZipNN.

Runs every (dtype x scenario) pair on identical inputs, verifies z4ai is
byte-exact lossless, and prints a ratio table plus a one-line verdict per row.
This is the single script behind the README's competitive claims - run it to
reproduce them on your hardware/data:

    python benchmarks/verify_vs_zipnn.py
    python benchmarks/verify_vs_zipnn.py --mb 64 --max-ratio

ZipNN supports bf16/fp16/fp32 only; fp64 (and integer dtypes) are a z4ai-only
capability, shown for completeness with no ZipNN column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import z4ai  # noqa: E402

try:
    import zipnn  # noqa: E402

    _HAVE_ZIPNN = True
except Exception:  # noqa: BLE001
    _HAVE_ZIPNN = False

_ZIPNN_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "fp64": 8}


def make(dtype: str, scenario: str, nbytes: int, seed: int = 0) -> bytes:
    """Synthesize weight bytes mimicking a trained-network distribution."""
    rng = np.random.default_rng(seed)
    n = nbytes // _BYTES[dtype]
    if dtype == "bf16":
        base = (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(
            np.uint16
        )
    elif dtype == "fp16":
        base = rng.standard_normal(n).astype(np.float16).view(np.uint16)
    elif dtype == "fp32":
        base = rng.standard_normal(n).astype(np.float32).view(np.uint32)
    else:
        base = rng.standard_normal(n).astype(np.float64).view(np.uint64)
    arr = base.copy()
    if scenario == "structured":  # tied/duplicated layers: a tile repeated
        k = max(1, n // 64)
        arr = np.tile(base[:k], 64)[:n].copy()
    elif scenario == "sparse":  # pruned model: 50% exact zeros
        arr[rng.random(n) < 0.5] = 0
    return arr.tobytes()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=16.0, help="bytes per case (MB)")
    p.add_argument(
        "--max-ratio",
        action="store_true",
        help="also compress at level 19 (enables the rANS entropy path)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    dtypes = ["bf16", "fp16", "fp32", "fp64"]
    scenarios = ["iid", "structured", "sparse"]
    nbytes = int(args.mb * 1e6)

    print(
        f"z4ai vs ZipNN ratio matrix - {args.mb:g} MB/case, "
        f"max_ratio={'on' if args.max_ratio else 'off'}, "
        f"zipnn={'yes' if _HAVE_ZIPNN else 'not installed'}"
    )
    hdr = f"{'dtype':<6}{'scenario':<12}{'z4ai':>9}{'zipnn':>9}{'lossless':>10}{'verdict':>9}"
    print(hdr)
    print("-" * len(hdr))

    wins = ties = comparable = 0
    _TIE_BAND = 0.005  # +/-0.5%: within the lossless entropy floor = a tie
    for dt in dtypes:
        for scen in scenarios:
            data = make(dt, scen, nbytes, args.seed)
            n = len(data)
            blob = z4ai.compress(data, dtype=dt)
            lossless = bytes(z4ai.decompress(blob)) == data
            best = n / len(blob)
            if args.max_ratio:
                blob19 = z4ai.compress(data, dtype=dt, level=19)
                lossless = lossless and bytes(z4ai.decompress(blob19)) == data
                best = max(best, n / len(blob19))

            zr = float("nan")
            verdict = "-"
            if _HAVE_ZIPNN and dt in _ZIPNN_DTYPE:
                z = zipnn.ZipNN(bytearray_dtype=_ZIPNN_DTYPE[dt], input_format="byte")
                zr = n / len(z.compress(bytearray(data)))
                comparable += 1
                # Differences within +/-0.5% are at the lossless entropy floor
                # where neither codec can do meaningfully better - report those
                # as TIE rather than overstating a microscopic win/loss.
                rel = best / zr - 1.0
                if rel > _TIE_BAND:
                    verdict = "WIN"
                    wins += 1
                elif rel < -_TIE_BAND:
                    verdict = "lose"
                else:
                    verdict = "tie"
                    ties += 1

            zr_str = f"{zr:9.3f}" if zr == zr else f"{'n/a':>9}"
            ok = "yes" if lossless else "NO!"
            print(f"{dt:<6}{scen:<12}{best:>9.3f}{zr_str}{ok:>10}{verdict:>9}")

    if comparable:
        losses = comparable - wins - ties
        print(
            f"\nz4ai vs ZipNN on ratio: {wins} WIN / {ties} tie / {losses} lose "
            f"of {comparable} comparable cases (tie = within +/-0.5%, the lossless "
            f"entropy floor). fp64 + integer dtypes are z4ai-only (ZipNN unsupported)."
        )


if __name__ == "__main__":
    main()
