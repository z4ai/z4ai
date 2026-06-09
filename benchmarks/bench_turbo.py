# SPDX-License-Identifier: Apache-2.0

"""Head-to-head benchmark for the parallel Turbo codec.

Compares, per weight scenario, on identical inputs:
  * raw Zstd        (baseline)
  * z4ai (reference codec, single-pass per plane)
  * z4ai.turbo      (parallel chunked, adaptive transform)
  * zipnn           (if importable)

Each codec is verified byte-exact lossless. Reports ratio + throughput.

    python benchmarks/bench_turbo.py            # 64 MB/scenario
    python benchmarks/bench_turbo.py --mb 256 --level 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so `import z4ai` works

import zstandard as zstd  # noqa: E402

from z4ai import turbo  # noqa: E402

try:
    import z4ai  # noqa: E402

    _HAVE_Z4AI = hasattr(z4ai, "compress")
except Exception:  # noqa: BLE001
    _HAVE_Z4AI = False

try:
    import zipnn  # noqa: E402

    _HAVE_ZIPNN = True
except Exception:  # noqa: BLE001
    _HAVE_ZIPNN = False


def make_bf16(scenario: str, nbytes: int, seed: int = 0) -> bytes:
    """Synthesize bf16 weight bytes for a scenario (bf16 = high 16 bits of fp32)."""
    rng = np.random.default_rng(seed)
    n = nbytes // 2
    if scenario == "iid":
        w = rng.standard_normal(n).astype(np.float32)
        return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if scenario == "structured":
        base = (
            rng.standard_normal(max(1, n // 64)).astype(np.float32).view(np.uint32)
            >> 16
        ).astype(np.uint16)
        return np.tile(base, 64)[:n].tobytes()
    if scenario == "sparse":
        w = rng.standard_normal(n).astype(np.float32)
        w[rng.random(n) < 0.5] = 0.0
        return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    raise ValueError(scenario)


def _best(fn, reps: int):
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _mbps(n, s):
    return (n / 1e6) / s if s > 0 else float("inf")


def run(name, comp, decomp, data, reps):
    blob, ct = _best(lambda: comp(data), reps)
    out, dt = _best(lambda: decomp(blob), reps)
    return {
        "name": name,
        "ratio": len(data) / len(blob),
        "comp": _mbps(len(data), ct),
        "decomp": _mbps(len(data), dt),
        "lossless": bytes(out) == data,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=64.0)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--scenarios", nargs="+", default=["iid", "structured", "sparse"])
    args = p.parse_args(argv)

    print(
        f"Turbo benchmark - {args.mb:g} MB/scenario, level={args.level}, "
        f"reps={args.reps}.  zipnn={'yes' if _HAVE_ZIPNN else 'no'}"
    )

    for scen in args.scenarios:
        data = make_bf16(scen, int(args.mb * 1e6))
        n = len(data)
        print(f"\n== {scen}  ({n/1e6:.1f} MB bf16) ==")
        hdr = f"{'codec':<14}{'ratio':>9}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
        print(hdr)
        print("-" * len(hdr))

        rows = []
        rows.append(
            run(
                f"zstd-{args.level}",
                lambda d: zstd.ZstdCompressor(level=args.level).compress(d),
                lambda b: zstd.ZstdDecompressor().decompress(b),
                data,
                args.reps,
            )
        )
        if _HAVE_Z4AI:
            try:
                rows.append(
                    run(
                        "z4ai",
                        lambda d: z4ai.compress(d, dtype="bf16"),
                        lambda b: z4ai.decompress(b),
                        data,
                        args.reps,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"  z4ai errored: {e!r}")
        rows.append(
            run(
                "z4ai.turbo",
                lambda d: turbo.compress(d, width=2, level=args.level),
                lambda b: turbo.decompress(b),
                data,
                args.reps,
            )
        )
        if _HAVE_ZIPNN:
            try:
                z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
                rows.append(
                    run(
                        "zipnn",
                        lambda d: z.compress(bytearray(d)),
                        lambda b: z.decompress(b),
                        data,
                        args.reps,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"  zipnn errored: {e!r}")

        for r in rows:
            print(
                f"{r['name']:<14}{r['ratio']:>9.3f}{r['comp']:>12.0f}"
                f"{r['decomp']:>14.0f}{('yes' if r['lossless'] else 'NO!'):>10}"
            )

        if _HAVE_ZIPNN:
            by = {r["name"]: r for r in rows}
            t, z = by.get("z4ai.turbo"), by.get("zipnn")
            if t and z:
                print(
                    f"  turbo vs zipnn: ratio {t['ratio']/z['ratio']:.2f}x, "
                    f"comp {t['comp']/z['comp']:.2f}x, decomp {t['decomp']/z['decomp']:.2f}x"
                )


if __name__ == "__main__":
    main()
