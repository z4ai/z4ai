# SPDX-License-Identifier: Apache-2.0

"""Benchmark z4ai against baselines (raw zstd) and ZipNN, if installed.

Reports, per dtype:
  * compression ratio  = original / compressed   (higher is better)
  * compress throughput   (MB/s, higher is better)
  * decompress throughput (MB/s, higher is better)
and verifies every codec is byte-exact (lossless).

Usage:
    python benchmarks/benchmark.py                 # default sizes, all dtypes
    python benchmarks/benchmark.py --mb 64 --dtypes bf16 fp32
    python benchmarks/benchmark.py --level 19 --threads 0

It degrades gracefully: if `z4ai` is not importable yet (core still in progress),
it still runs the zstd and ZipNN reference numbers so we know the target to beat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))           # for `_data`
sys.path.insert(0, str(_HERE.parent))    # repo root, so `import z4ai` works
from _data import bytes_per_element, make_scenario, make_weights  # noqa: E402

import zstandard as zstd  # required dependency


def _time(fn, *, repeats: int = 3):
    """Return (result_of_last_call, best_seconds) over `repeats` runs."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def _mbps(nbytes: int, seconds: float) -> float:
    return (nbytes / 1e6) / seconds if seconds > 0 else float("inf")


# --- codec adapters: (compress_fn, decompress_fn) ------------------------------

def zstd_codec(level: int, threads: int):
    threads = -1 if threads == 0 else threads  # zstandard: -1 = all cores
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    dctx = zstd.ZstdDecompressor()
    return cctx.compress, dctx.decompress


def z4ai_codec(level, threads, dtype):
    import z4ai

    def comp(data):
        return z4ai.compress(data, dtype=dtype, level=level, threads=threads)

    return comp, z4ai.decompress


def zipnn_codec(dtype: str):
    """Return a *lossless* ZipNN (compress, decompress) pair, or None.

    IMPORTANT: ZipNN's ``input_format="byte"`` path is NOT byte-lossless on this
    build (it returns the right length but ~75% wrong bytes), so we must NOT
    benchmark against it - a non-lossless codec is not a valid baseline. We
    delegate to ``zipnn_adapter.make_zipnn_codec``, which drives ZipNN through
    its verified-lossless numpy (fp16/fp32/fp64) or torch (bf16) paths and
    self-tests byte-exactness before returning. If ZipNN cannot losslessly
    handle ``dtype`` here, this returns None and the benchmark omits it.
    """
    try:
        from zipnn_adapter import make_zipnn_codec
    except Exception:  # noqa: BLE001
        return None
    comp, decomp, available = make_zipnn_codec(dtype)
    if not available:
        return None

    def c(data):
        blob, _orig_len = comp(bytes(data))
        return blob

    def d(blob):
        return decomp((blob, 0))

    return c, d


def run_one(name, comp_fn, decomp_fn, data: bytes, repeats: int):
    # Some codecs (e.g. ZipNN's byte path) consume/mutate a bytearray input in
    # place, so hand each compress call a fresh copy and keep `data` pristine as
    # the ground-truth for the losslessness check.
    comp, c_sec = _time(lambda: comp_fn(bytearray(data)), repeats=repeats)
    out, d_sec = _time(lambda: decomp_fn(comp), repeats=repeats)
    lossless = bytes(out) == data
    n = len(data)
    return {
        "name": name,
        "ratio": n / len(comp) if comp else float("inf"),
        "comp_mbps": _mbps(n, c_sec),
        "decomp_mbps": _mbps(n, d_sec),
        "lossless": lossless,
        "comp_bytes": len(comp),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mb", type=float, default=32.0, help="data size per dtype (MB)")
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32"])
    p.add_argument("--level", type=int, default=None, help="zstd level (None=defaults)")
    p.add_argument("--threads", type=int, default=0, help="0=all cores")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default="iid", choices=["iid", "structured", "sparse"],
                   help="weight structure: iid (tie), structured (z4ai wins), sparse")
    args = p.parse_args(argv)

    have_z4ai = _try_import("z4ai")
    have_zipnn = _try_import("zipnn")
    zstd_level = args.level if args.level is not None else 3

    print(f"z4ai benchmark - {args.mb:g} MB/dtype, scenario={args.scenario}, "
          f"level={args.level or 'default'}, threads={args.threads}, repeats={args.repeats}")
    native = False
    if have_z4ai:
        try:
            from z4ai._accel import HAVE_NATIVE as native  # noqa: N811
        except Exception:  # noqa: BLE001
            native = False
    print(f"z4ai={'yes' if have_z4ai else 'MISSING (core in progress)'}  "
          f"native_accel={'yes' if native else 'no (numpy fallback)'}  "
          f"zipnn={'yes' if have_zipnn else 'not installed'}\n")

    for dtype in args.dtypes:
        bpe = bytes_per_element(dtype)
        n_elem = max(bpe, int(args.mb * 1e6) // bpe)
        data = make_scenario(args.scenario, n_elem, dtype=dtype, seed=args.seed)
        print(f"== {dtype}  ({len(data)/1e6:.1f} MB, {n_elem:,} elems) ==")
        header = f"{'codec':<16}{'ratio':>8}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
        print(header)
        print("-" * len(header))

        rows = []
        # baseline: raw zstd on the unmodified buffer
        c, d = zstd_codec(zstd_level, args.threads)
        rows.append(run_one(f"zstd-{zstd_level}", c, d, data, args.repeats))

        if have_z4ai:
            c, d = z4ai_codec(args.level, args.threads, dtype)
            try:
                rows.append(run_one("z4ai", c, d, data, args.repeats))
            except Exception as e:  # noqa: BLE001
                print(f"  z4ai errored: {e!r}")

        if have_zipnn:
            pair = zipnn_codec(dtype)
            if pair is not None:
                try:
                    rows.append(run_one("zipnn", pair[0], pair[1], data, args.repeats))
                except Exception as e:  # noqa: BLE001
                    print(f"  zipnn errored: {e!r}")

        for r in rows:
            print(f"{r['name']:<16}{r['ratio']:>8.3f}{r['comp_mbps']:>12.0f}"
                  f"{r['decomp_mbps']:>14.0f}{('yes' if r['lossless'] else 'NO!'):>10}")

        _verdict(rows)
        print()


def _verdict(rows):
    by = {r["name"]: r for r in rows}
    if "z4ai" not in by:
        return
    z = by["z4ai"]
    for base in ("zipnn", f"zstd-3", "zstd-19"):
        if base in by:
            b = by[base]
            dr = (z["ratio"] / b["ratio"] - 1) * 100
            print(f"  z4ai vs {base}: ratio {dr:+.1f}%, "
                  f"comp {z['comp_mbps']/b['comp_mbps']:.2f}x, "
                  f"decomp {z['decomp_mbps']/b['decomp_mbps']:.2f}x")


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
