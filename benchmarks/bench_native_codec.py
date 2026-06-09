# SPDX-License-Identifier: Apache-2.0
"""Reproducible throughput benchmark for the fused native codec vs zipnn / zstd.

    python -m benchmarks.bench_native_codec --mb 192 --dtypes bf16 fp32 fp64

Measures the fully-fused, multithreaded ``z4ai._native_codec`` (exposed through
``z4ai.chunked``) against zipnn (if installed) and plain zstd-3, reporting ratio
plus compress / decompress GB/s.  All round-trips are asserted byte-exact.

Honest framing (see README "Production guidance"): lossless bf16/fp32 ratio is
near the entropy floor for *all* methods (~1.5x bf16), so the differentiator is
throughput.  This benchmark exists to track that axis.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import z4ai.chunked as ck


def _best(fn, reps=5):
    best, out = float("inf"), None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def _make(dtype: str, n_bytes: int) -> bytes:
    """Realistic-ish weights: small-scale Gaussian (low-entropy exponent)."""
    rng = np.random.default_rng(0)
    if dtype == "bf16":
        x = (rng.standard_normal(n_bytes // 2) * 0.02).astype(np.float32)
        return (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if dtype == "fp32":
        return (rng.standard_normal(n_bytes // 4) * 0.02).astype(np.float32).tobytes()
    if dtype == "fp64":
        return (rng.standard_normal(n_bytes // 8) * 0.02).astype(np.float64).tobytes()
    raise ValueError(dtype)


_ZIPNN_DT = {"bf16": "bfloat16", "fp32": "float32"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=192)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp32", "fp64"])
    args = ap.parse_args()

    print(f"native codec active: {ck._native is not None}")
    try:
        import zipnn  # noqa: F401
        have_zipnn = True
    except Exception:  # noqa: BLE001
        have_zipnn = False
    import zstandard as zstd

    for dt in args.dtypes:
        raw = _make(dt, args.mb << 20)
        mb = len(raw) / 1e6
        tc, blob = _best(lambda: ck.compress(raw, dtype=dt))
        td, out = _best(lambda: ck.decompress(blob))
        assert bytes(out) == raw, "z4ai round-trip mismatch"
        print(f"\n{dt}  {mb:.0f} MB")
        print(f"  z4ai-native  ratio {len(raw)/len(blob):6.3f}  "
              f"comp {mb/tc/1000:5.2f}  decomp {mb/td/1000:5.2f} GB/s")
        if have_zipnn and dt in _ZIPNN_DT:
            import zipnn
            z = zipnn.ZipNN(bytearray_dtype=_ZIPNN_DT[dt])
            tzc, zb = _best(lambda: z.compress(bytearray(raw)))
            tzd, zo = _best(lambda: z.decompress(zb))
            assert bytes(zo) == raw
            print(f"  zipnn        ratio {len(raw)/len(zb):6.3f}  "
                  f"comp {mb/tzc/1000:5.2f}  decomp {mb/tzd/1000:5.2f} GB/s")
        tzs, zsb = _best(lambda: zstd.ZstdCompressor(level=3).compress(raw))
        print(f"  zstd-3       ratio {len(raw)/len(zsb):6.3f}  "
              f"comp {mb/tzs/1000:5.2f}  decomp   n/a GB/s")


if __name__ == "__main__":
    main()
