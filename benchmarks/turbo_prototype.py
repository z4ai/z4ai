# SPDX-License-Identifier: Apache-2.0

"""Self-contained PROOF-OF-CONCEPT: parallel chunked z4ai that beats ZipNN on
compression throughput in pure Python.

This is a reference for the core team (codec.py owners), not a public API.  It
demonstrates that the throughput gap to ZipNN is a *parallelism* gap, not a
language gap: `zstandard` releases the GIL, so chunked compression across a
thread pool scales, and the only single-threaded bottleneck (the bit-field
split) is fixed by dropping the uint64 widening.

Measured recipe (16-core macOS, see NOTES/profiling-throughput.md):
  * zstd parallel-chunked compress ceiling: ~6.4 GB/s  (> ZipNN ~4.3 GB/s)
  * fast byte/uint16 split:                 ~3.5 GB/s  (vs uint64 split 1.2 GB/s)

Container is a tiny independent format (magic b"ZTB1"); fully lossless,
verified byte-exact here against the same inputs ZipNN sees.

Run:  python benchmarks/turbo_prototype.py [--mb 64] [--dtypes bf16 fp32]
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import bytes_per_element, make_weights  # noqa: E402

MAGIC = b"ZTB1"
_BLOCK = 1 << 20  # 1 MiB blocks - best throughput/ratio tradeoff measured


# --------------------------------------------------------------------------- #
# Fast, allocation-light field/byte split (no uint64 widening)
# --------------------------------------------------------------------------- #
def _split_streams(raw: bytes, dtype: str):
    """Return a list of homogeneous uint8 streams for the dtype.

    bf16 -> sign(packed), exponent, mantissa (bit-field, native uint16 ops).
    others -> byte planes via a strided reshape (no per-element python).
    """
    if dtype == "bf16":
        u = np.frombuffer(raw, dtype="<u2")
        # Native-width ops: uint16 >> small_int stays uint16, masks fit in uint8.
        sign = np.packbits((u >> 15).astype(np.uint8))
        exponent = ((u >> 7) & 0xFF).astype(np.uint8)
        mantissa = (u & 0x7F).astype(np.uint8)
        return ["bf16", sign.tobytes(), exponent.tobytes(), mantissa.tobytes()]
    width = bytes_per_element(dtype)
    n = len(raw) // width
    arr = np.frombuffer(raw, dtype=np.uint8, count=n * width).reshape(n, width)
    return ["bp%d" % width] + [
        np.ascontiguousarray(arr[:, j]).tobytes() for j in range(width)
    ]


def _join_streams(kind: str, streams, n_elems: int) -> bytes:
    if kind == "bf16":
        sign_b, exp_b, mant_b = streams
        sign = np.unpackbits(np.frombuffer(sign_b, np.uint8), count=n_elems).astype(np.uint16)
        exp = np.frombuffer(exp_b, np.uint8).astype(np.uint16)
        mant = np.frombuffer(mant_b, np.uint8).astype(np.uint16)
        u = (sign << 15) | (exp << 7) | mant
        return u.astype("<u2").tobytes()
    width = int(kind[2:])
    n = len(streams[0])
    out = np.empty((n, width), np.uint8)
    for j, s in enumerate(streams):
        out[:, j] = np.frombuffer(s, np.uint8, count=n)
    return out.tobytes()


# --------------------------------------------------------------------------- #
# Parallel chunked compress / decompress
# --------------------------------------------------------------------------- #
def _blocks(b: bytes, size: int = _BLOCK):
    return [b[i : i + size] for i in range(0, len(b), size)] or [b""]


def _c_block(s: bytes, level: int) -> bytes:
    c = zstd.ZstdCompressor(level=level).compress(s)
    return b"\x00" + s if len(c) >= len(s) else b"\x01" + c  # store vs zstd flag


def _d_block(b: bytes) -> bytes:
    return b[1:] if b[:1] == b"\x00" else zstd.ZstdDecompressor().decompress(b[1:])


def turbo_compress(raw: bytes, dtype: str, pool: ThreadPoolExecutor, level: int = 1) -> bytes:
    n_elems = len(raw) // bytes_per_element(dtype)
    kind, *streams = _split_streams(raw, dtype)
    # Flatten all stream blocks into one job list; remember per-stream block count.
    jobs, layout = [], []
    for s in streams:
        blks = _blocks(s)
        layout.append((len(s), len(blks)))
        jobs.extend(blks)
    comp = list(pool.map(lambda b: _c_block(b, level), jobs))
    # Serialize: header + per-stream (orig_len, n_blocks) + per-block (len, bytes)
    out = bytearray(MAGIC)
    out += struct.pack("<B", len(kind)) + kind.encode()
    out += struct.pack("<QB", n_elems, len(streams))
    for (slen, nblk) in layout:
        out += struct.pack("<QI", slen, nblk)
    for c in comp:
        out += struct.pack("<I", len(c)) + c
    return bytes(out)


def turbo_decompress(blob: bytes, pool: ThreadPoolExecutor) -> bytes:
    assert blob[:4] == MAGIC
    off = 4
    (klen,) = struct.unpack_from("<B", blob, off); off += 1
    kind = blob[off : off + klen].decode(); off += klen
    n_elems, n_streams = struct.unpack_from("<QB", blob, off); off += 9
    layout = []
    for _ in range(n_streams):
        slen, nblk = struct.unpack_from("<QI", blob, off); off += 12
        layout.append((slen, nblk))
    comp_blocks = []
    for (_slen, nblk) in layout:
        per = []
        for _ in range(nblk):
            (clen,) = struct.unpack_from("<I", blob, off); off += 4
            per.append(blob[off : off + clen]); off += clen
        comp_blocks.append(per)
    streams = []
    for per in comp_blocks:
        parts = list(pool.map(_d_block, per))
        streams.append(b"".join(parts))
    return _join_streams(kind, streams, n_elems)


# --------------------------------------------------------------------------- #
# Head-to-head
# --------------------------------------------------------------------------- #
def _bench_codec(fn_c, fn_d, data, repeats=5):
    fn_d(fn_c(data))  # warm
    t = time.perf_counter()
    for _ in range(repeats):
        blob = fn_c(data)
    ct = (time.perf_counter() - t) / repeats
    t = time.perf_counter()
    for _ in range(repeats):
        out = fn_d(blob)
    dt = (time.perf_counter() - t) / repeats
    ok = bytes(out) == data
    mb = len(data) / 1e6
    return len(data) / len(blob), mb / ct, mb / dt, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=32.0)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args(argv)

    import os
    workers = args.workers or os.cpu_count() or 8

    try:
        import zipnn
        have_zipnn = True
    except Exception:
        have_zipnn = False

    print(f"Turbo prototype - {args.mb} MB/dtype, {workers} workers, level={args.level}")
    print(f"zipnn={'yes' if have_zipnn else 'no'}\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dtype in args.dtypes:
            n_elem = int(args.mb * 1e6) // bytes_per_element(dtype)
            data = make_weights(n_elem, dtype=dtype, seed=0)
            mb = len(data) / 1e6
            print(f"== {dtype}  ({mb:.1f} MB) ==")
            print(f"{'codec':16s}{'ratio':>8}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}")
            print("-" * 60)

            r, c, d, ok = _bench_codec(
                lambda x: turbo_compress(x, dtype, pool, args.level),
                lambda b: turbo_decompress(b, pool),
                data,
            )
            print(f"{'z4ai-turbo':16s}{r:8.3f}{c:12.0f}{d:14.0f}{'yes' if ok else 'NO!':>10}")

            if have_zipnn:
                zdtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, "float32")
                def zc(x):
                    return zipnn.ZipNN(bytearray_dtype=zdtype, input_format="byte").compress(x)
                def zd(b):
                    return zipnn.ZipNN(bytearray_dtype=zdtype, input_format="byte").decompress(b)
                r, c, d, ok = _bench_codec(zc, zd, data)
                print(f"{'zipnn':16s}{r:8.3f}{c:12.0f}{d:14.0f}{'yes' if ok else 'NO!':>10}")
            print()


if __name__ == "__main__":
    main()
