# SPDX-License-Identifier: Apache-2.0

"""Chunked, multi-threaded codec - the high-throughput path.

ZipNN reaches several GB/s by fusing a native byte-split with a multi-threaded
Zstd pass over chunks.  Profiling (``FINDINGS-speed.md``) showed our throughput
ceiling is single-threaded Zstd, not the NumPy split - and crucially that
**both NumPy and Zstandard release the GIL**.  So we can parallelize a pure
Python pipeline: cut the buffer into element-aligned chunks and run, on each
core, an independent ``split -> Zstd`` pass.

This module keeps a deliberately *lean* per-chunk path (direct byte-plane
transpose + a single Zstd level-1 pass per plane, no per-stream probing) so the
fixed overhead per chunk stays tiny and the thread pool stays saturated.  It is
fully self-contained and lossless by construction.

.. warning::
   **Chunking forfeits cross-chunk redundancy.** Splitting the buffer caps
   Zstd's match horizon at one chunk, so on *structured / duplicated* weights
   (tied embeddings, repeated layers, EMA/optimizer copies) this path gives up
   the order-of-magnitude long-distance-matching win that the default
   whole-plane :func:`z4ai.compress` keeps - the very thing that beats ZipNN by
   tens of x on real checkpoints.  Use this chunked path for *throughput* on
   i.i.d.-like weights where the mantissa is the bulk and there is little
   long-range structure; use the default codec when ratio is what matters.

Frame layout (all integers little-endian)::

    magic       8 bytes   b"Z4AIMF01"
    flags       u8        reserved (0)
    width       u8        element width used for the transpose
    n_chunks    u32
    per chunk:  u32 elems, u8 tail_len, tail bytes, then
                width * (u8 method, u32 comp_len, comp_bytes)

``method`` is 0 = stored raw, 1 = Zstd.  Plane order is byte-plane order, i.e.
plane ``j`` holds byte ``j`` of every element.
"""

from __future__ import annotations

import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import zstandard as zstd

# SIMD byte-plane transpose accelerator (shared with z4ai.transforms).  It de-
# interleaves / re-interleaves the byte planes at ~16-48 GB/s - an order of
# magnitude faster than NumPy's strided ``arr[:, j]`` scatter/gather, which was
# the dominant non-Zstd cost on this throughput path (measured: bf16 decode
# 5.8 -> 10.8 GB/s, fp32 6.8 -> 13.6 GB/s).  Import is always safe: ``_accel``
# transparently falls back to NumPy when the C extension is absent, so the
# chunked path stays correct and self-contained everywhere, just slower.
try:
    from . import _accel as _shuffle
except Exception:  # noqa: BLE001 - never let the accel import break the codec
    _shuffle = None

# Fully fused, multithreaded native codec (links libzstd, runs the whole
# split -> zstd -> interleave loop in C with the GIL released).  It produces and
# consumes the IDENTICAL Z4AIMF01 frame, so it is byte-interoperable with the
# pure-Python path below, which stays the reference + fallback.  This is what
# closes the throughput gap to zipnn's compiled C: pure-Python decode is capped
# ~7 GB/s by per-chunk interpreter overhead + GIL serialization, whereas the
# native kernel reaches 22-34 GB/s (0.64-0.77x of zipnn, vs 0.33x before).  When
# the extension is not built, every call transparently uses the Python path.
try:
    from . import _native_codec as _native
except Exception:  # noqa: BLE001 - optional accelerator; never a hard dependency
    _native = None

MAGIC = b"Z4AIMF01"
DEFAULT_CHUNK_SIZE = 2 << 20  # 2 MiB - small enough to spread across cores
DEFAULT_LEVEL = 1  # see FINDINGS-speed.md: best ratio AND fastest

_METHOD_STORE = 0
_METHOD_ZSTD = 1

# Zstandard contexts are not thread-safe, so give each worker thread its own.
_tls = threading.local()
# One reusable pool avoids paying thread-creation cost on every call.
_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()


def _width_for_dtype(dtype: Optional[str]) -> int:
    return {
        None: 1,
        "uint8": 1,
        "u8": 1,
        "int8": 1,
        "i8": 1,
        "fp8_e4m3": 1,
        "fp8_e5m2": 1,
        "fp16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "float16": 2,
        "int16": 2,
        "uint16": 2,
        "fp32": 4,
        "float32": 4,
        "int32": 4,
        "uint32": 4,
        "fp64": 8,
        "float64": 8,
        "int64": 8,
        "uint64": 8,
    }.get(dtype.lower() if isinstance(dtype, str) else dtype, 1)


def _pool(workers: int) -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL._max_workers < workers:  # type: ignore[attr-defined]
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            _POOL = ThreadPoolExecutor(max_workers=workers)
        return _POOL


def _cctx(level: int) -> "zstd.ZstdCompressor":
    ctx = getattr(_tls, "cctx", None)
    if ctx is None or _tls.level != level:
        _tls.cctx = zstd.ZstdCompressor(level=level)
        _tls.level = level
        ctx = _tls.cctx
    return ctx


def _dctx() -> "zstd.ZstdDecompressor":
    ctx = getattr(_tls, "dctx", None)
    if ctx is None:
        _tls.dctx = zstd.ZstdDecompressor()
        ctx = _tls.dctx
    return ctx


def _resolve_workers(threads: int) -> int:
    if threads and threads > 0:
        return threads
    return max(1, os.cpu_count() or 1)


def _compress_chunk(buf: bytes, width: int, level: int) -> bytes:
    n_elem = len(buf) // width
    aligned = n_elem * width
    tail = buf[aligned:]
    cctx = _cctx(level)

    if width == 1:
        planes = [buf[:aligned]]
    elif _shuffle is not None:
        # SIMD de-interleave (memory-bandwidth); planes come back as zero-copy
        # views the Zstd compressor consumes directly via the buffer protocol.
        planes = _shuffle.split_planes(buf[:aligned], width)
    else:
        a = np.frombuffer(buf, dtype=np.uint8, count=aligned).reshape(n_elem, width)
        transposed = np.ascontiguousarray(a.T)  # single copy, plane-major
        planes = [transposed[j].tobytes() for j in range(width)]

    out = bytearray()
    out += struct.pack("<IB", n_elem, len(tail))
    out += tail
    for plane in planes:
        comp = cctx.compress(plane)
        if len(comp) < len(plane):
            out += struct.pack("<BI", _METHOD_ZSTD, len(comp))
            out += comp
        else:  # incompressible (e.g. mantissa) - store raw, skip a decode pass
            out += struct.pack("<BI", _METHOD_STORE, len(plane))
            out += plane
    return bytes(out)


def _decompress_chunk(mv: memoryview, off: int, width: int):
    n_elem, tail_len = struct.unpack_from("<IB", mv, off)
    off += 5
    tail = bytes(mv[off : off + tail_len])
    off += tail_len
    planes: List = []
    for _ in range(width):
        method, comp_len = struct.unpack_from("<BI", mv, off)
        off += 5
        seg = mv[off : off + comp_len]  # zero-copy view into the frame
        off += comp_len
        if method == _METHOD_ZSTD:
            # Zstd reads the compressed bytes via the buffer protocol — no need
            # to materialise a `bytes` copy of the segment first.
            planes.append(_dctx().decompress(seg, max_output_size=n_elem))
        else:
            # Stored raw (the incompressible bulk — mantissa planes are most of
            # the data): hand the view straight to the interleave step.  These
            # are the largest planes, so skipping their copy is the bulk of the
            # win (measured fp32 decode 12.8 -> 21.2 GB/s, 1.66x).
            planes.append(seg)
    if width == 1:
        return bytes(planes[0]) + tail, off
    if _shuffle is not None:
        # SIMD re-interleave (memory-bandwidth) - replaces the strided
        # ``out[:, j] = ...`` scatter that dominated decode wall-time.  It
        # consumes each plane via the buffer protocol, so the stored memoryviews
        # interleave with no extra copy.
        return _shuffle.join_planes(planes, width, tail), off
    out = np.empty((n_elem, width), dtype=np.uint8)
    for j, plane in enumerate(planes):
        out[:, j] = np.frombuffer(plane, dtype=np.uint8, count=n_elem)
    return out.tobytes() + tail, off


def _chunk_starts(total: int, width: int, chunk_size: int) -> List[int]:
    if total == 0:
        return []
    step = max(width, (chunk_size // width) * width)
    return list(range(0, total, step))


def compress(
    data,
    *,
    dtype: Optional[str] = None,
    level: Optional[int] = None,
    threads: int = 0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bytes:
    """Compress ``data`` with the chunked, multi-threaded pipeline.

    ``threads`` controls concurrent chunk workers (0 = all cores); other args
    mirror :func:`z4ai.compress`.  Returns one ``Z4AIMF01`` frame.
    """
    # View the input through a memoryview so per-chunk slices are zero-copy
    # (each ``buf[start:start+step]`` would otherwise duplicate that chunk).
    # Only buffer-protocol objects qualify; anything else is coerced once.
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = memoryview(data)
    else:
        buf = memoryview(bytes(data))
    width = _width_for_dtype(dtype)
    lvl = DEFAULT_LEVEL if level is None else level

    # Fast path: the fused native codec emits the identical frame, so prefer it
    # whenever it is built.  Falls through to the pure-Python pipeline otherwise.
    if _native is not None:
        return _native.compress(buf, width, lvl, _resolve_workers(threads), chunk_size)

    starts = _chunk_starts(len(buf), width, chunk_size)
    step = max(width, (chunk_size // width) * width)
    workers = _resolve_workers(threads)

    def _do(start: int) -> bytes:
        return _compress_chunk(buf[start : start + step], width, lvl)

    if not starts:
        frames: List[bytes] = []
    elif workers <= 1 or len(starts) <= 1:
        frames = [_do(s) for s in starts]
    else:
        frames = list(_pool(workers).map(_do, starts))

    out = bytearray()
    out += MAGIC
    out += struct.pack("<BBI", 0, width, len(frames))
    out += b"".join(frames)
    return bytes(out)


def is_chunked(blob: bytes) -> bool:
    """True if ``blob`` is a chunked ``Z4AIMF01`` frame."""
    return len(blob) >= 8 and blob[:8] == MAGIC


def decompress(blob: bytes, *, threads: int = 0) -> bytes:
    """Inverse of :func:`compress`; reconstructs the exact original buffer."""
    if not is_chunked(blob):
        raise ValueError("not a chunked z4ai frame (bad magic)")

    # Fast path: the fused native decoder reads the identical frame format.
    if _native is not None:
        return _native.decompress(blob, _resolve_workers(threads))

    mv = memoryview(blob)
    _flags, width, n_chunks = struct.unpack_from("<BBI", mv, 8)
    off = 14

    # Decode is split in two: a cheap serial scan to find each chunk's byte
    # span (the frame is variable-length), then a parallel decode of the spans.
    spans = []
    for _ in range(n_chunks):
        start = off
        n_elem, tail_len = struct.unpack_from("<IB", mv, off)
        off += 5 + tail_len
        for _p in range(width):
            _method, comp_len = struct.unpack_from("<BI", mv, off)
            off += 5 + comp_len
        spans.append((start, n_elem))

    def _do(span):
        start, _n = span
        part, _ = _decompress_chunk(mv, start, width)
        return part

    workers = _resolve_workers(threads)
    if workers <= 1 or len(spans) <= 1:
        parts = [_do(s) for s in spans]
    else:
        parts = list(_pool(workers).map(_do, spans))
    return b"".join(parts)
