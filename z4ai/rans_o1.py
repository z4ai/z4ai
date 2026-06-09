# Copyright 2026 The z4ai Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Order-1 (context = previous byte) rANS entropy coder.

This is z4ai's ratio edge over ZipNN on *dense* weights.  ZipNN (and z4ai's own
order-0 path) code the exponent plane with an order-0 model, which is provably
optimal only for an i.i.d. source.  Real trained-model exponent planes are *not*
i.i.d.: measured on cached models, ``distilgpt2``'s bf16 exponent has order-0
entropy H0 = 3.10 bits but conditional (order-1) entropy H1 = 2.36 bits — a 6.4%
smaller stream than any order-0 / Zstd-FSE coder, and smaller than Zstd's
whole-plane long-distance-matching pass too.  Coding the exponent with an
order-1 model captures that conditional structure.

The model is "an array of order-0 tables", one per previous byte (Giesen's rANS
notes; jkbonfield's rANS_static).  The native kernel (``_native/rans.c``) codes
NSEG contiguous segments, each with its own rANS state and running context, so
the segments stay independent (instruction-level parallelism) while each keeps
an exact order-1 context.  Here we additionally split a large plane into chunks
that share one model and code in parallel across a thread pool (ctypes releases
the GIL), turning the ~270 MB/s single-stream kernel into multi-GB/s.

It is always used as a *best-of* candidate (see :mod:`z4ai.backend`): the codec
keeps it only when it is smaller than the Zstd result, so it never regresses a
plane (e.g. a duplicate-heavy plane where Zstd-LDM wins stays on Zstd).

References: NeuZip (arXiv 2410.20650), DFloat11 (2504.11651), Duda's ANS paper
(1311.2540), Giesen "rANS notes"/"interleaved entropy coders" (1402.3392), and
the public-domain ryg_rans (github.com/rygorous/ryg_rans).
"""

from __future__ import annotations

import ctypes
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Optional

import numpy as np

import zstandard as zstd

from . import rans as _rans  # reuse its lazy native-build + library handle

_pool = None
_pool_lock = Lock()


def _thread_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(
                    max_workers=os.cpu_count() or 1, thread_name_prefix="z4ai-o1"
                )
    return _pool


O1_PROB_BITS = 12
O1_SCALE = 1 << O1_PROB_BITS  # 4096 — must match O1_SCALE in _native/rans.c
_NSEG = 4  # contiguous segments per chunk (ILP within a chunk)

# Plane payload framing.  All little-endian.
_MAGIC = b"R1"
_HDR = struct.Struct("<2sBxIIH")  # magic, nseg, pad, n_total, freq_blob_len, n_chunks
_U32 = struct.Struct("<I")

# Below this a single chunk already codes in well under a few ms, so parallel
# fan-out costs more than it saves: code it as one chunk.
_MIN_CHUNK_BYTES = 1 << 20  # 1 MiB
_MAX_CHUNKS = 64

_bound_lib = None


def _lib():
    """Return the bound native library, or ``None`` if unavailable.

    Reuses :func:`z4ai.rans._load` to lazily build the shared object, then binds
    the order-1 entry points on top of it.
    """
    global _bound_lib
    if _bound_lib is not None:
        return _bound_lib
    lib = _rans._load()
    if lib is None:
        return None
    try:
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        lib.z4ai_rans_o1_encode.restype = ctypes.c_size_t
        lib.z4ai_rans_o1_encode.argtypes = [
            u8,
            ctypes.c_size_t,
            ctypes.c_int,
            u16,
            u8,
            ctypes.c_size_t,
        ]
        lib.z4ai_rans_o1_decode.restype = None
        lib.z4ai_rans_o1_decode.argtypes = [
            u8,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            u16,
            u8,
        ]
    except AttributeError:  # pragma: no cover - stale prebuilt lib without order-1
        return None
    # Native order-1 joint-histogram builder (optional; absent in an older
    # prebuilt lib — guarded so a stale .so still loads and _build_model falls
    # back to NumPy).  Building the 256x256 model in C kills the ~85%-of-encode
    # cost a NumPy bincount-over-65536-bins otherwise paid.
    u32 = ctypes.POINTER(ctypes.c_uint32)
    try:
        lib.z4ai_rans_o1_hist.restype = None
        lib.z4ai_rans_o1_hist.argtypes = [
            u8,
            ctypes.c_size_t,
            u32,
            ctypes.c_size_t,
            u32,
        ]
    except AttributeError:  # pragma: no cover - stale prebuilt lib
        pass
    _bound_lib = lib
    return lib


def available() -> bool:
    """True if the native order-1 rANS kernel is built and loadable."""
    try:
        return _lib() is not None
    except OSError:
        return False


def _chunk_count(n: int) -> int:
    if n < 2 * _MIN_CHUNK_BYTES:
        return 1
    workers = os.cpu_count() or 1
    return max(1, min(_MAX_CHUNKS, workers, n // _MIN_CHUNK_BYTES))


def _bounds(n: int, k: int):
    return [(i * n // k, (i + 1) * n // k) for i in range(k)]


def _build_model(src: np.ndarray, n_chunks: int) -> np.ndarray:
    """Build the 256x256 row-normalised order-1 frequency table for ``src``.

    The context of byte ``i`` is byte ``i-1``, except at every chunk *and*
    segment boundary, where it is reset to 0 — exactly mirroring the native
    coder, so the model the decoder is handed matches what the encoder used.
    """
    n = src.size
    # Context-reset positions: segment starts of every chunk (and position 0),
    # where the running context is forced to 0.  Sorted + unique so the native
    # builder's single-pass reset fix-up is correct.
    resets = {0}
    for lo, hi in _bounds(n, n_chunks):
        cn = hi - lo
        for s in range(_NSEG):
            start = lo + cn * s // _NSEG
            if start < n:
                resets.add(start)
    reset_arr = np.array(sorted(resets), dtype=np.uint32)

    lib = _lib()
    if lib is not None and hasattr(lib, "z4ai_rans_o1_hist"):
        # Native C scatter (GIL released, 4 accumulator tables) — replaces a
        # NumPy bincount that widened the whole plane to int64 and scattered
        # into 65536 bins (the dominant cost of order-1 compress).
        joint_flat = np.zeros(256 * 256, dtype=np.uint32)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u32 = ctypes.POINTER(ctypes.c_uint32)
        lib.z4ai_rans_o1_hist(
            src.ctypes.data_as(u8),
            ctypes.c_size_t(n),
            reset_arr.ctypes.data_as(u32),
            ctypes.c_size_t(reset_arr.size),
            joint_flat.ctypes.data_as(u32),
        )
        joint = joint_flat.reshape(256, 256)
    else:
        # Pure-NumPy fallback (no native lib): build the joint histogram with a
        # uint16 index (ctx*256+cur fits in 16 bits) to avoid the int64 blow-up.
        idx = np.empty(n, dtype=np.uint16)
        idx[0] = src[0]
        if n > 1:
            idx[1:] = (src[:-1].astype(np.uint16) << 8) | src[1:]
        for r in reset_arr:
            if 0 < r < n:
                idx[r] = src[r]  # context 0 -> idx = src[r]
        joint = np.bincount(idx, minlength=256 * 256).reshape(256, 256)

    freq = np.zeros((256, 256), dtype=np.uint16)
    for c in range(256):
        row = joint[c]
        if row.sum() == 0:
            continue
        freq[c] = (
            _rans._normalize_to(row, O1_SCALE)
            if hasattr(_rans, "_normalize_to")
            else _normalize_row(row)
        )
    return freq


def _normalize_row(hist: np.ndarray) -> np.ndarray:
    """Scale a 256-bin row to integer freqs summing to O1_SCALE (each used >=1)."""
    total = int(hist.sum())
    f = np.zeros(256, dtype=np.int64)
    if total == 0:
        return f.astype(np.uint16)
    nz = hist > 0
    f = np.floor(hist.astype(np.float64) * (O1_SCALE / total)).astype(np.int64)
    f[nz & (f == 0)] = 1
    diff = O1_SCALE - int(f.sum())
    if diff > 0:
        f[int(np.argmax(f))] += diff
    elif diff < 0:
        need = -diff
        for i in np.argsort(-f):
            if need == 0:
                break
            room = int(f[i]) - 1
            if room > 0:
                take = min(room, need)
                f[i] -= take
                need -= take
    return f.astype(np.uint16)


def _serialize_model(freq: np.ndarray) -> bytes:
    """Compactly serialize the sparse 256x256 freq table, then Zstd it.

    Only non-empty contexts and their non-zero symbols are stored, so the blob is
    tiny (~1 KB for the ~40-symbol exponent alphabet) — negligible next to a
    multi-MB plane.
    """
    parts = []
    for c in range(256):
        nz = np.nonzero(freq[c])[0]
        if nz.size == 0:
            continue
        parts.append(bytes([c, len(nz) & 0xFF, (len(nz) >> 8) & 0xFF]))
        for s in nz:
            parts.append(bytes([int(s)]) + int(freq[c][s]).to_bytes(2, "little"))
    raw = b"".join(parts)
    return zstd.ZstdCompressor(level=19).compress(raw)


def _deserialize_model(blob: bytes) -> np.ndarray:
    raw = zstd.ZstdDecompressor().decompress(blob)
    freq = np.zeros((256, 256), dtype=np.uint16)
    mv = memoryview(raw)
    off = 0
    while off < len(raw):
        c = mv[off]
        nsym = mv[off + 1] | (mv[off + 2] << 8)
        off += 3
        for _ in range(nsym):
            s = mv[off]
            f = mv[off + 1] | (mv[off + 2] << 8)
            freq[c, s] = f
            off += 3
    return freq


def _encode_chunk(lib, src: np.ndarray, freq: np.ndarray) -> bytes:
    n = src.size
    cap = n * 2 + 4096 + _NSEG * 4
    out = np.empty(cap, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    length = lib.z4ai_rans_o1_encode(
        src.ctypes.data_as(u8),
        ctypes.c_size_t(n),
        ctypes.c_int(_NSEG),
        freq.ctypes.data_as(u16),
        out.ctypes.data_as(u8),
        ctypes.c_size_t(cap),
    )
    if length == 0:
        raise RuntimeError("order-1 rANS encode overflow")
    return out[:length].tobytes()


def _decode_chunk(lib, payload: bytes, freq: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint8)
    p = np.frombuffer(payload, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    lib.z4ai_rans_o1_decode(
        p.ctypes.data_as(u8),
        ctypes.c_size_t(len(payload)),
        ctypes.c_size_t(n),
        ctypes.c_int(_NSEG),
        freq.ctypes.data_as(u16),
        out.ctypes.data_as(u8),
    )
    return out


def compress(data: bytes) -> bytes:
    """Order-1 rANS-compress a byte buffer into a self-describing frame.

    Raises :class:`RuntimeError` if the native coder is unavailable.
    """
    lib = _lib()
    if lib is None:
        raise RuntimeError("native order-1 rANS unavailable (no C compiler?)")
    src = np.frombuffer(data, dtype=np.uint8)
    n = src.size
    if n == 0:
        return _HDR.pack(_MAGIC, _NSEG, 0, 0, 0)
    k = _chunk_count(n)
    freq = _build_model(src, k)
    blob = _serialize_model(freq)
    bounds = _bounds(n, k)
    if k == 1:
        payloads = [_encode_chunk(lib, src, freq)]
    else:
        payloads = list(
            _thread_pool().map(
                lambda b: _encode_chunk(lib, src[b[0] : b[1]], freq), bounds
            )
        )
    head = _HDR.pack(_MAGIC, _NSEG, n, len(blob), k)
    parts = [head, blob]
    for b, pay in zip(bounds, payloads):
        parts.append(_U32.pack(len(pay)))
    parts.extend(payloads)
    return b"".join(parts)


def decompress(blob: bytes) -> bytes:
    """Invert :func:`compress`."""
    lib = _lib()
    magic, nseg, n, blob_len, k = _HDR.unpack_from(blob, 0)
    if magic != _MAGIC:
        raise ValueError("not an order-1 rANS frame")
    if n == 0:
        return b""
    if lib is None:
        raise RuntimeError("native order-1 rANS unavailable for decode")
    off = _HDR.size
    freq = _deserialize_model(bytes(blob[off : off + blob_len]))
    off += blob_len
    clens = []
    for _ in range(k):
        (cl,) = _U32.unpack_from(blob, off)
        off += _U32.size
        clens.append(cl)
    payloads = []
    for cl in clens:
        payloads.append(bytes(blob[off : off + cl]))
        off += cl
    bounds = _bounds(n, k)
    chunk_ns = [hi - lo for lo, hi in bounds]
    if k == 1:
        parts = [_decode_chunk(lib, payloads[0], freq, chunk_ns[0])]
    else:
        idx = list(range(k))
        parts = list(
            _thread_pool().map(
                lambda i: _decode_chunk(lib, payloads[i], freq, chunk_ns[i]), idx
            )
        )
    return np.concatenate(parts).tobytes()
