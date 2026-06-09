# SPDX-License-Identifier: Apache-2.0

"""Sparse-aware transform: nonzero bitmap + densified values.

Pruned / sparse model checkpoints contain many *exact-zero* weights (magnitude
pruning, structured sparsity, MoE gating, optimizer masks).  ZipNN's
byte-grouping + Huffman captures zeros only moderately: a zero ``bf16`` value
``0x0000`` still spends bits in every byte plane, and at 256 KiB-chunk
granularity the zero runs are diluted.

z4ai instead separates *where* the nonzeros are from *what* they are:

* a **nonzero bitmap** - one bit per element, ``np.packbits``-ed and Zstd-ed.
  At sparsity ``s`` this costs ~``(1-s)``-dependent entropy, far below a byte.
* the **densified values** - only the nonzero elements, run through the fast
  byte-plane codec (:mod:`z4ai.turbo`), so the surviving weights compress
  exactly as well as a dense tensor of the same size.

The result beats ZipNN by a widening margin as sparsity rises (measured: bf16
50% 2.47x vs 2.20x; 95% 17.3x vs 6.3x).  The frame is self-describing (magic
``b"ZSP1"``), fully lossless, and optionally records NumPy dtype/shape so it
composes with :func:`z4ai.compress_ndarray`.

This module is deliberately standalone (only NumPy + Zstd + :mod:`z4ai.turbo`)
so the core codec can adopt it as one more AUTO best-of candidate without
coupling.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np
import zstandard as zstd

from . import turbo

MAGIC = b"ZSP1"
_VERSION = 1
_FLAG_NUMPY = 1 << 0

# Element widths we can interpret as fixed-width unsigned integers for the
# "is this element exactly zero?" test.  Anything else is not a sparse case we
# handle (the caller keeps the dense frame).
_UINT_DTYPE = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}

# Bitmap is tiny and very compressible; a mid Zstd level is cheap and helps.
_BITMAP_LEVEL = 9


def supported_width(width: int) -> bool:
    """True if ``width`` can be treated as a fixed-width unsigned integer."""
    return width in _UINT_DTYPE


def _uint_view(raw: bytes, width: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=_UINT_DTYPE[width])


def sparsity(raw: bytes, width: int) -> float:
    """Fraction of elements that are exactly zero (all-zero bytes).

    Returns 0.0 for unsupported widths or empty/partial input so callers can
    use it as a cheap gate.
    """
    if not supported_width(width) or not raw:
        return 0.0
    count = len(raw) // width
    if count == 0:
        return 0.0
    u = _uint_view(raw[: count * width], width)
    return float(count - np.count_nonzero(u)) / count


# Sampled-gate tuning.  Below this element count the exact full scan is already
# cheap, so we just do it.  Above it we first estimate sparsity on a strided
# subsample of ~_SPARSE_SAMPLE_ELEMS elements and only fast-path the *confidently
# dense* case (sample sparsity below ``threshold * _SPARSE_SAMPLE_SAFETY``);
# anything near the threshold falls through to the exact scan, so sparse and
# borderline inputs behave identically.  The 0.5 safety margin is far wider than
# the subsample's statistical error, so a real >=10%-sparse buffer is never
# fast-skipped (its sample reads well above 5%).
_SPARSE_SAMPLE_MIN = 1 << 20      # 1M elements
_SPARSE_SAMPLE_ELEMS = 1 << 20    # total subsample size
_SPARSE_SAMPLE_WINDOWS = 64       # contiguous windows spread across the buffer
_SPARSE_SAMPLE_SAFETY = 0.5


def should_use(raw: bytes, width: int, threshold: float = 0.10) -> bool:
    """Cheap gate: is it worth building a sparse candidate for this buffer?

    Conservative by design - the caller still keeps the smaller of the sparse
    and dense frames, so a false positive only costs one extra compression
    pass, never ratio.

    For large buffers a strided subsample is checked first so dense weights (the
    common case) skip the full-buffer ``count_nonzero`` scan, which is otherwise
    paid on *every* compress and scales with the whole model size.  The exact
    scan still runs for anything near the threshold, so the gate's decision is
    unchanged for sparse/borderline inputs.
    """
    if not supported_width(width) or not raw:
        return False
    count = len(raw) // width
    if count == 0:
        return False
    if count > _SPARSE_SAMPLE_MIN:
        u = _uint_view(raw[: count * width], width)
        # Sample several *contiguous* windows spread evenly across the buffer:
        # contiguous count_nonzero is SIMD-fast (a strided view is not), and
        # spreading the windows keeps the estimate robust to clustered zeros
        # (block/structured pruning) that a single contiguous slice would miss.
        win = _SPARSE_SAMPLE_ELEMS // _SPARSE_SAMPLE_WINDOWS
        starts = np.linspace(0, count - win, _SPARSE_SAMPLE_WINDOWS).astype(np.intp)
        nz = sum(int(np.count_nonzero(u[s : s + win])) for s in starts)
        sampled = _SPARSE_SAMPLE_WINDOWS * win
        samp_sparsity = float(sampled - nz) / sampled
        if samp_sparsity < threshold * _SPARSE_SAMPLE_SAFETY:
            return False  # confidently dense — skip the full scan
    return sparsity(raw, width) >= threshold


def peek_is_sparse(blob: bytes) -> bool:
    """True if ``blob`` is a z4ai sparse frame (starts with :data:`MAGIC`)."""
    return len(blob) >= 4 and blob[:4] == MAGIC


def compress(
    raw: bytes,
    width: int,
    *,
    level: int = 1,
    is_numpy: bool = False,
    np_dtype: str = "",
    shape: Tuple[int, ...] = (),
) -> bytes:
    """Encode ``raw`` (a fixed-width buffer) as a sparse ``b"ZSP1"`` frame.

    Args:
        raw: little-endian element bytes.
        width: element width in bytes (must be 1/2/4/8).
        level: Zstd level for the densified values.
        is_numpy/np_dtype/shape: optional NumPy metadata to round-trip.

    Returns:
        A self-describing sparse frame.

    Raises:
        ValueError: if ``width`` is unsupported.
    """
    if not supported_width(width):
        raise ValueError(f"sparse codec does not support width {width}")

    raw = bytes(raw)
    count = len(raw) // width
    body = raw[: count * width]
    tail = raw[count * width :]  # bytes not forming a whole element, kept raw

    if count:
        u = _uint_view(body, width)
        mask = u != 0
        nnz = int(np.count_nonzero(mask))
        bitmap_packed = np.packbits(mask).tobytes()
        # ``np.compress`` is the boolean-select primitive and is ~3x faster than
        # the equivalent fancy index ``u[mask]`` (measured 11.5 ms vs 31.6 ms on
        # a 4M-element half-sparse uint16 buffer): it gathers in one C pass
        # instead of materialising and walking an intermediate index array.
        dense = np.compress(mask, u).tobytes()
    else:
        nnz = 0
        bitmap_packed = b""
        dense = b""

    bitmap_comp = (
        zstd.ZstdCompressor(level=_BITMAP_LEVEL).compress(bitmap_packed)
        if bitmap_packed
        else b""
    )
    values = turbo.compress(dense, width=width, level=level) if dense else b""

    dtype_bytes = np_dtype.encode("ascii") if is_numpy else b""
    if len(dtype_bytes) > 255:
        raise ValueError("dtype string too long")
    shape = tuple(int(d) for d in shape) if is_numpy else ()
    if len(shape) > 255:
        raise ValueError("too many dimensions")

    out = bytearray()
    out += MAGIC
    out += struct.pack("<BBBx", _VERSION, width, _FLAG_NUMPY if is_numpy else 0)
    out += struct.pack("<QQI", count, nnz, len(tail))
    out += struct.pack("<BB", len(dtype_bytes), len(shape))
    out += dtype_bytes
    for dim in shape:
        out += struct.pack("<Q", dim)
    out += tail
    out += struct.pack("<Q", len(bitmap_comp))
    out += bitmap_comp
    out += values  # remainder
    return bytes(out)


def decompress(blob: bytes) -> bytes:
    """Inverse of :func:`compress`; returns the byte-exact original buffer."""
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai sparse frame (bad magic)")
    off = 4
    version, width, flags = struct.unpack_from("<BBBx", mv, off)
    off += 4
    if version != _VERSION:
        raise ValueError(f"unsupported sparse frame version {version}")
    count, nnz, tail_len = struct.unpack_from("<QQI", mv, off)
    off += 20
    dtype_len, ndim = struct.unpack_from("<BB", mv, off)
    off += 2
    off += dtype_len  # NumPy metadata is consumed by the higher-level helper
    off += 8 * ndim
    tail = bytes(mv[off : off + tail_len])
    off += tail_len
    (bitmap_len,) = struct.unpack_from("<Q", mv, off)
    off += 8
    bitmap_comp = bytes(mv[off : off + bitmap_len])
    off += bitmap_len
    values = bytes(mv[off:])

    if count == 0:
        return tail

    bitmap_packed = (
        zstd.ZstdDecompressor().decompress(bitmap_comp) if bitmap_comp else b""
    )
    mask = np.unpackbits(
        np.frombuffer(bitmap_packed, dtype=np.uint8), count=count
    ).astype(bool)

    dtype = _UINT_DTYPE[width]
    u = np.zeros(count, dtype=dtype)
    if nnz:
        dense = turbo.decompress(values)
        # Scatter via integer indices rather than the boolean mask: with
        # ``idx = flatnonzero(mask)`` (fast on a bool array), ``u[idx] = dense``
        # is ~4x faster than ``u[mask] = dense`` (measured 7.6 ms vs 32 ms),
        # because masked assignment re-walks the whole mask while indexed
        # assignment scatters directly.
        idx = np.flatnonzero(mask)
        u[idx] = np.frombuffer(dense, dtype=dtype)
    return u.tobytes() + tail


def read_numpy_metadata(blob: bytes) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """Return ``(dtype_str, shape)`` if the sparse frame carries NumPy metadata.

    Used by :func:`z4ai.decompress_ndarray` to restore an array when the sparse
    frame won the AUTO best-of.  Returns ``None`` for non-array frames.
    """
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai sparse frame (bad magic)")
    _version, _width, flags = struct.unpack_from("<BBBx", mv, 4)
    if not (flags & _FLAG_NUMPY):
        return None
    off = 4 + 4 + 20
    dtype_len, ndim = struct.unpack_from("<BB", mv, off)
    off += 2
    dtype_str = bytes(mv[off : off + dtype_len]).decode("ascii")
    off += dtype_len
    shape = tuple(struct.unpack_from("<Q", mv, off + 8 * i)[0] for i in range(ndim))
    return dtype_str, shape


__all__ = [
    "MAGIC",
    "supported_width",
    "sparsity",
    "should_use",
    "peek_is_sparse",
    "compress",
    "decompress",
    "read_numpy_metadata",
]
