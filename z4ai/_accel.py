# SPDX-License-Identifier: Apache-2.0

"""Native-accelerated byte-plane shuffle with a transparent NumPy fallback.

``split_planes`` / ``join_planes`` here are drop-in replacements for the ones in
:mod:`z4ai.transforms`: identical semantics, but they use the compiled
``_native_shuffle`` extension when it is importable (memory-bandwidth SIMD
transpose) and fall back to NumPy otherwise. Import never fails for lack of a
compiler - that is the whole point.

``HAVE_NATIVE`` tells callers (and the benchmark) which path is live.
"""
from __future__ import annotations

from typing import List

import numpy as np

try:  # pragma: no cover - presence depends on whether the ext was built
    from . import _native_shuffle as _native

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    _native = None
    HAVE_NATIVE = False


def split_planes(buf: bytes, width: int) -> List[bytes]:
    """Split ``buf`` into ``width`` byte planes (see transforms.split_planes)."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    n = len(buf) // width
    if width == 1:
        return [bytes(buf[: n])]
    if n == 0:
        return [b"" for _ in range(width)]

    if HAVE_NATIVE:
        concat = _native.split(buf, width)  # one buffer, plane j at [j*n:(j+1)*n]
        mv = memoryview(concat)
        # Zero-copy views; the backend consumes them via the buffer protocol.
        return [mv[j * n : (j + 1) * n] for j in range(width)]

    arr = np.frombuffer(buf, dtype=np.uint8, count=n * width).reshape(n, width)
    return [np.ascontiguousarray(arr[:, j]).tobytes() for j in range(width)]


def bf16_split(raw):
    """Split a bf16 little-endian buffer into (sign_packed, exponent, mantissa).

    Byte-exact with :func:`z4ai.bitfield.split_fields`'s NumPy path: ``sign`` is
    bit-packed big-bit-order (``np.packbits`` default), ``exponent`` is one byte
    per element, ``mantissa`` is one byte per element.

    Uses the **two-pass C** ``_native.bf16_split`` when the extension is built,
    else the vectorized NumPy fallback below.  An earlier note here claimed the
    native split was 3.3-4.2x SLOWER (1288 MB/s); that measured the original
    *single-pass scalar* routine, whose sign bit-pack (`|=` into shared bytes)
    serialized.  The routine is now split into two vectorizable passes (exp/mant
    deinterleave + independent sign bytes); re-measured on the rebuilt binary it
    is **2.08x FASTER than NumPy (9839 vs 4734 MB/s, 64MB)** and byte-exact.  See
    COORDINATION.md iter7 for the repro; rebuild with ``python -m
    z4ai.build_native`` if you measure otherwise.
    """
    mv = memoryview(raw)
    n = len(mv) // 2
    if HAVE_NATIVE:
        return _native.bf16_split(mv[: n * 2])
    u = np.frombuffer(mv, dtype="<u2", count=n)
    sign = (u >> 15).astype(np.uint8)
    exp = ((u >> 7) & 0xFF).astype(np.uint8)
    mant = (u & 0x7F).astype(np.uint8)
    return np.packbits(sign).tobytes(), exp.tobytes(), mant.tobytes()


def bf16_join(sign_packed, exponent, mantissa, n: int) -> bytes:
    """Inverse of :func:`bf16_split`: rebuild the bf16 little-endian buffer.

    Uses the single-pass C routine when available, else builds the two byte
    planes with NumPy and interleaves them with :func:`join_planes`.
    """
    if HAVE_NATIVE:
        return _native.bf16_join(sign_packed, exponent, mantissa, n)
    exp = np.frombuffer(exponent, dtype=np.uint8, count=n)
    mant = np.frombuffer(mantissa, dtype=np.uint8, count=n)
    sign = np.unpackbits(np.frombuffer(sign_packed, dtype=np.uint8), count=n)
    low = (mant & 0x7F) | ((exp & 1) << 7)
    high = (exp >> 1) | (sign << 7)
    return join_planes([low, high], 2)


def join_planes(planes, width: int, tail: bytes = b"") -> bytes:
    """Inverse of :func:`split_planes` (see transforms.join_planes)."""
    if len(planes) != width:
        raise ValueError(f"expected {width} planes, got {len(planes)}")
    if width == 1:
        return bytes(planes[0]) + tail
    n = len(planes[0])
    if n == 0:
        return b"" + tail

    if HAVE_NATIVE:
        return _native.join(planes, width, n) + tail

    out = np.empty((n, width), dtype=np.uint8)
    for j, plane in enumerate(planes):
        if len(plane) != n:
            raise ValueError("planes have inconsistent lengths")
        out[:, j] = np.frombuffer(plane, dtype=np.uint8, count=n)
    return out.tobytes() + tail
