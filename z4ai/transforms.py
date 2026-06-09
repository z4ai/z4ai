# SPDX-License-Identifier: Apache-2.0

"""Byte-plane (a.k.a. byte-grouping) transforms.

Floating-point model weights have a peculiar statistical structure: the sign
and exponent bits are highly repetitive (most weights share a handful of
exponents), while the mantissa bits are close to uniform noise.  A general
purpose compressor sees the *interleaved* byte stream and cannot exploit this,
because every compressible exponent byte is sandwiched between two incompressible
mantissa bytes.

The fix is a *byte transpose*: regroup the buffer so that all the bytes at
position 0 of every element are contiguous, then all the bytes at position 1,
and so on.  Each resulting plane is statistically homogeneous, so the
entropy backend can crush the exponent-heavy planes and cheaply skip the
mantissa-heavy ones.

This module performs the transpose with zero-copy NumPy reshapes; it is the
heart of why z4ai beats a vanilla Zstd pass on model weights.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Optional SIMD C accelerator. When the `_native_shuffle` extension has been
# built (`python -m z4ai.build_native`), it does the transpose at memory
# bandwidth (~10-16x faster than the NumPy strided copy below). Import is always
# safe: `_accel` itself falls back to NumPy when the extension is absent, so the
# pure-NumPy implementations in this module stay the reference fallback path.
try:
    from . import _accel
    _HAVE_ACCEL = _accel.HAVE_NATIVE
except Exception:  # noqa: BLE001
    _accel = None
    _HAVE_ACCEL = False


def split_planes(buf: bytes, width: int) -> List[bytes]:
    """Split ``buf`` into ``width`` byte planes.

    Element ``i`` of the original buffer occupies bytes
    ``buf[i*width : (i+1)*width]``.  Plane ``j`` collects byte ``j`` of every
    element, i.e. ``plane_j[i] == buf[i*width + j]``.

    Any trailing bytes that do not form a complete element are returned by the
    caller separately (see :func:`tail_len`); this function only consumes the
    aligned prefix.

    Args:
        buf: raw little-endian element bytes.
        width: bytes per element (1, 2, 4 or 8 in practice).

    Returns:
        A list of ``width`` ``bytes`` objects, each of length ``len(buf)//width``.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if _HAVE_ACCEL:
        return _accel.split_planes(buf, width)
    n = len(buf) // width
    aligned = n * width
    arr = np.frombuffer(buf, dtype=np.uint8, count=aligned).reshape(n, width)
    # arr[:, j] is a strided view; ascontiguousarray makes each plane packed.
    return [np.ascontiguousarray(arr[:, j]).tobytes() for j in range(width)]


def join_planes(planes: List[bytes], width: int, tail: bytes = b"") -> bytes:
    """Inverse of :func:`split_planes`.

    Args:
        planes: list of ``width`` equal-length byte planes.
        width: bytes per element; must equal ``len(planes)``.
        tail: trailing bytes that were not part of an aligned element.

    Returns:
        The reconstructed, byte-exact original buffer.
    """
    if len(planes) != width:
        raise ValueError(
            f"expected {width} planes, got {len(planes)}"
        )
    if width == 1:
        return bytes(planes[0]) + tail
    if _HAVE_ACCEL:
        return _accel.join_planes(planes, width, tail)
    n = len(planes[0])
    out = np.empty((n, width), dtype=np.uint8)
    for j, plane in enumerate(planes):
        if len(plane) != n:
            raise ValueError("planes have inconsistent lengths")
        out[:, j] = np.frombuffer(plane, dtype=np.uint8, count=n)
    return out.tobytes() + tail


def tail_len(length: int, width: int) -> int:
    """Number of trailing bytes that do not fit into a whole element."""
    return length % width


# Mapping from a NumPy dtype's itemsize to the natural transpose width.  We
# only transpose multi-byte numeric types; 1-byte types are left as a single
# plane (the transpose would be a no-op).
def width_for_itemsize(itemsize: int) -> int:
    """Return the byte-plane width to use for a given element size."""
    return itemsize if itemsize > 1 else 1
