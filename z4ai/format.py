# SPDX-License-Identifier: Apache-2.0

"""On-disk container format for z4ai blobs.

A blob is fully self-describing: it carries the transform that was applied, the
element dtype, optional NumPy dtype/shape metadata, the trailing bytes that did
not fit a whole element, and the list of (possibly raw-stored) sub-streams.

Layout (all integers little-endian)::

    magic        4 bytes   b"Z4AI"
    version      u8        format version (== FORMAT_VERSION)
    flags        u8        bit0: payload is a NumPy array (np_dtype/shape valid)
    transform    u8        0 = opaque, 1 = byte-plane, 2 = bit-field
    width        u8        byte-plane width (1 == no transpose); 0 for bit-field
    n_planes     u8        number of sub-streams following
    np_dtype_len u8        length of the NumPy dtype string (may be 0)
    np_dtype     bytes     NumPy dtype string, e.g. "<f4"
    dcode_len    u8        length of the public dtype code (may be 0)
    dtype_code   bytes     public dtype code, e.g. "bf16"/"fp32" (bit-field only)
    ndim         u8        number of shape dimensions
    shape        ndim*u64  array shape
    tail_len     u32       trailing bytes not covered by the transform
    tail         bytes     the trailing bytes, verbatim
    per plane:
        method   u8        0 = store, 1 = zstd
        orig_len u64       uncompressed plane length
        comp_len u64       stored/compressed length
        payload  bytes     comp_len bytes
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Tuple

MAGIC = b"Z4AI"
FORMAT_VERSION = 1

FLAG_NUMPY = 1 << 0

# Transform identifiers.
TRANSFORM_OPAQUE = 0  # single stream, no field/byte split
TRANSFORM_BYTEPLANE = 1  # byte-transpose into ``width`` planes
TRANSFORM_BITFIELD = 2  # sign / exponent / mantissa field split


@dataclass
class PlaneEntry:
    method: int
    orig_len: int
    data: bytes


@dataclass
class Container:
    """In-memory representation of a z4ai blob."""

    planes: List[PlaneEntry]
    transform: int = TRANSFORM_OPAQUE
    width: int = 1
    dtype_code: str = ""
    tail: bytes = b""
    is_numpy: bool = False
    dtype: str = ""  # NumPy dtype string (np frames only)
    shape: Tuple[int, ...] = field(default_factory=tuple)


def serialize(container: Container) -> bytes:
    """Encode a :class:`Container` to its byte representation."""
    flags = FLAG_NUMPY if container.is_numpy else 0
    np_dtype_bytes = container.dtype.encode("ascii")
    dcode_bytes = container.dtype_code.encode("ascii")
    if len(np_dtype_bytes) > 255 or len(dcode_bytes) > 255:
        raise ValueError("dtype string too long")
    if len(container.planes) > 255:
        raise ValueError("too many planes")

    # Assemble the frame as a list of byte chunks joined once at the end.  The
    # plane payloads (and the opaque/byteplane candidates' compressed bodies) can
    # be tens of MiB; appending them to a growing bytearray and then calling
    # bytes() on it copied the whole payload twice.  b"".join over the chunk list
    # allocates the result exactly once, which is a measurable win because the
    # best-of in codec.compress serializes several large candidate frames.
    head = bytearray()
    head += MAGIC
    head += struct.pack(
        "<BBBBB",
        FORMAT_VERSION,
        flags,
        container.transform,
        container.width,
        len(container.planes),
    )
    head += struct.pack("<B", len(np_dtype_bytes))
    head += np_dtype_bytes
    head += struct.pack("<B", len(dcode_bytes))
    head += dcode_bytes
    head += struct.pack("<B", len(container.shape))
    for dim in container.shape:
        head += struct.pack("<Q", dim)
    head += struct.pack("<I", len(container.tail))

    parts: List[bytes] = [bytes(head), container.tail]
    for plane in container.planes:
        parts.append(struct.pack("<BQQ", plane.method, plane.orig_len, len(plane.data)))
        parts.append(plane.data)
    return b"".join(parts)


def deserialize(blob: bytes) -> Container:
    """Decode a z4ai blob into a :class:`Container`."""
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai blob (bad magic)")
    off = 4
    version, flags, transform, width, n_planes = struct.unpack_from("<BBBBB", mv, off)
    off += 5
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported z4ai format version {version}")

    (np_dtype_len,) = struct.unpack_from("<B", mv, off)
    off += 1
    np_dtype = bytes(mv[off : off + np_dtype_len]).decode("ascii")
    off += np_dtype_len

    (dcode_len,) = struct.unpack_from("<B", mv, off)
    off += 1
    dtype_code = bytes(mv[off : off + dcode_len]).decode("ascii")
    off += dcode_len

    (ndim,) = struct.unpack_from("<B", mv, off)
    off += 1
    shape = []
    for _ in range(ndim):
        (dim,) = struct.unpack_from("<Q", mv, off)
        off += 8
        shape.append(dim)

    (tlen,) = struct.unpack_from("<I", mv, off)
    off += 4
    tail = bytes(mv[off : off + tlen])
    off += tlen

    planes: List[PlaneEntry] = []
    for _ in range(n_planes):
        method, orig_len, comp_len = struct.unpack_from("<BQQ", mv, off)
        off += 17
        # Zero-copy: keep the plane payload as a memoryview slice into ``blob``
        # rather than copying it out with ``bytes(...)``.  The compressed payload
        # is the bulk of the frame (tens of MiB for model weights), and every
        # downstream consumer reads it via the buffer protocol -- zstd decode
        # accepts a memoryview directly, and a stored plane flows straight into the
        # NumPy/native join.  Copying it here was ~all of deserialize's cost and a
        # measurable slice of total decompress (our weakest axis vs ZipNN); the
        # slice borrows ``blob``, which the caller holds alive for the whole decode.
        data = mv[off : off + comp_len]
        off += comp_len
        planes.append(PlaneEntry(method, orig_len, data))

    return Container(
        planes=planes,
        transform=transform,
        width=width,
        dtype_code=dtype_code,
        tail=tail,
        is_numpy=bool(flags & FLAG_NUMPY),
        dtype=np_dtype,
        shape=tuple(shape),
    )


def peek_is_z4ai(blob: bytes) -> bool:
    """Return True if ``blob`` starts with the z4ai magic."""
    return len(blob) >= 4 and blob[:4] == MAGIC
