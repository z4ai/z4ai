# SPDX-License-Identifier: Apache-2.0

"""Cross-checkpoint delta compression — a lossless win ZipNN does not offer.

Fine-tuning, continued pre-training, RLHF and EMA/optimizer bookkeeping all save
a *sequence* of checkpoints that differ only slightly from one another.  Stored
naively, each is compressed from scratch at the ~1.5x float-entropy ceiling (see
``NOTES/entropy-bound-analysis.md``).  But consecutive checkpoints are nearly
identical: most weights are unchanged bit-for-bit, and the ones that move keep
their sign and exponent and perturb only a few low mantissa bits.

z4ai exploits that by storing checkpoint *N* as a delta against checkpoint *N-1*:

    delta = current XOR reference

Bytes that did not change become ``0x00``; bytes that did change keep only the
bits that actually flipped.  The result is a buffer dominated by long zero runs
with sparse low-entropy noise, which the existing z4ai codec (LZ + entropy +
zero-aware sparse path) crushes far below the from-scratch ratio.  Reconstruction
is exact and trivial — XOR is its own inverse::

    current = delta XOR reference

XOR (not numeric subtraction) is used deliberately: it is bit-exact and perfectly
reversible for *any* dtype, with no float round-off, so the round-trip is lossless
by construction regardless of element type.

The reference checkpoint is supplied by the caller at both ends (you already have
checkpoint *N-1* on disk when you write *N*).  A delta frame is self-describing
(magic ``b"Z4DL"``) and records the reference length + a fast digest so a
mismatched reference is rejected loudly rather than silently corrupting output.

This is the technique reported in the low-precision lossless literature
(arXiv:2508.19263) as reaching exponent-stream ratios as low as ~0.07 and ~37%
total size on BF16 training-checkpoint deltas.

Example::

    >>> import z4ai
    >>> from z4ai import delta
    >>> blob = delta.compress(ckpt_n_bytes, reference=ckpt_prev_bytes, dtype="bf16")
    >>> restored = delta.decompress(blob, reference=ckpt_prev_bytes)
    >>> assert bytes(restored) == ckpt_n_bytes
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional

import numpy as np

from . import codec

MAGIC = b"Z4DL"
_VERSION = 2
# Header: magic(4) version(u8) flags(u8) reserved(u16) ref_len(u64) ref_digest(16)
_HEADER = struct.Struct("<4sBBHQ16s")
# Width of the reference digest in bytes (BLAKE2b-128).
_DIGEST_BYTES = 16


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    raise TypeError(f"expected bytes-like or numpy.ndarray, got {type(data).__name__}")


def _xor(a: bytes, b: bytes) -> bytes:
    """Byte-wise XOR of two equal-length buffers (vectorised)."""
    ua = np.frombuffer(a, dtype=np.uint8)
    ub = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(ua, ub).tobytes()


def _digest(buf: bytes) -> bytes:
    """Strong 128-bit reference digest (BLAKE2b) to detect a wrong reference.

    A delta decodes correctly ONLY against the exact reference it was built
    against; the wrong reference silently corrupts the output, which is the worst
    failure mode for a lossless codec.  The digest is therefore a *correctness*
    guard, not a speed knob: an earlier CRC32 here was linear (GF(2)-affine) and
    not collision-resistant — a same-length reference whose CRC32 happened to (or
    was forged to) match passed the check and decoded to wrong bytes silently.
    BLAKE2b-128 makes both accidental (~2^-128) and adversarial collisions
    infeasible, so a mismatched reference is always rejected.  It is C-backed and
    runs at multiple GB/s, negligible next to the XOR + compression of the same
    buffer.
    """
    return hashlib.blake2b(buf, digest_size=_DIGEST_BYTES).digest()


def peek_is_delta(blob: bytes) -> bool:
    """True if ``blob`` is a z4ai delta frame (starts with :data:`MAGIC`)."""
    return len(blob) >= 4 and blob[:4] == MAGIC


def compress(
    current,
    reference,
    *,
    dtype: Optional[str] = None,
    level: Optional[int] = None,
    threads: int = 0,
) -> bytes:
    """Compress ``current`` as a lossless delta against ``reference``.

    Args:
        current: the checkpoint to store (``bytes``/``bytearray``/``memoryview``
            or a NumPy array).
        reference: the previous checkpoint to diff against.  Must be the same
            byte length as ``current``.
        dtype: element type code passed through to the inner codec (e.g.
            ``"bf16"``), so the delta buffer is transformed/entropy-coded with the
            right element width.  ``None`` treats the delta as opaque bytes.
        level / threads: forwarded to :func:`z4ai.codec.compress`.

    Returns:
        A self-describing ``b"Z4DL"`` frame.  Decompress with the *same*
        ``reference`` via :func:`decompress`.

    Raises:
        ValueError: if ``current`` and ``reference`` differ in length.
    """
    cur = _as_bytes(current)
    ref = _as_bytes(reference)
    if len(cur) != len(ref):
        raise ValueError(
            "delta requires equal-length buffers: "
            f"current={len(cur)} bytes, reference={len(ref)} bytes"
        )

    diff = _xor(cur, ref) if cur else b""
    kw = {}
    if level is not None:
        kw["level"] = level
    inner = codec.compress(diff, dtype=dtype, threads=threads, **kw)

    header = _HEADER.pack(MAGIC, _VERSION, 0, 0, len(ref), _digest(ref))
    return header + inner


def decompress(blob: bytes, reference, *, out: Optional[bytearray] = None) -> bytes:
    """Reconstruct the original buffer from a delta frame and its ``reference``.

    Args:
        blob: a frame produced by :func:`compress`.
        reference: the SAME reference buffer used at compression time.
        out: optional writable buffer to fill and return.

    Returns:
        The byte-exact original ``current`` buffer.

    Raises:
        ValueError: if ``blob`` is not a delta frame, or ``reference`` does not
            match the one used at compression time (length or digest mismatch).
    """
    mv = memoryview(blob)
    if len(mv) < _HEADER.size or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai delta frame (bad magic)")
    magic, version, _flags, _resv, ref_len, ref_digest = _HEADER.unpack_from(mv, 0)
    if version != _VERSION:
        raise ValueError(f"unsupported delta frame version {version}")

    ref = _as_bytes(reference)
    if len(ref) != ref_len:
        raise ValueError(
            f"reference length mismatch: frame expects {ref_len} bytes, "
            f"got {len(ref)} — wrong reference checkpoint?"
        )
    if _digest(ref) != ref_digest:
        raise ValueError(
            "reference digest mismatch — the supplied reference is not the one "
            "this delta was built against (decoding would corrupt the output)"
        )

    diff = codec.decompress(bytes(mv[_HEADER.size :]))
    result = _xor(diff, ref) if ref else b""

    if out is not None:
        n = len(result)
        if len(out) < n:
            raise ValueError(f"out buffer too small: need {n}, have {len(out)}")
        out[:n] = result
        return out
    return result


__all__ = ["MAGIC", "peek_is_delta", "compress", "decompress"]
