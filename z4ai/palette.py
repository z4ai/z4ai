# SPDX-License-Identifier: Apache-2.0

"""Palette (dictionary) transform for low-effective-precision weight tensors.

A growing share of *distributed* model weights are not full-precision: they are
quantized (INT4/INT8/FP8 via GPTQ / AWQ / bitsandbytes / ``compressed-tensors``)
and then **dequantized back into a wide float container** (bf16/fp16/fp32) for
shipping or inference.  Such a tensor has only a *small set of distinct values*
-- e.g. a per-tensor INT4 weight dequantized to bf16 holds at most ~16 distinct
bit patterns, an INT8 one ~256 -- spread thinly across the 16/32-bit value space.

A byte/bit transpose does not exploit that directly: the distinct values are
*spread out* in the value space, so each byte plane still carries a wide
alphabet.  The fix is a classic **palette / dictionary transform**: relabel the
elements to a dense ``0..k-1`` index stream plus a tiny codebook of the distinct
values.  The dense index alphabet is exactly ``k`` symbols and packs into the
narrowest integer that holds it (``uint8`` for ``k<=256``), so

* for ``k<=256`` the index stream is already *half* (bf16/fp16) or *a quarter*
  (fp32) the size of the input **before** entropy coding, and
* the dense, contiguous alphabet is far more compressible for Zstd than the
  original sparse-in-value-space stream (measured: dequantised INT8 +11.7%, INT4
  +27.5%, INT6 +15.8% vs z4ai's byte-plane path; INT4 per-group +19.2%).

The transform is a bijective relabelling, so it is **exactly lossless** for every
input.  It is gated by a cheap distinct-count probe and kept only when it
actually shrinks the frame (an AUTO best-of candidate), so it never fires on a
genuine full-entropy tensor (native gaussian bf16 has thousands of distinct
values and the gate skips it) and never regresses ratio.

The frame is self-describing (magic ``b"ZPL1"``) and optionally records NumPy
dtype/shape, mirroring :mod:`z4ai.sparse`, so the core codec adopts it as one
more best-of candidate without coupling.

References:
  * ZipNN: Lossless Compression for AI Models, Hershcovitch et al., arXiv:2411.05239.
  * Lossless Compression of NN Components ... in Low-Precision Formats, arXiv:2508.19263.
  * compressed-tensors (INT4/INT8/FP8 quantized safetensors),
    https://huggingface.co/docs/transformers/en/quantization/compressed_tensors.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np
import zstandard as zstd

MAGIC = b"ZPL1"
_VERSION = 1
_FLAG_NUMPY = 1 << 0

# Element widths we can interpret as fixed-width unsigned integers.  Width 1 is
# excluded: a single byte is already its own minimal alphabet, so a palette only
# adds a codebook with no win.
_UINT_DTYPE = {2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}

# The codebook is tiny (<= a few thousand entries); compress it hard for free.
_CODEBOOK_LEVEL = 19
# Index-stream level.  The codec passes its resolved level through; the default
# (1) is both the fastest AND the best ratio on the per-tensor-quantised case
# that matters most (zstd ratio is non-monotonic on the dense index stream -
# measured INT4 per-tensor: L1 5.42x > L3 4.65x > L9 4.95x, all >> the byte-plane
# 4.13x).  Higher levels only help the marginal per-group case, where the palette
# loses to the byte-plane path anyway and the best-of ``min()`` discards it.
_INDEX_LEVEL = 1

# Distinct-value gate.  The palette's reliable, large win is the ``k <= 256``
# regime: the index then fits in a ``uint8``, i.e. half (bf16/fp16) or a quarter
# (fp32) of the input size *before* entropy coding.  This is exactly per-tensor
# INT8/INT6/INT4 and FP8 weights dequantised into a wide container.  Above 256 the
# index needs >= 2 bytes and the win evaporates: per-group-quantised tensors
# (~1-1.3k distinct) and native weights (~4k+ distinct) both compress better
# through the byte-plane path at the fast default level, so we do not even build
# the candidate.  The best-of ``min()`` in the codec is the real safety net (it
# can never regress); this gate just avoids wasted passes on non-palette data.
# (Max-ratio mode may pass a larger ``max_distinct`` to also try per-group.)
_MAX_DISTINCT = 256


def supported_width(width: int) -> bool:
    """True if ``width`` can be treated as a fixed-width unsigned integer (2/4/8)."""
    return width in _UINT_DTYPE


def _uint_view(raw: bytes, width: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=_UINT_DTYPE[width])


def _factorize(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(vals, idx)``: the sorted distinct values and the dense 0..k-1
    index of every element, such that ``vals[idx] == u`` exactly.

    For the 2-byte case this is O(n) via a 65536-entry lookup table (a
    ``bincount`` finds the present values, a scatter builds value->index), which
    is several times faster than the sort-based ``np.unique`` on the multi-million
    element planes this runs on.  Wider widths fall back to ``np.unique`` (only
    reached for fp32/64 palettes, which the distinct gate keeps small).
    """
    if u.dtype.itemsize == 2:
        u16 = u.view(np.uint16)
        counts = np.bincount(u16, minlength=1 << 16)
        vals = np.nonzero(counts)[0].astype(np.uint16)
        lut = np.empty(1 << 16, dtype=_index_dtype(vals.size))
        lut[vals] = np.arange(vals.size, dtype=lut.dtype)
        idx = lut[u16]
        return vals, idx
    vals, inv = np.unique(u, return_inverse=True)
    return vals, inv.astype(_index_dtype(int(vals.size)), copy=False)


def _index_dtype(k: int) -> np.dtype:
    """Narrowest unsigned integer dtype that indexes ``k`` distinct values."""
    if k <= (1 << 8):
        return np.dtype("u1")
    if k <= (1 << 16):
        return np.dtype("<u2")
    return np.dtype("<u4")


# Sampled gate tuning (mirrors z4ai.sparse): below this element count the exact
# distinct count is cheap; above it we estimate on a strided subsample first and
# only build the palette when the sample's alphabet is already small.
_SAMPLE_MIN = 1 << 20  # 1M elements
_SAMPLE_ELEMS = 1 << 18  # 256K-element subsample


def should_use(raw: bytes, width: int, max_distinct: int = _MAX_DISTINCT) -> bool:
    """Cheap gate: is it worth building a palette candidate for this buffer?

    Conservative by design -- the caller keeps the smaller of the palette and the
    dense frame, so a false positive only costs one extra pass, never ratio.  For
    large buffers the distinct count is estimated on a strided subsample first so
    dense weights (the common case) skip the full ``np.unique`` scan: if even a
    256K-element sample already holds more than ``max_distinct`` distinct values
    the whole tensor certainly does, so we skip.  A small sample alphabet falls
    through to the exact count, so quantised/borderline inputs decide exactly.
    """
    if not supported_width(width) or not raw:
        return False
    count = len(raw) // width
    if count == 0:
        return False
    u = _uint_view(raw[: count * width], width)
    # Small buffer: the exact distinct count is cheap, decide exactly.
    if count <= _SAMPLE_MIN:
        return _distinct_at_least(u, max_distinct + 1) <= max_distinct
    # Large 2-byte buffer: a 65536-bin bincount is O(n) and fast, so still decide
    # exactly (no sampling error).
    if width == 2:
        return _distinct_at_least(u, max_distinct + 1) <= max_distinct
    # Large wide buffer: an exact np.unique is a full O(n log n) sort - too slow to
    # pay on every compress AND redundant with the factorize in compress().  Decide
    # on a strided subsample instead; a false positive only costs one pass that the
    # best-of min() discards, never ratio.
    step = count // _SAMPLE_ELEMS
    sample = u[::step] if step > 1 else u
    return _distinct_at_least(sample, max_distinct + 1) <= max_distinct


def _distinct_at_least(u: np.ndarray, cap: int) -> int:
    """Number of distinct values in ``u``, but stop counting once ``cap`` reached.

    For the 2-byte case a 65536-bin ``bincount`` is faster than a sort-based
    ``np.unique`` and is naturally capped.  For wider widths we fall back to
    ``np.unique`` (still gated by the subsample above for large inputs).
    """
    if u.dtype.itemsize == 2:
        counts = np.bincount(u.view(np.uint16), minlength=1)
        return int(np.count_nonzero(counts))
    # Wider: unique is O(n log n); the subsample gate bounds n for large inputs.
    return int(np.unique(u).size)


def peek_is_palette(blob: bytes) -> bool:
    """True if ``blob`` is a z4ai palette frame (starts with :data:`MAGIC`)."""
    return len(blob) >= 4 and blob[:4] == MAGIC


def compress(
    raw: bytes,
    width: int,
    *,
    level: int = _INDEX_LEVEL,
    is_numpy: bool = False,
    np_dtype: str = "",
    shape: Tuple[int, ...] = (),
) -> bytes:
    """Encode ``raw`` (a fixed-width buffer) as a palette ``b"ZPL1"`` frame.

    Args:
        raw: little-endian element bytes.
        width: element width in bytes (2/4/8).
        level: Zstd level for the index stream.
        is_numpy/np_dtype/shape: optional NumPy metadata to round-trip.

    Returns:
        A self-describing palette frame.  Fully lossless.

    Raises:
        ValueError: if ``width`` is unsupported.
    """
    if not supported_width(width):
        raise ValueError(f"palette codec does not support width {width}")

    raw = bytes(raw)
    count = len(raw) // width
    body = raw[: count * width]
    tail = raw[count * width :]  # bytes not forming a whole element, kept raw

    if count:
        u = _uint_view(body, width)
        vals, idx = _factorize(u)
        k = int(vals.size)
        codebook = vals.astype(_UINT_DTYPE[width]).tobytes()
        index_width = idx.dtype.itemsize
        codebook_comp = zstd.ZstdCompressor(level=_CODEBOOK_LEVEL).compress(codebook)
        index_comp = zstd.ZstdCompressor(level=level).compress(idx.tobytes())
    else:
        k = 0
        index_width = 1
        codebook_comp = b""
        index_comp = b""

    dtype_bytes = np_dtype.encode("ascii") if is_numpy else b""
    if len(dtype_bytes) > 255:
        raise ValueError("dtype string too long")
    shape = tuple(int(d) for d in shape) if is_numpy else ()
    if len(shape) > 255:
        raise ValueError("too many dimensions")

    out = bytearray()
    out += MAGIC
    out += struct.pack(
        "<BBBB", _VERSION, width, _FLAG_NUMPY if is_numpy else 0, index_width
    )
    out += struct.pack("<QII", count, k, len(tail))
    out += struct.pack("<BB", len(dtype_bytes), len(shape))
    out += dtype_bytes
    for dim in shape:
        out += struct.pack("<Q", dim)
    out += tail
    out += struct.pack("<I", len(codebook_comp))
    out += codebook_comp
    out += index_comp  # remainder
    return bytes(out)


def decompress(blob: bytes) -> bytes:
    """Inverse of :func:`compress`; returns the byte-exact original buffer."""
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai palette frame (bad magic)")
    off = 4
    version, width, flags, index_width = struct.unpack_from("<BBBB", mv, off)
    off += 4
    if version != _VERSION:
        raise ValueError(f"unsupported palette frame version {version}")
    count, k, tail_len = struct.unpack_from("<QII", mv, off)
    off += 16
    dtype_len, ndim = struct.unpack_from("<BB", mv, off)
    off += 2
    off += dtype_len  # NumPy metadata is consumed by the higher-level helper
    off += 8 * ndim
    tail = bytes(mv[off : off + tail_len])
    off += tail_len
    (codebook_len,) = struct.unpack_from("<I", mv, off)
    off += 4
    codebook_comp = bytes(mv[off : off + codebook_len])
    off += codebook_len
    index_comp = bytes(mv[off:])

    if count == 0:
        return tail

    vals = np.frombuffer(
        zstd.ZstdDecompressor().decompress(codebook_comp), dtype=_UINT_DTYPE[width]
    )
    if vals.size != k:
        raise ValueError("palette codebook length does not match k")
    idx_dtype = (
        np.dtype("u1")
        if index_width == 1
        else (np.dtype("<u2") if index_width == 2 else np.dtype("<u4"))
    )
    idx = np.frombuffer(zstd.ZstdDecompressor().decompress(index_comp), dtype=idx_dtype)
    if idx.size != count:
        raise ValueError("palette index length does not match element count")
    # Gather: one vectorised lookup reconstructs every element exactly.
    out = vals[idx.astype(np.intp, copy=False)]
    return out.tobytes() + tail


def read_numpy_metadata(blob: bytes) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """Return ``(dtype_str, shape)`` if the palette frame carries NumPy metadata.

    Used by :func:`z4ai.decompress_ndarray` to restore an array when the palette
    frame won the AUTO best-of.  Returns ``None`` for non-array frames.
    """
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai palette frame (bad magic)")
    _version, _width, flags, _iw = struct.unpack_from("<BBBB", mv, 4)
    if not (flags & _FLAG_NUMPY):
        return None
    off = 4 + 4 + 16
    dtype_len, ndim = struct.unpack_from("<BB", mv, off)
    off += 2
    dtype_str = bytes(mv[off : off + dtype_len]).decode("ascii")
    off += dtype_len
    shape = tuple(struct.unpack_from("<Q", mv, off + 8 * i)[0] for i in range(ndim))
    return dtype_str, shape


__all__ = [
    "MAGIC",
    "supported_width",
    "should_use",
    "peek_is_palette",
    "compress",
    "decompress",
    "read_numpy_metadata",
]
