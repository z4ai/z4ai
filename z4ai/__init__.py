# SPDX-License-Identifier: Apache-2.0

"""z4ai: a lossless compression codec tuned for neural-network model weights.

z4ai transposes a tensor into byte planes (concentrating the low-entropy
exponent bytes) and compresses each plane with Zstandard configured for
long-distance matching over a large window, with a sparse zero-aware path for
pruned checkpoints.  That LZ stage captures the long-range redundancy in real
checkpoints - repeated values, pruned-to-zero regions, shared quantization
levels - which ZipNN's order-0 entropy coder cannot see.  On a real BF16
checkpoint this is roughly a +70% ratio win over ZipNN, while remaining a strict
match on purely random data.  See FINDINGS.md for measurements.

The public surface is intentionally tiny and mirrors ZipNN so z4ai is a
drop-in alternative (see ``API_CONTRACT.md``)::

    >>> import z4ai
    >>> blob = z4ai.compress(weights_bytes, dtype="bf16")
    >>> assert bytes(z4ai.decompress(blob)) == weights_bytes

For NumPy arrays the dtype and shape are inferred and round-tripped exactly::

    >>> import numpy as np, z4ai
    >>> w = np.random.randn(1000, 1000).astype(np.float32)
    >>> blob = z4ai.compress_ndarray(w)
    >>> np.array_equal(w, z4ai.decompress_ndarray(blob))
    True
"""

from __future__ import annotations

import os
from typing import Optional, Union

from . import auto as _autocodec
from . import backend, codec, format, transforms
from .codec import (
    DEFAULT_ESCALATE_LEVEL,
    DEFAULT_LEVEL,
    compress_array,
    decompress_array,
)
from .format import peek_is_z4ai
from . import delta
from . import model_delta

__all__ = [
    "compress",
    "decompress",
    "compress_ndarray",
    "decompress_ndarray",
    "compress_array",
    "decompress_array",
    "compress_delta",
    "decompress_delta",
    "delta",
    "model_delta",
    "peek_is_z4ai",
    "dtype_width",
    "DTYPE_WIDTHS",
    "DEFAULT_LEVEL",
    "DEFAULT_ESCALATE_LEVEL",
    "__version__",
]

__version__ = "0.1.0"

#: Element width in bytes for every dtype code understood by :func:`compress`.
#: The byte-plane transpose width is derived directly from these.  ``None`` and
#: any 1-byte code are treated as an opaque buffer (no float-plane split).
DTYPE_WIDTHS = {
    None: 1,
    "raw": 1,
    "uint8": 1,
    "int8": 1,
    "bool": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "bf16": 2,
    "fp16": 2,
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
}


def dtype_width(dtype: Optional[str]) -> int:
    """Return the byte-plane width for a dtype code (see :data:`DTYPE_WIDTHS`)."""
    try:
        return DTYPE_WIDTHS[dtype]
    except KeyError:
        known = sorted(k for k in DTYPE_WIDTHS if k is not None)
        raise ValueError(
            f"unknown dtype code {dtype!r}; known codes: {known}"
        ) from None


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    raise TypeError(
        "data must be bytes-like or a numpy.ndarray, got "
        f"{type(data).__name__}"
    )


def _codec_kwargs(level: Optional[int], threads: int) -> dict:
    """Translate the public ``level``/``threads`` knobs to codec internals."""
    kw: dict = {}
    if level is not None:
        kw["level"] = level
    # ``threads`` follows the ZipNN convention: 0 = auto (all cores), 1 =
    # single-threaded, n = n workers.  We fan out across planes (each plane is
    # independent); zstd's own threads stay at 0 for deterministic output.
    if threads == 1:
        kw["plane_workers"] = 1
    elif threads and threads > 1:
        kw["plane_workers"] = threads
    else:  # 0 or None -> auto
        kw["plane_workers"] = max(1, os.cpu_count() or 1)
    return kw


def compress(
    data: Union[bytes, bytearray, memoryview, "object"],
    *,
    dtype: Optional[str] = None,
    level: Optional[int] = None,
    threads: int = 0,
    auto: bool = True,
    effort: Optional[str] = None,
) -> bytes:
    """Compress a buffer (or NumPy array) into a self-describing z4ai frame.

    Args:
        data: raw ``bytes``/``bytearray``/``memoryview``, or a ``numpy.ndarray``
            (dtype and shape are then inferred and round-tripped).
        dtype: element type code when ``data`` is a raw buffer, e.g. ``"bf16"``,
            ``"fp16"``, ``"fp32"``, ``"fp64"``.  ``None`` treats the buffer as
            opaque bytes (still compressed, no float-plane split).  Ignored when
            ``data`` is a NumPy array.
        level: Zstd level (1..22).  ``None`` uses the codec default.
        threads: ``0`` = auto (all cores), ``1`` = single-threaded, ``n`` = n
            workers.
        auto: when ``True`` (default), also try plain whole-buffer Zstd, the
            byte-plane transpose, and the sparse zero-aware path, keeping the
            smallest frame.  Set ``False`` to force the dtype's primary
            transform only (fastest compress; skips the best-of passes).
        effort: ratio/speed tier. ``None`` (default) uses the fast streaming
            codec (the balanced default — multi-GB/s, ratio ties-or-beats ZipNN).
            ``"max"`` selects the high-ratio AUTO codec, which adds a
            context-modeling backend (brotli, chunk-parallel) on top of the
            field split for the **smallest possible file** — the right choice for
            model *distribution* (compress once, download many): measured
            **+3% to +12% smaller than ZipNN** on real transformers, at a
            ~3-12 MB/s offline compress and ~1 GB/s decompress.  ``"balanced"``
            is an intermediate tier.  See ``README.md`` for the comparison table.

    Returns:
        A ``bytes`` frame: ``b"Z4AI"`` for the default codec, ``b"ZARA"`` for
        ``effort="max"``/``"balanced"``.  :func:`decompress` accepts either;
        decompression needs no side information.
    """
    # Lazy ndarray detection so ``import z4ai`` never requires numpy at import
    # time beyond what the codec already pulls in.
    is_array = hasattr(data, "dtype") and hasattr(data, "shape") and hasattr(data, "tobytes")

    # Always build the fast streaming frame first: it is the floor (multi-GB/s)
    # and, on long-range-duplicate weights, its whole-buffer LDM beats the AUTO
    # codec's chunk-parallel brotli (which forfeits cross-chunk dedup).
    #
    # ``effort="fast"`` additionally drops the per-plane rANS entropy candidates:
    # rANS reaches the entropy floor (a small ratio win) but its streams decode
    # ~6x slower than the block-parallel Zstd path, so the decode-speed tier keeps
    # the fast multi-core decode.  The ratio-first default / "max" tiers leave it on.
    use_rans = effort != "fast"
    if is_array:
        default_frame = compress_array(
            data, auto=auto, rans=use_rans, **_codec_kwargs(level, threads)
        )
    else:
        buf = _as_bytes(data)
        dtype_width(dtype)  # validate dtype code early for a friendly error
        default_frame = codec.compress(
            buf, dtype=dtype, auto=auto, rans=use_rans, **_codec_kwargs(level, threads)
        )

    want_max = effort is not None and effort != "fast"
    if not want_max:
        return default_frame

    # High-ratio tier: also build the AUTO frame (field split + brotli/context
    # backend) and keep whichever is smaller.  This makes ``effort="max"`` a
    # strict superset of the default — never worse in ratio, and on data where
    # brotli does not help (e.g. i.i.d. mantissa) it transparently keeps the fast
    # default frame (which also decodes faster).
    if is_array:
        auto_frame = _autocodec.compress(data, effort=effort)
    else:
        auto_frame = _autocodec.compress(buf, dtype=dtype, effort=effort)
    return auto_frame if len(auto_frame) < len(default_frame) else default_frame


def decompress(blob: bytes, *, out: Optional[Union[bytearray, memoryview]] = None):
    """Decompress a z4ai frame back to the original buffer (byte-identical).

    Args:
        blob: a frame produced by :func:`compress`.
        out: optional writable buffer to decode into; when given it is filled
            and returned.

    Returns:
        The original ``bytes`` (or ``out`` if provided).
    """
    # ``effort="max"`` frames carry the ``b"ZARA"`` magic; everything else is a
    # streaming ``b"Z4AI"`` (or sparse) frame handled by the core codec.
    if len(blob) >= 4 and bytes(blob[:4]) == b"ZARA":
        result = _autocodec.decompress(blob)
    else:
        result = codec.decompress(blob)
    if out is not None:
        mv = memoryview(out)
        if len(mv) < len(result):
            raise ValueError(
                f"out buffer too small: {len(mv)} < {len(result)}"
            )
        mv[: len(result)] = result
        return out
    return result


# --- NumPy convenience helpers ------------------------------------------------

def compress_ndarray(
    arr, *, level: Optional[int] = None, threads: int = 0, effort: Optional[str] = None
) -> bytes:
    """Compress a NumPy array, preserving dtype and shape (see :func:`decompress_ndarray`).

    ``effort="max"`` selects the high-ratio AUTO codec (see :func:`compress`).
    """
    return compress(arr, level=level, threads=threads, effort=effort)


def decompress_ndarray(blob: bytes):
    """Reconstruct the original NumPy array (dtype + shape) from a z4ai frame."""
    if len(blob) >= 4 and bytes(blob[:4]) == b"ZARA":
        return _autocodec.decompress_array(blob)
    return decompress_array(blob)


# --- Cross-checkpoint delta -------------------------------------------------

def compress_delta(
    current,
    reference,
    *,
    dtype: Optional[str] = None,
    level: Optional[int] = None,
    threads: int = 0,
) -> bytes:
    """Losslessly compress ``current`` as a delta against a ``reference`` checkpoint.

    Consecutive training/fine-tuning checkpoints differ only slightly; storing
    the bit-exact XOR delta (``current ^ reference``) makes most bytes zero and
    compresses far below the from-scratch float-entropy ceiling.  Decompress with
    :func:`decompress_delta` and the *same* ``reference``.  See :mod:`z4ai.delta`.
    """
    return delta.compress(
        current, reference, dtype=dtype, level=level, threads=threads
    )


def decompress_delta(blob: bytes, reference, *, out: Optional[bytearray] = None):
    """Reconstruct a checkpoint from a delta frame and its ``reference`` (see
    :func:`compress_delta`)."""
    return delta.decompress(blob, reference, out=out)
