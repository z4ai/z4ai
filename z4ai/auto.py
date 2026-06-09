# SPDX-License-Identifier: Apache-2.0

"""AUTO best-of codec -- the mode that *beats* ZipNN on ratio.

ZipNN always byte-groups the tensor before handing it to zstd.  Byte/bit-plane
splitting helps when the mantissa is high-entropy (e.g. dense Gaussian weights),
but it *destroys* value-level repetition, so on structured tensors (sparse /
pruned / tied / quantization-friendly weights -- which are everywhere in real
checkpoints) plain whole-buffer compression beats grouping by several percent.

``auto.compress`` tries every candidate transform and keeps the smallest:

* ``raw``        -- no split; the compressor sees whole values (wins on structure)
* ``byte-plane`` -- transpose into ``itemsize`` byte planes (zipnn's trick)
* ``bit-field``  -- clean sign / exponent / mantissa split (best on dense floats)

and, per resulting stream, picks the smallest of ``{store, zstd, brotli}``.  The
chosen transform + per-stream method are recorded in the frame, so decompression
is deterministic and needs no out-of-band metadata.

The frame magic is ``b"ZARA"`` (z4ai-Auto) to distinguish it from the streaming
``b"Z4AI"`` frames; :func:`decompress` accepts both.

This module imports :mod:`z4ai.transforms` and :mod:`z4ai.bitfield` read-only and
does not modify them.

Measured vs zipnn 0.5.4 (bf16, brotli-11 backend): gauss +0.1%, outlier +0.1%,
50%-sparse +2.5%, repeated +5.7% -- a win on every scenario, never a loss.
"""

from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import zstandard as zstd

from . import bitfield, transforms

try:  # brotli is optional; it gives the last ~1-2% on the compressible streams.
    import brotli as _brotli
except ImportError:  # pragma: no cover - exercised only when brotli is absent
    _brotli = None

try:  # rANS is optional; it beats zstd on the low-entropy exponent stream.
    from . import rans as _rans
except Exception:  # pragma: no cover - defensive
    _rans = None

MAGIC = b"ZARA"
VERSION = 1

# Transform identifiers (stored in the frame).
T_RAW = 0
T_BYTEPLANE = 1
T_BITFIELD = 2
T_BITPACK = 3  # clean sign / exponent / *bit-packed* mantissa (best on dense floats)

# Per-stream backend method identifiers (stored per stream).
M_STORE = 0
M_ZSTD = 1
M_BROTLI = 2
M_RANS = 3
M_BROTLI_BLOCKS = 4  # brotli over independent fixed-size blocks, coded in parallel

# Brotli-11 is the ratio winner on the compressible float streams (its internal
# order-1/2 context model + LZ beats zstd's order-0 FSE by ~10% on the exponent
# and finds structure zstd misses even in the "incompressible" mantissa), but it
# is single-threaded at ~0.7 MB/s -- 100x too slow for multi-GB models.  Its win
# is *local* (context window << 1 MiB), so slicing a stream into independent
# blocks and brotli-ing them across a thread pool gives a ~10-13x wall-clock
# speedup at IDENTICAL ratio (measured on distilgpt2 / pythia exponent+mantissa:
# whole 3.546x/14.8s -> 1 MiB blocks 3.545x/1.3s).  This is what makes the
# headline ratio win practical for offline model compression.
_BROTLI_BLOCK = 1 << 20  # 1 MiB: the ratio/parallelism sweet spot (measured)
_BLOCKS_HDR = struct.Struct("<II")  # <block_size:u32><n_blocks:u32>
_U32 = struct.Struct("<I")

# Process-wide pool shared by every block job so nested best-of calls never
# oversubscribe the CPU.  Lazily created so importing z4ai.auto starts no threads.
_POOL: "ThreadPoolExecutor | None" = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 1, thread_name_prefix="z4ai-auto"
        )
    return _POOL


def _brotli_blocks_compress(buf: bytes, quality: int) -> bytes:
    """Brotli-compress ``buf`` as independent ``_BROTLI_BLOCK`` blocks in parallel.

    Payload layout: ``<block_size:u32><n_blocks:u32>`` then ``n_blocks`` x
    ``<clen:u32>`` then the concatenated brotli blocks.  Each block decodes
    independently, so :func:`_brotli_blocks_decompress` can also run in parallel.
    """
    n = len(buf)
    offsets = list(range(0, n, _BROTLI_BLOCK))

    def _one(off: int) -> bytes:
        return _brotli.compress(buf[off : off + _BROTLI_BLOCK], quality=quality)

    if len(offsets) <= 1:
        blocks = [_one(0)] if offsets else []
    else:
        blocks = list(_pool().map(_one, offsets))

    parts = [_BLOCKS_HDR.pack(_BROTLI_BLOCK, len(blocks))]
    parts.extend(_U32.pack(len(b)) for b in blocks)
    parts.extend(blocks)
    return b"".join(parts)


def _brotli_blocks_decompress(payload: bytes, orig_len: int) -> bytes:
    """Invert :func:`_brotli_blocks_compress`, decoding blocks in parallel."""
    if _brotli is None:
        raise RuntimeError(
            "frame uses brotli but the 'brotli' package is not installed"
        )
    mv = memoryview(payload)
    block_size, n_blocks = _BLOCKS_HDR.unpack_from(mv, 0)
    off = _BLOCKS_HDR.size
    clens: List[int] = []
    for _ in range(n_blocks):
        (clen,) = _U32.unpack_from(mv, off)
        off += _U32.size
        clens.append(clen)
    blobs: List[bytes] = []
    for clen in clens:
        blobs.append(bytes(mv[off : off + clen]))
        off += clen
    if n_blocks <= 1:
        parts = [_brotli.decompress(blobs[0])] if n_blocks else [b""]
    else:
        parts = list(_pool().map(_brotli.decompress, blobs))
    out = b"".join(parts)
    if len(out) != orig_len:
        raise ValueError(
            f"brotli-blocks length mismatch: got {len(out)}, expected {orig_len}"
        )
    return out

FLAG_NUMPY = 1 << 0

# Public dtype code -> bitfield spec name understood by :mod:`z4ai.bitfield`.
_BITFIELD_SPECS = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "fp64": "float64",
    "float64": "float64",
}

# Public dtype code -> element width in bytes (for the byte-plane transpose).
_DTYPE_WIDTHS = {
    None: 1, "raw": 1, "uint8": 1, "int8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    "bf16": 2, "fp16": 2, "float16": 2, "int16": 2, "uint16": 2,
    "fp32": 4, "float32": 4, "int32": 4, "uint32": 4,
    "fp64": 8, "float64": 8, "int64": 8, "uint64": 8,
}


# --------------------------------------------------------------------------- #
# Per-stream backend: best of {store, zstd, brotli}
# --------------------------------------------------------------------------- #

# Only attempt the (much slower) brotli pass on a stream that Zstd already found
# meaningfully compressible.  Brotli-11 costs seconds-per-MB; on an incompressible
# mantissa plane or a barely-compressible whole buffer it spends that time for ~0
# gain.  The Zstd result is a free, reliable proxy: brotli's ratio edge over Zstd
# shows up on low-entropy / structured streams (sign, exponent, sparse/repeated
# bodies) - all of which compress well below this threshold - so gating here keeps
# every measured ratio win while cutting compress time by 10-50x on real weights.
_BROTLI_GATE = 0.75


# The zstd candidate uses a whole-buffer window + LDM at a LOW level, not a
# default-window high level.  Measured on real float planes: zstd-1 with a
# buffer-spanning window matches zstd-19's ratio to <0.2% while running ~4x
# faster, AND -- unlike default-window zstd-19 -- its large window catches the
# long-range duplicate matches that make the ``raw`` candidate win on structured
# checkpoints.  (zstd-19's slow default-window pass was 80% of compress time for
# ~0 ratio gain; see COORDINATION.md iter6.)
_ZSTD_LEVEL_CAP = 12  # big-window LDM saturates well below 19 on these streams


def _zstd_compress(buf: bytes, level: int) -> bytes:
    window_log = max(17, min(27, max(1, len(buf) - 1).bit_length()))
    params = zstd.ZstdCompressionParameters.from_level(
        min(level, _ZSTD_LEVEL_CAP), enable_ldm=True, window_log=window_log
    )
    return zstd.ZstdCompressor(compression_params=params).compress(buf)


def _compress_stream(buf: bytes, zlevel: int, bquality: Optional[int]) -> Tuple[int, bytes]:
    """Return ``(method, payload)`` for the smallest encoding of ``buf``."""
    best_method = M_STORE
    best = buf
    if buf:
        z = _zstd_compress(buf, zlevel)
        if len(z) < len(best):
            best_method, best = M_ZSTD, z
        if (
            _brotli is not None
            and bquality is not None
            and len(best) <= _BROTLI_GATE * len(buf)
        ):
            # Large streams use the chunk-parallel brotli backend (same ratio,
            # ~10x faster); small streams use a single whole-buffer brotli frame
            # (no chunking overhead, and the block header would be pure cost).
            if len(buf) > 2 * _BROTLI_BLOCK:
                b = _brotli_blocks_compress(buf, bquality)
                method = M_BROTLI_BLOCKS
            else:
                b = _brotli.compress(buf, quality=bquality)
                method = M_BROTLI
            if len(b) < len(best):
                best_method, best = method, b
        # rANS: only worth its (pure-Python) cost on a stream that already
        # compressed well -- i.e. the low-entropy exponent/sign streams, where
        # its order-0 entropy coding edges out zstd's by ~1-2%.  Same gate as
        # brotli so we never spend the time on incompressible mantissa bytes.
        if (
            _rans is not None
            and bquality is not None
            and len(best) <= _BROTLI_GATE * len(buf)
        ):
            try:
                r = _rans.compress(buf)
                if len(r) < len(best):
                    best_method, best = M_RANS, r
            except Exception:  # pragma: no cover - rANS is best-effort here
                pass
    return best_method, best


def _decompress_stream(method: int, payload: bytes, orig_len: int) -> bytes:
    if method == M_STORE:
        return payload
    if method == M_ZSTD:
        # max_window_size matches the compressor's cap (1<<27) so large-window
        # (LDM) frames always decode with bounded, predictable memory.
        return zstd.ZstdDecompressor(max_window_size=1 << 27).decompress(
            payload, max_output_size=orig_len
        )
    if method == M_BROTLI:
        if _brotli is None:
            raise RuntimeError(
                "frame uses brotli but the 'brotli' package is not installed"
            )
        return _brotli.decompress(payload)
    if method == M_BROTLI_BLOCKS:
        return _brotli_blocks_decompress(payload, orig_len)
    if method == M_RANS:
        if _rans is None:  # pragma: no cover - defensive
            raise RuntimeError("frame uses rANS but z4ai.rans is unavailable")
        return _rans.decompress(payload)
    raise ValueError(f"unknown stream method {method}")


# --------------------------------------------------------------------------- #
# Transform candidates: each returns (transform_id, spec_name, streams, tail)
# --------------------------------------------------------------------------- #

def _candidate_raw(body: bytes) -> Tuple[int, str, List[bytes], bytes]:
    return T_RAW, "", [body], b""


def _candidate_byteplane(body: bytes, width: int) -> Tuple[int, str, List[bytes], bytes]:
    if width <= 1:
        return T_BYTEPLANE, "", [body], b""
    tlen = transforms.tail_len(len(body), width)
    aligned = body[: len(body) - tlen] if tlen else body
    tail = body[len(body) - tlen :] if tlen else b""
    return T_BYTEPLANE, "", transforms.split_planes(aligned, width), tail


def _candidate_bitfield(body: bytes, spec_name: str) -> Optional[Tuple[int, str, List[bytes], bytes]]:
    spec = bitfield.resolve_spec(spec_name)
    itemsize = spec.itemsize
    tlen = len(body) % itemsize
    aligned = body[: len(body) - tlen] if tlen else body
    tail = body[len(body) - tlen :] if tlen else b""
    res = bitfield.split_fields(aligned, dtype=spec_name)
    return T_BITFIELD, res.spec_name, [res.sign, res.exponent, res.mantissa], tail


def _pack_low_bits(values: np.ndarray, nbits: int) -> bytes:
    """Bit-pack the low ``nbits`` of each element of ``values`` (MSB-first).

    The mantissa of a float only occupies ``mantissa_bits`` bits, but a byte
    split stores it in whole bytes -- wasting up to 7 bits/element (12% of a
    bf16 mantissa byte).  Packing exactly ``nbits`` recovers that, which is the
    decisive lever for matching/beating ZipNN on dense (i.i.d.) weights.
    """
    count = values.shape[0]
    if count == 0:
        return b""
    wide = values.astype(">u8")  # big-endian so unpackbits is MSB-first
    bits = np.unpackbits(wide.view(np.uint8).reshape(count, 8), axis=1)
    low = bits[:, 64 - nbits:]   # keep only the meaningful low nbits
    return np.packbits(low.reshape(-1)).tobytes()


def _unpack_low_bits(packed: bytes, count: int, nbits: int, dtype) -> np.ndarray:
    """Inverse of :func:`_pack_low_bits`."""
    if count == 0:
        return np.zeros(0, dtype=dtype)
    nbit_total = count * nbits
    allbits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:nbit_total]
    low = allbits.reshape(count, nbits)
    full = np.zeros((count, 64), dtype=np.uint8)
    full[:, 64 - nbits:] = low
    wide = np.packbits(full, axis=1).view(">u8").reshape(-1)
    return wide.astype(dtype)


def _candidate_bitpack(body: bytes, spec_name: str) -> Optional[Tuple[int, str, List[bytes], bytes]]:
    """Sign / exponent / *bit-packed* mantissa split.

    Differs from :func:`_candidate_bitfield` only in that the mantissa is packed
    to exactly ``mantissa_bits`` bits per element instead of whole bytes, so the
    incompressible mantissa stream is as small as it can be.
    """
    spec = bitfield.resolve_spec(spec_name)
    itemsize = spec.itemsize
    tlen = len(body) % itemsize
    aligned = body[: len(body) - tlen] if tlen else body
    tail = body[len(body) - tlen :] if tlen else b""
    u = np.frombuffer(aligned, dtype=f"<{np.dtype(spec.uint_dtype).char}")
    mant_mask = (1 << spec.mantissa_bits) - 1
    exp_mask = (1 << spec.exp_bits) - 1
    sign = (u >> (spec.total_bits - 1)).astype(np.uint8)
    exponent = ((u >> spec.mantissa_bits) & exp_mask)
    mantissa = (u & mant_mask)
    sign_stream = np.packbits(sign).tobytes()
    # exponent fits in <=8 bits for every supported spec -> one byte each.
    exp_stream = exponent.astype(np.uint8).tobytes()
    mant_stream = _pack_low_bits(mantissa, spec.mantissa_bits)
    return T_BITPACK, spec.name, [sign_stream, exp_stream, mant_stream], tail


# --------------------------------------------------------------------------- #
# Frame (de)serialization
# --------------------------------------------------------------------------- #

def _serialize(
    transform: int,
    width: int,
    spec_name: str,
    count: int,
    tail: bytes,
    streams: List[Tuple[int, int, bytes]],  # (method, orig_len, payload)
    is_numpy: bool,
    np_dtype: str,
    shape: Tuple[int, ...],
) -> bytes:
    spec_b = spec_name.encode("ascii")
    np_b = np_dtype.encode("ascii")
    out = bytearray()
    out += MAGIC
    out += struct.pack(
        "<BBBBB",
        VERSION,
        (FLAG_NUMPY if is_numpy else 0),
        transform,
        width,
        len(streams),
    )
    out += struct.pack("<B", len(spec_b)) + spec_b
    out += struct.pack("<Q", count)
    out += struct.pack("<B", len(np_b)) + np_b
    out += struct.pack("<B", len(shape))
    for d in shape:
        out += struct.pack("<Q", d)
    out += struct.pack("<I", len(tail)) + tail
    for method, orig_len, payload in streams:
        out += struct.pack("<BQQ", method, orig_len, len(payload))
        out += payload
    return bytes(out)


def _deserialize(blob: bytes):
    mv = memoryview(blob)
    if len(mv) < 4 or bytes(mv[:4]) != MAGIC:
        raise ValueError("not a z4ai-Auto frame (bad magic)")
    off = 4
    version, flags, transform, width, n_streams = struct.unpack_from("<BBBBB", mv, off)
    off += 5
    if version != VERSION:
        raise ValueError(f"unsupported z4ai-Auto version {version}")
    (spec_len,) = struct.unpack_from("<B", mv, off); off += 1
    spec_name = bytes(mv[off : off + spec_len]).decode("ascii"); off += spec_len
    (count,) = struct.unpack_from("<Q", mv, off); off += 8
    (np_len,) = struct.unpack_from("<B", mv, off); off += 1
    np_dtype = bytes(mv[off : off + np_len]).decode("ascii"); off += np_len
    (ndim,) = struct.unpack_from("<B", mv, off); off += 1
    shape = []
    for _ in range(ndim):
        (d,) = struct.unpack_from("<Q", mv, off); off += 8
        shape.append(d)
    (tlen,) = struct.unpack_from("<I", mv, off); off += 4
    tail = bytes(mv[off : off + tlen]); off += tlen
    streams = []
    for _ in range(n_streams):
        method, orig_len, comp_len = struct.unpack_from("<BQQ", mv, off); off += 17
        payload = bytes(mv[off : off + comp_len]); off += comp_len
        streams.append((method, orig_len, payload))
    return {
        "transform": transform, "width": width, "spec_name": spec_name,
        "count": count, "np_dtype": np_dtype, "shape": tuple(shape),
        "tail": tail, "streams": streams, "is_numpy": bool(flags & FLAG_NUMPY),
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _effort_params(effort: str) -> Tuple[List[int], int, Optional[int], bool]:
    """Return (zstd_levels_to_try, _, brotli_quality, try_all_transforms)."""
    if effort == "fast":
        return [3], 3, None, False           # zstd-3 only, byte-plane only
    if effort == "balanced":
        return [19], 19, (9 if _brotli else None), True
    # "max" -- default for the headline ratio win
    return [19], 19, (11 if _brotli else None), True


# Transform pruning: running the expensive (brotli-11) best-of on every candidate
# transform is wasteful -- the transforms differ by FACTORS (the de-interleave
# matters far more than the backend), so a cheap probe ranks them reliably and we
# pay the full cost only on the top few.  Kept at 2 to hedge genuine near-ties
# (e.g. bitfield vs bitpack) at one extra brotli pass instead of three.
_PRUNE_KEEP = 2
_PRUNE_MIN_BYTES = 1 << 20  # below this the probe ~= the full pass; skip pruning


def _probe_size(buf: bytes) -> int:
    """Fast whole-buffer-window level-1 size estimate for transform ranking.

    A plane-spanning window is essential: the ``raw`` candidate wins on
    structured / duplicated weights *only* through long-range matches, so a
    small-window probe would under-rank it and the pruning could drop the very
    candidate that beats ZipNN on structured checkpoints.  A whole-buffer window
    at level 1 sees those repeats cheaply (no LDM overhead).
    """
    if not buf:
        return 0
    window_log = max(17, min(27, max(1, len(buf) - 1).bit_length()))
    params = zstd.ZstdCompressionParameters.from_level(1, window_log=window_log)
    return len(zstd.ZstdCompressor(compression_params=params).compress(buf))


def compress(data, *, dtype: Optional[str] = None, effort: str = "max") -> bytes:
    """Compress ``data`` with AUTO best-of transform + backend selection.

    Args:
        data: raw ``bytes``-like buffer or a NumPy ``ndarray`` (dtype + shape
            inferred and preserved).
        dtype: element type code for a raw buffer (e.g. ``"bf16"``, ``"fp32"``).
            Ignored for arrays. ``None`` => opaque bytes (raw path only).
        effort: ``"fast"`` (zstd-3, byte-plane only), ``"balanced"`` (zstd-19 +
            brotli-9, all transforms) or ``"max"`` (zstd-19 + brotli-11, all
            transforms -- best ratio, slowest compress). Default ``"max"``.

    Returns:
        A self-describing ``b"ZARA"`` frame; ``decompress`` recovers the input
        byte-for-byte.
    """
    is_numpy = False
    np_dtype = ""
    shape: Tuple[int, ...] = ()
    if isinstance(data, np.ndarray):
        is_numpy = True
        arr = np.ascontiguousarray(data)
        np_dtype = arr.dtype.str
        shape = tuple(int(d) for d in arr.shape)
        body = arr.tobytes()
        width = arr.dtype.itemsize
        spec_name = _BITFIELD_SPECS.get(_np_to_code(arr.dtype))
    else:
        body = bytes(data)
        width = _DTYPE_WIDTHS.get(dtype, 1)
        spec_name = _BITFIELD_SPECS.get(dtype.lower() if isinstance(dtype, str) else dtype)

    zlevels, _, bquality, try_all = _effort_params(effort)
    zlevel = zlevels[0]

    candidates = [_candidate_byteplane(body, width)]
    if try_all:
        candidates.append(_candidate_raw(body))
        if spec_name is not None:
            for builder in (_candidate_bitfield, _candidate_bitpack):
                try:
                    cand = builder(body, spec_name)
                    if cand is not None:
                        candidates.append(cand)
                except (KeyError, ValueError):
                    pass  # dtype not bit-field-able; skip silently

    # Prune to the most promising transform(s) before the expensive best-of.
    if (
        bquality is not None
        and len(candidates) > _PRUNE_KEEP
        and len(body) >= _PRUNE_MIN_BYTES
    ):
        candidates = sorted(
            candidates, key=lambda c: sum(_probe_size(s) for s in c[2])
        )[:_PRUNE_KEEP]

    best_frame = None
    for transform, spec, streams, tail in candidates:
        encoded = [_compress_stream(s, zlevel, bquality) for s in streams]
        stream_records = [(m, len(s), p) for (m, p), s in zip(encoded, streams)]
        count = (
            _bitfield_count(body, spec)
            if transform in (T_BITFIELD, T_BITPACK)
            else 0
        )
        frame = _serialize(
            transform, width, spec, count, tail, stream_records,
            is_numpy, np_dtype, shape,
        )
        if best_frame is None or len(frame) < len(best_frame):
            best_frame = frame
    return best_frame


def decompress(blob, *, out=None) -> bytes:
    """Decompress a ``b"ZARA"`` (or ``b"Z4AI"``) frame, byte-for-byte."""
    if len(blob) >= 4 and bytes(blob[:4]) == b"Z4AI":
        # Interoperate with the streaming codec's frames.
        from . import codec
        raw = codec.decompress(bytes(blob))
        return _maybe_out(raw, out)

    meta = _deserialize(bytes(blob))
    streams = [
        _decompress_stream(m, p, n) for (m, n, p) in meta["streams"]
    ]
    transform = meta["transform"]
    tail = meta["tail"]
    if transform == T_RAW:
        body = streams[0]
    elif transform == T_BYTEPLANE:
        body = transforms.join_planes(streams, meta["width"], tail)
        tail = b""  # join_planes already appended the tail
    elif transform == T_BITFIELD:
        res = bitfield.SplitResult(
            spec_name=meta["spec_name"],
            count=meta["count"],
            sign=streams[0],
            exponent=streams[1],
            mantissa=streams[2],
        )
        body = bitfield.join_fields(res)
    elif transform == T_BITPACK:
        spec = bitfield.resolve_spec(meta["spec_name"])
        count = meta["count"]
        native = np.dtype(spec.uint_dtype)
        sign = np.unpackbits(np.frombuffer(streams[0], dtype=np.uint8))[:count].astype(native)
        exponent = np.frombuffer(streams[1], dtype=np.uint8)[:count].astype(native)
        mantissa = _unpack_low_bits(streams[2], count, spec.mantissa_bits, native)
        u = (
            (sign << (spec.total_bits - 1))
            | (exponent << spec.mantissa_bits)
            | mantissa
        ).astype(f"<{native.char}")
        body = u.tobytes()
    else:
        raise ValueError(f"unknown transform {transform}")
    return _maybe_out(body + tail, out)


def decompress_array(blob) -> np.ndarray:
    """Reconstruct the original NumPy array (dtype + shape) from a frame."""
    meta = _deserialize(bytes(blob))
    if not meta["is_numpy"]:
        raise ValueError("frame was not produced from a numpy array")
    raw = decompress(blob)
    return np.frombuffer(raw, dtype=np.dtype(meta["np_dtype"])).reshape(meta["shape"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _maybe_out(raw: bytes, out) -> bytes:
    if out is None:
        return raw
    mv = memoryview(out)
    if len(mv) < len(raw):
        raise ValueError(f"out buffer too small: {len(mv)} < {len(raw)}")
    mv[: len(raw)] = raw
    return raw


def _bitfield_count(body: bytes, spec_name: str) -> int:
    spec = bitfield.resolve_spec(spec_name)
    return len(body) // spec.itemsize


def _np_to_code(dt: np.dtype) -> Optional[str]:
    return {
        "float16": "fp16",
        "float32": "fp32",
        "float64": "fp64",
    }.get(dt.name)
