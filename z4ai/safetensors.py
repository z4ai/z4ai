# SPDX-License-Identifier: Apache-2.0

"""Apply the z4ai codec to ``.safetensors`` checkpoints, tensor by tensor.

This is the production use case ZipNN is built around (``zipnn_safetensors`` /
``zipnn_hf``): take a Hugging Face / vLLM ``.safetensors`` checkpoint and shrink
it losslessly.  The z4ai codec is applied to **each tensor independently** with
its per-dtype byte-plane + long-distance-matching pipeline, and the results are
packed into a self-describing ``ZSTN`` container - the codec's on-disk format
for a whole checkpoint.  It uses the tensor's declared dtype so
BF16/FP16/FP32 tensors take the model-weight fast path while everything else is
compressed opaquely.  The round-trip is **byte-identical** to the original file.

Why per-tensor (not whole-file Zstd)?  Because z4ai's exponent-plane separation
only helps when applied to homogeneous float data; mixing tensors of different
dtypes into one stream defeats it.  Per-tensor also means z4ai's long-distance
matching can dedup *within* a large tensor (tied embeddings, repeated blocks) -
exactly where z4ai beats ZipNN's 256 KiB chunking.

Random access - the container property that matters for distribution
---------------------------------------------------------------------
A ``ZSTN`` v3 container carries a **tensor index** and a fixed-size **footer**,
so a reader can locate and decompress **one tensor without touching the rest**
of the file (:class:`ZstnReader`).  This preserves the single most valuable
property of ``safetensors`` - per-tensor lazy loading / partial reads - which a
plain "compress the whole file" approach destroys.  Tied weights are stored once
and referenced, and the reader resolves the reference transparently.

No third-party ``safetensors`` package is required: the format is a tiny,
well-specified container that we parse directly.

safetensors v1 source format::

    8 bytes      little-endian u64 = header length N
    N bytes      UTF-8 JSON header: { name: {dtype, shape, data_offsets:[b,e]},
                 optional "__metadata__": {str: str} }
    rest         the data buffer; tensor `name` occupies buffer[b:e]

ZSTN v3 container layout::

    "ZSTN"                         4-byte magic
    u8  version (= 3)
    u8  contiguous flag (1 = per-tensor index present, 0 = single opaque frame)
    u64 hlen ; <hlen> bytes        original safetensors JSON header, verbatim
    u64 data_len                   length of the original data buffer
    -- if not contiguous --
        u64 frame_len ; <frame_len> bytes   one opaque z4ai frame of the buffer
    -- if contiguous --
        <frames section>           unique tensor frames, back to back
        <index section>            starts at index_offset (see footer):
            u32 n_tensors
            per tensor (emission order):
                u16 name_len ; <name>
                u64 dest_begin ; u64 dest_end      destination in the data buffer
                u8  kind (0 = FRAME, 1 = REF)
                if FRAME: u64 frame_offset ; u64 frame_len   (absolute in file)
                if REF:   u32 ref_index                      (earlier FRAME entry)
        <footer> (fixed 20 bytes, always last):
            u64 index_offset ; u64 index_len ; "ZSTN"

Public API
----------
* :func:`compress_bytes` / :func:`decompress_bytes` - in-memory, whole file.
* :func:`compress_file` / :func:`decompress_file` - on-disk, returns stats.
* :class:`ZstnReader` - lazy, random-access single-tensor reads from a file or
  buffer (``read_raw`` / ``read_numpy`` / ``names`` / ``header``).
"""
from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import BinaryIO, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from . import codec

_T = TypeVar("_T")
_R = TypeVar("_R")


def _resolve_workers(workers: Optional[int], n_tasks: int) -> int:
    """Number of Python threads to use for ``n_tasks`` independent codec calls.

    ``None`` (the default) auto-sizes to all cores, capped at the task count.
    Per-tensor compression/decompression is dominated by the GIL-releasing
    native zstd call, so fanning the *independent* tensors across a thread pool
    scales nearly linearly with cores - measured ~8x on a 200-tensor model -
    while leaving the on-disk frame byte-identical (tensors are still emitted in
    the same order; only the order they are *computed* changes).
    """
    if workers is None:
        workers = os.cpu_count() or 1
    return max(1, min(int(workers), max(1, n_tasks)))


def _map(fn: Callable[[_T], _R], items: List[_T], workers: int) -> List[_R]:
    """``[fn(x) for x in items]``, fanned across ``workers`` threads when it pays.

    Order-preserving and exception-propagating (``ThreadPoolExecutor.map``), so
    callers get exactly the serial result, just faster.  Falls back to a plain
    serial loop for a single worker or a single item (no pool overhead, and the
    exact prior behaviour for tiny models).
    """
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# Bytes sampled from each end + the middle to fingerprint a tensor for dedup.
_DEDUP_SAMPLE = 512


def _dedup_fingerprint(raw: bytes):
    """A cheap, collision-resistant key for finding byte-identical tensors.

    Cross-tensor dedup only needs to spot tied / duplicated weights, which are
    byte-exact.  Keying the ``seen`` map on the *whole* tensor makes Python hash
    every tensor's full bytes — a measured ~30% of whole-model compress on a
    dense checkpoint, where almost nothing is actually a duplicate.  A
    length-plus-sampled-head/middle/tail fingerprint separates distinct tensors
    at a few-KB cost instead; the rare fingerprint collision is resolved by a
    full byte compare before any reference is emitted (see :func:`compress_bytes`),
    so dedup stays exactly as correct as a full-bytes key.
    """
    n = len(raw)
    if n <= 3 * _DEDUP_SAMPLE:
        return raw  # small tensor: the full bytes is already a cheap key
    mv = memoryview(raw)
    s = _DEDUP_SAMPLE
    return (n, bytes(mv[:s]), bytes(mv[n // 2 : n // 2 + s]), bytes(mv[n - s :]))

# Container magic for a z4ai-compressed safetensors file (also the footer
# sentinel, so the trailer can be validated from the end of the stream).
MAGIC = b"ZSTN"
_VERSION = 3

# Fixed footer: u64 index_offset, u64 index_len, 4-byte trailing magic.
_FOOTER_FMT = "<QQ4s"
_FOOTER_LEN = struct.calcsize(_FOOTER_FMT)

# Per-tensor record kinds.
_K_FRAME = 0  # an inline z4ai-compressed frame of the tensor's bytes
_K_REF = 1    # a back-reference: this tensor's bytes equal an earlier tensor's

# safetensors dtype string -> (z4ai float dtype code or None, NumPy dtype str).
# Float types take z4ai's model-weight path; the rest are compressed opaquely
# (still lossless).  ``np`` is the NumPy dtype string for ``read_numpy`` or None
# when NumPy has no native equivalent (bf16, fp8) - those still ``read_raw``.
@dataclass(frozen=True)
class _DTypeInfo:
    zcode: Optional[str]
    np: Optional[str]


_ST_DTYPE: Dict[str, _DTypeInfo] = {
    "F64": _DTypeInfo("fp64", "<f8"),
    "F32": _DTypeInfo("fp32", "<f4"),
    "F16": _DTypeInfo("fp16", "<f2"),
    "BF16": _DTypeInfo("bf16", None),
    "F8_E4M3": _DTypeInfo(None, None),
    "F8_E5M2": _DTypeInfo(None, None),
    "I64": _DTypeInfo(None, "<i8"),
    "U64": _DTypeInfo(None, "<u8"),
    "I32": _DTypeInfo(None, "<i4"),
    "U32": _DTypeInfo(None, "<u4"),
    "I16": _DTypeInfo(None, "<i2"),
    "U16": _DTypeInfo(None, "<u2"),
    "I8": _DTypeInfo(None, "|i1"),
    "U8": _DTypeInfo(None, "|u1"),
    "BOOL": _DTypeInfo(None, "|b1"),
}


@dataclass(frozen=True)
class _TensorEntry:
    name: str
    dtype: str
    begin: int
    end: int


@dataclass(frozen=True)
class _IndexEntry:
    """A decoded index record; enough to locate one tensor's bytes."""

    name: str
    begin: int        # destination span in the original data buffer
    end: int
    kind: int         # _K_FRAME or _K_REF
    frame_offset: int  # absolute file offset of the frame (FRAME only)
    frame_len: int     # length of the frame (FRAME only)
    ref_index: int     # emission index of the referenced FRAME (REF only)


def _parse_header(blob: bytes) -> Tuple[int, dict, List[_TensorEntry]]:
    """Parse a safetensors buffer into (header_len, header_dict, entries).

    ``entries`` are sorted by their ``begin`` offset (data-buffer order).
    Raises ``ValueError`` on a malformed buffer.
    """
    if len(blob) < 8:
        raise ValueError("not a safetensors buffer: shorter than 8 bytes")
    (header_len,) = struct.unpack_from("<Q", blob, 0)
    if 8 + header_len > len(blob):
        raise ValueError("safetensors header length exceeds buffer size")
    try:
        header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid safetensors JSON header: {exc}") from exc

    entries: List[_TensorEntry] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            raise ValueError(f"tensor {name!r} missing data_offsets")
        begin, end = meta["data_offsets"]
        entries.append(_TensorEntry(name, meta.get("dtype", "U8"), int(begin), int(end)))
    entries.sort(key=lambda e: e.begin)
    return header_len, header, entries


def compress_bytes(
    blob: bytes,
    *,
    level: Optional[int] = None,
    threads: int = 0,
    workers: Optional[int] = None,
) -> bytes:
    """Compress a safetensors buffer; returns a self-describing ``ZSTN`` frame.

    The original JSON header is preserved verbatim so reconstruction is exactly
    byte-identical (including key order and whitespace).  Each tensor's bytes are
    compressed with :func:`z4ai.compress` using the tensor's dtype, and a tensor
    index + footer are appended so :class:`ZstnReader` can read one tensor
    without decompressing the others.

    If the tensors do not contiguously cover the data buffer (unexpected for
    valid safetensors), the whole data section is compressed opaquely as a
    single stream - still byte-exact, just without per-tensor random access.
    """
    header_len, _header, entries = _parse_header(blob)
    data_start = 8 + header_len
    data = blob[data_start:]
    data_len = len(data)

    contiguous = _is_contiguous(entries, data_len)
    header_bytes = blob[8 : 8 + header_len]

    out = bytearray()
    out += MAGIC
    out += struct.pack("<BB", _VERSION, 1 if contiguous else 0)
    out += struct.pack("<Q", len(header_bytes))
    out += header_bytes
    out += struct.pack("<Q", data_len)

    if not contiguous:
        frame = codec.compress(data, dtype=None, level=level, threads=threads)
        out += struct.pack("<Q", len(frame))
        out += frame
        return bytes(out)

    # Frames section: emit each unique tensor frame, recording where it landed.
    # Cross-tensor dedup: tied weights (e.g. embed_tokens == lm_head) are
    # byte-identical *separate* tensors.  Per-tensor compression alone cannot
    # exploit that, and ZipNN's 256 KiB chunking can't dedup megabyte-apart
    # copies either - so a back-reference is a clean, guaranteed win on the very
    # common tied-embedding case.  We key on exact content.
    #
    # Pass 1 (serial, cheap): resolve dedup against earlier tensors and collect
    # the unique tensors that need a full codec pass.  Pass 2 compresses those
    # unique tensors *in parallel* - each ``codec.compress`` is an independent,
    # GIL-releasing zstd call, so this scales nearly linearly with cores (~8x on
    # a real model) and is the dominant cost.  Pass 3 stitches the frames into
    # ``out`` in emission order, so the container is byte-identical to a fully
    # serial build - only the order frames are *computed* changes.
    records: List[Tuple[_TensorEntry, int, int, int, int]] = []
    # (entry, kind, frame_offset, frame_len, ref_index)
    # ``seen`` maps a cheap fingerprint -> the list of earlier unique tensors that
    # share it (a "collision bucket").  Identical tensors always share a
    # fingerprint; a fingerprint match is confirmed by a full byte compare before
    # we emit a reference, so this is exactly as correct as keying on full bytes
    # but never hashes a whole tensor unless it is tiny.
    seen: Dict[object, List[Tuple[int, bytes]]] = {}
    plan: List[Tuple[int, int]] = []  # per entry: (_K_FRAME, idx) | (_K_REF, ref_index)
    unique: List[Tuple[bytes, Optional[str]]] = []  # (raw, zcode) to compress, in idx order
    unique_idx: List[int] = []                       # the emission index of each unique tensor
    for idx, e in enumerate(entries):
        raw = data[e.begin : e.end]
        fp = _dedup_fingerprint(raw)
        ref = None
        bucket = seen.get(fp)
        if bucket is not None:
            for cand_idx, cand_raw in bucket:
                if cand_raw == raw:  # confirm byte-identity (fingerprints can collide)
                    ref = cand_idx
                    break
        if ref is not None:
            plan.append((_K_REF, ref))
        else:
            seen.setdefault(fp, []).append((idx, raw))
            info = _ST_DTYPE.get(e.dtype, _DTypeInfo(None, None))
            unique.append((raw, info.zcode))
            unique_idx.append(idx)
            plan.append((_K_FRAME, idx))

    n_workers = _resolve_workers(workers, len(unique))
    # Split the core budget across the ONE level of parallelism that pays.  The
    # outer fan-out already runs up to one tensor per core; if each inner
    # ``codec.compress`` then also let zstd spin up its own ``cpu_count`` worker
    # threads (the ``threads=0`` auto default), the process would oversubscribe to
    # ``n_workers * cpu_count`` threads and lose throughput to context-switching.
    # Give each concurrently-compressed tensor an equal slice of zstd's internal
    # threads so the total stays ~= cpu_count: many small tensors (n_workers high)
    # compress single-threaded and parallelise across tensors; a few large tensors
    # (n_workers low) still get multi-threaded zstd to fill the cores.  An explicit
    # non-zero ``threads`` is honoured verbatim.  Output is byte-identical either
    # way - only how the work is scheduled changes.
    inner_threads = threads
    if threads == 0 and n_workers > 1:
        inner_threads = max(1, (os.cpu_count() or 1) // n_workers)
    frames = _map(
        lambda rc: codec.compress(rc[0], dtype=rc[1], level=level, threads=inner_threads),
        unique,
        n_workers,
    )
    frame_by_idx: Dict[int, bytes] = dict(zip(unique_idx, frames))

    for idx, e in enumerate(entries):
        kind, ref = plan[idx]
        if kind == _K_FRAME:
            frame = frame_by_idx[idx]
            frame_offset = len(out)
            out += frame
            records.append((e, _K_FRAME, frame_offset, len(frame), 0))
        else:
            records.append((e, _K_REF, 0, 0, ref))

    # Index section.
    index_offset = len(out)
    out += struct.pack("<I", len(records))
    for e, kind, frame_offset, frame_len, ref_index in records:
        name_b = e.name.encode("utf-8")
        out += struct.pack("<H", len(name_b))
        out += name_b
        out += struct.pack("<QQ", e.begin, e.end)
        out += struct.pack("<B", kind)
        if kind == _K_FRAME:
            out += struct.pack("<QQ", frame_offset, frame_len)
        else:
            out += struct.pack("<I", ref_index)
    index_len = len(out) - index_offset

    # Footer (fixed size, always last).
    out += struct.pack(_FOOTER_FMT, index_offset, index_len, MAGIC)
    return bytes(out)


def decompress_bytes(frame: bytes, *, workers: Optional[int] = None) -> bytes:
    """Invert :func:`compress_bytes`; returns the byte-identical safetensors buffer."""
    if frame[:4] != MAGIC:
        raise ValueError("not a z4ai-safetensors (ZSTN) frame")
    off = 4
    version, contiguous = struct.unpack_from("<BB", frame, off)
    off += 2
    if version != _VERSION:
        raise ValueError(f"unsupported ZSTN version {version}")
    (hlen,) = struct.unpack_from("<Q", frame, off)
    off += 8
    header_bytes = frame[off : off + hlen]
    off += hlen
    (data_len,) = struct.unpack_from("<Q", frame, off)
    off += 8

    out = bytearray()
    out += struct.pack("<Q", hlen)
    out += header_bytes

    if not contiguous:
        (flen,) = struct.unpack_from("<Q", frame, off)
        off += 8
        data = codec.decompress(frame[off : off + flen])
        if len(data) != data_len:
            raise ValueError("data length mismatch on decompress")
        out += data
        return bytes(out)

    entries = _read_index(frame)
    data = bytearray(data_len)
    decoded: List[Optional[bytes]] = [None] * len(entries)

    # Decode every FRAME entry in parallel - each is an independent, GIL-releasing
    # zstd decode, our weakest axis vs ZipNN, so fanning them across cores is a
    # near-linear throughput win.  REF entries are pure back-references and cost
    # nothing, so they are resolved afterwards (every REF points to a FRAME, which
    # is now decoded - no forward-reference hazard).  The reconstructed buffer is
    # byte-identical to a serial decode.
    frame_positions = [i for i, e in enumerate(entries) if e.kind != _K_REF]
    n_workers = _resolve_workers(workers, len(frame_positions))
    frame_bytes = _map(
        lambda i: _decode_entry(frame, entries[i]), frame_positions, n_workers
    )
    for i, raw in zip(frame_positions, frame_bytes):
        decoded[i] = raw
    for i, e in enumerate(entries):
        raw = decoded[i] if e.kind != _K_REF else decoded[e.ref_index]
        if raw is None:  # forward ref should never happen, but stay honest
            raw = _decode_entry(frame, entries[e.ref_index])
            decoded[e.ref_index] = raw
        data[e.begin : e.end] = raw
    out += data
    return bytes(out)


def _decode_entry(frame: bytes, e: _IndexEntry) -> bytes:
    """Decompress a single FRAME entry's bytes from the container."""
    return codec.decompress(frame[e.frame_offset : e.frame_offset + e.frame_len])


def _read_index(frame: bytes) -> List[_IndexEntry]:
    """Parse the footer + index of a contiguous ``ZSTN`` buffer."""
    if len(frame) < _FOOTER_LEN:
        raise ValueError("ZSTN frame too short for a footer")
    index_offset, index_len, trailer = struct.unpack_from(
        _FOOTER_FMT, frame, len(frame) - _FOOTER_LEN
    )
    if trailer != MAGIC:
        raise ValueError("ZSTN footer sentinel missing - file truncated or not v3")
    return _parse_index(frame[index_offset : index_offset + index_len])


def _parse_index(buf: bytes) -> List[_IndexEntry]:
    off = 0
    (n,) = struct.unpack_from("<I", buf, off)
    off += 4
    entries: List[_IndexEntry] = []
    for _ in range(n):
        (nlen,) = struct.unpack_from("<H", buf, off)
        off += 2
        name = buf[off : off + nlen].decode("utf-8")
        off += nlen
        begin, end = struct.unpack_from("<QQ", buf, off)
        off += 16
        (kind,) = struct.unpack_from("<B", buf, off)
        off += 1
        if kind == _K_FRAME:
            frame_offset, frame_len = struct.unpack_from("<QQ", buf, off)
            off += 16
            entries.append(_IndexEntry(name, begin, end, kind, frame_offset, frame_len, 0))
        elif kind == _K_REF:
            (ref_index,) = struct.unpack_from("<I", buf, off)
            off += 4
            entries.append(_IndexEntry(name, begin, end, kind, 0, 0, ref_index))
        else:
            raise ValueError(f"unknown tensor record kind {kind}")
    return entries


def _is_contiguous(entries: List[_TensorEntry], data_len: int) -> bool:
    """True iff the tensors partition ``[0, data_len)`` with no gaps/overlaps."""
    cursor = 0
    for e in entries:
        if e.begin != cursor or e.end < e.begin:
            return False
        cursor = e.end
    return cursor == data_len


def compress_file(
    in_path: str,
    out_path: str,
    *,
    level: Optional[int] = None,
    threads: int = 0,
    workers: Optional[int] = None,
) -> dict:
    """Compress a ``.safetensors`` file to a ``ZSTN`` file. Returns stats."""
    with open(in_path, "rb") as f:
        blob = f.read()
    frame = compress_bytes(blob, level=level, threads=threads, workers=workers)
    with open(out_path, "wb") as f:
        f.write(frame)
    orig, comp = len(blob), len(frame)
    return {
        "original_bytes": orig,
        "compressed_bytes": comp,
        "ratio": orig / comp if comp else float("inf"),
        "saved_pct": (1 - comp / orig) * 100 if orig else 0.0,
    }


def decompress_file(
    in_path: str, out_path: str, *, workers: Optional[int] = None
) -> dict:
    """Decompress a ``ZSTN`` file back to a byte-identical ``.safetensors`` file."""
    with open(in_path, "rb") as f:
        frame = f.read()
    blob = decompress_bytes(frame, workers=workers)
    with open(out_path, "wb") as f:
        f.write(blob)
    return {"bytes": len(blob)}


class ZstnReader:
    """Random-access reader for a ``ZSTN`` v3 container.

    Reads only the header and tensor index up front; each :meth:`read_raw` /
    :meth:`read_numpy` seeks to a single tensor's frame and decompresses just
    that tensor - the rest of the file is never touched.  This is the lazy /
    partial-load path that makes ``ZSTN`` a drop-in for ``safetensors``'
    per-tensor access rather than a whole-file blob.

    Open from a path (recommended for large checkpoints - uses ``seek`` so only
    the needed frames are read) or from an in-memory buffer::

        with ZstnReader.open("model.zstn") as r:
            w = r.read_numpy("model.embed_tokens.weight")

        r = ZstnReader.from_bytes(frame)
    """

    def __init__(
        self,
        *,
        fh: Optional[BinaryIO] = None,
        buf: Optional[bytes] = None,
        owns_file: bool = False,
    ) -> None:
        if (fh is None) == (buf is None):
            raise ValueError("provide exactly one of fh / buf")
        self._fh = fh
        self._buf = buf
        self._owns_file = owns_file
        self._read_prefix()

    # -- construction --------------------------------------------------------
    @classmethod
    def open(cls, path: str) -> "ZstnReader":
        """Open a ``.zstn`` file for lazy, seek-based random access."""
        return cls(fh=open(path, "rb"), owns_file=True)

    @classmethod
    def from_bytes(cls, frame: Union[bytes, bytearray, memoryview]) -> "ZstnReader":
        """Open an in-memory ``ZSTN`` buffer."""
        return cls(buf=bytes(frame))

    # -- context manager -----------------------------------------------------
    def __enter__(self) -> "ZstnReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_file and self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- low-level byte access ----------------------------------------------
    def _size(self) -> int:
        if self._buf is not None:
            return len(self._buf)
        assert self._fh is not None
        self._fh.seek(0, 2)
        return self._fh.tell()

    def _read(self, offset: int, length: int) -> bytes:
        if self._buf is not None:
            return self._buf[offset : offset + length]
        assert self._fh is not None
        self._fh.seek(offset)
        return self._fh.read(length)

    def _read_prefix(self) -> None:
        magic_ver = self._read(0, 6)
        if magic_ver[:4] != MAGIC:
            raise ValueError("not a z4ai-safetensors (ZSTN) frame")
        version, contiguous = struct.unpack_from("<BB", magic_ver, 4)
        if version != _VERSION:
            raise ValueError(f"unsupported ZSTN version {version}")
        self._contiguous = bool(contiguous)
        (hlen,) = struct.unpack("<Q", self._read(6, 8))
        self._header_bytes = self._read(14, hlen)
        self._header = json.loads(self._header_bytes.decode("utf-8"))
        (self._data_len,) = struct.unpack("<Q", self._read(14 + hlen, 8))

        if not self._contiguous:
            raise ValueError(
                "this ZSTN file stores one opaque frame and has no tensor index; "
                "use z4ai.safetensors.decompress_bytes for whole-file decode"
            )

        size = self._size()
        idx_off, idx_len, trailer = struct.unpack(
            _FOOTER_FMT, self._read(size - _FOOTER_LEN, _FOOTER_LEN)
        )
        if trailer != MAGIC:
            raise ValueError("ZSTN footer sentinel missing - file truncated or not v3")
        entries = _parse_index(self._read(idx_off, idx_len))
        self._entries = entries
        self._by_name: Dict[str, int] = {e.name: i for i, e in enumerate(entries)}

    # -- public API ----------------------------------------------------------
    def names(self) -> List[str]:
        """Tensor names, in data-buffer order."""
        return [e.name for e in self._entries]

    def header(self) -> dict:
        """The original safetensors JSON header (dtype/shape/offsets, metadata)."""
        return self._header

    def metadata(self) -> Dict[str, str]:
        """The ``__metadata__`` sub-dict, or empty if absent."""
        return dict(self._header.get("__metadata__", {}))

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def _resolve(self, name: str) -> _IndexEntry:
        try:
            e = self._entries[self._by_name[name]]
        except KeyError:
            raise KeyError(f"no tensor named {name!r} in this ZSTN file") from None
        if e.kind == _K_REF:
            e = self._entries[e.ref_index]  # tied weight: read the original frame
        return e

    def read_raw(self, name: str) -> bytes:
        """Decompress and return the raw bytes of a single tensor.

        Only that tensor's frame is read from disk; tied-weight references are
        resolved to the original frame transparently.
        """
        e = self._resolve(name)
        frame = self._read(e.frame_offset, e.frame_len)
        return codec.decompress(frame)

    def frame_size(self, name: str) -> int:
        """Compressed (on-disk) size of a tensor's frame, in bytes."""
        return self._resolve(name).frame_len

    def read_numpy(self, name: str):
        """Return a tensor as a NumPy array with its declared dtype and shape.

        Raises ``TypeError`` for dtypes NumPy cannot represent natively (bf16,
        fp8) - use :meth:`read_raw` for those.
        """
        import numpy as np

        meta = self._header.get(name)
        if meta is None:
            raise KeyError(f"no tensor named {name!r} in this ZSTN file")
        info = _ST_DTYPE.get(meta["dtype"], _DTypeInfo(None, None))
        if info.np is None:
            raise TypeError(
                f"tensor {name!r} dtype {meta['dtype']} has no native NumPy dtype; "
                "use read_raw()"
            )
        raw = self.read_raw(name)
        return np.frombuffer(raw, dtype=np.dtype(info.np)).reshape(meta["shape"])


__all__ = [
    "MAGIC",
    "ZstnReader",
    "compress_bytes",
    "decompress_bytes",
    "compress_file",
    "decompress_file",
]
