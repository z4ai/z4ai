# SPDX-License-Identifier: Apache-2.0

"""Whole-checkpoint delta compression, aligned tensor-by-tensor by name.

:mod:`z4ai.delta` deltas two *byte buffers* that are already positionally
aligned.  Real ``.safetensors`` checkpoints are not: a fine-tune may add, drop,
rename or re-order tensors relative to its base, which shifts every following
byte and makes a raw whole-buffer XOR align nothing (high entropy, no win).  This
module deltas at the *tensor* granularity instead -- it matches each tensor in the
target to the base tensor of the **same name and shape**, so unchanged and
slightly-changed tensors delta cleanly regardless of how the surrounding layout
moved.

This is the lossless lever that matters most in production.  The bulk of stored
checkpoints are *derived*: a fine-tune of a base model, the next training step, an
RLHF / DPO iteration, an EMA copy, or a plain re-upload.  Their weights are mostly
bit-identical to a reference, so the redundancy lives *between files* and is
invisible to any single-file codec (z4ai included) or to ZipNN.  Measured on
synthetic fine-tunes: a checkpoint with ~2 % of weights changed stores ~25x
smaller as a name-aligned delta than compressed from scratch; ~10 % changed,
~6x; a re-upload, unboundedly (every tensor is a zero-cost COPY).  See the
low-precision lossless literature (arXiv:2508.19263) for the same effect on real
BF16 training-checkpoint sequences.

Each target tensor is stored as one of:

* ``COPY``  -- byte-identical to the base tensor of the same name; nothing stored.
* ``DELTA`` -- a :mod:`z4ai.delta` frame of ``target_tensor XOR base_tensor``
  (kept only when it beats a from-scratch frame; ``delta.compress`` itself picks
  the smaller of XOR-delta vs full, so this never loses to plain compression).
* ``FULL``  -- a standalone :func:`z4ai.compress` frame, for tensors with no
  shape/name match in the base (new or reshaped tensors).

Reconstruction is bit-exact: the target's original safetensors JSON header is
stored verbatim, every tensor is restored to its exact byte span, and a digest of
the base data buffer is recorded so decoding against the wrong base fails loudly
instead of silently corrupting weights.  The base checkpoint is supplied by the
caller at both ends (you already have it on disk when you write the derived one).

Example::

    >>> from z4ai import model_delta
    >>> blob = model_delta.compress(finetuned_safetensors, base_safetensors)
    >>> restored = model_delta.decompress(blob, base_safetensors)
    >>> assert restored == finetuned_safetensors   # byte-identical
"""

from __future__ import annotations

import hashlib
import struct
from typing import Dict, Optional

from . import codec, delta
from .safetensors import _parse_header

MAGIC = b"Z4DM"
_VERSION = 1

# Per-tensor record kinds.
_K_FULL = 0  # standalone z4ai frame (no usable base match)
_K_DELTA = 1  # z4ai.delta frame of (target XOR base) for the same-named tensor
_K_COPY = 2  # byte-identical to the base tensor; payload omitted

_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


def _as_bytes(buf) -> bytes:
    if isinstance(buf, bytes):
        return buf
    if isinstance(buf, (bytearray, memoryview)):
        return bytes(buf)
    raise TypeError(f"expected a bytes-like buffer, got {type(buf).__name__}")


def _digest(buf: bytes) -> bytes:
    """8-byte digest of a buffer, to detect a mismatched base at decode time."""
    return hashlib.blake2b(buf, digest_size=8).digest()


def peek_is_model_delta(blob) -> bool:
    """True if ``blob`` is a z4ai whole-checkpoint delta frame."""
    b = bytes(blob[:4]) if not isinstance(blob, bytes) else blob[:4]
    return b == MAGIC


def compress(
    target,
    base,
    *,
    level: Optional[int] = None,
    threads: int = 0,
) -> bytes:
    """Compress safetensors ``target`` as a name-aligned delta against ``base``.

    Both arguments are raw ``.safetensors`` buffers (bytes-like).  Returns a
    self-describing ``Z4DM`` frame that :func:`decompress` turns back into the
    byte-identical ``target`` given the same ``base``.  The result is never larger
    than compressing ``target`` standalone: every tensor independently keeps the
    smaller of its delta and from-scratch encoding.
    """
    target = _as_bytes(target)
    base = _as_bytes(base)

    t_hlen, _t_header, t_entries = _parse_header(target)
    t_data_start = 8 + t_hlen
    t_data = target[t_data_start:]
    t_header_bytes = target[8 : 8 + t_hlen]

    b_hlen, _b_header, b_entries = _parse_header(base)
    b_data_start = 8 + b_hlen
    b_data = base[b_data_start:]
    # Map base tensor name -> its raw bytes (one slice per tensor).
    base_by_name: Dict[str, bytes] = {
        e.name: b_data[e.begin : e.end] for e in b_entries
    }
    base_dtype: Dict[str, str] = {e.name: e.dtype for e in b_entries}

    out = bytearray()
    out += MAGIC
    out += struct.pack("<B", _VERSION)
    out += _U64.pack(len(t_header_bytes))
    out += t_header_bytes
    out += _U64.pack(len(t_data))
    out += _digest(b_data)  # 8 bytes: guards against decoding with the wrong base
    out += _U32.pack(len(t_entries))

    from .safetensors import _ST_DTYPE  # local import: avoid a cycle at module load

    for e in t_entries:
        tb = t_data[e.begin : e.end]
        info = _ST_DTYPE.get(e.dtype)
        zcode = info.zcode if info is not None else None

        ref = base_by_name.get(e.name)
        aligned = (
            ref is not None
            and len(ref) == len(tb)
            and base_dtype.get(e.name) == e.dtype
        )

        name_b = e.name.encode("utf-8")
        if aligned and ref == tb:
            kind, payload = _K_COPY, b""
        elif aligned:
            # delta.compress internally keeps min(full, XOR-delta), so this branch
            # can only match or beat a standalone frame.
            payload = delta.compress(tb, ref, dtype=zcode, level=level, threads=threads)
            kind = _K_DELTA
        else:
            payload = codec.compress(
                tb, dtype=zcode, level=level, **_plane_kwargs(threads)
            )
            kind = _K_FULL

        out += _U16.pack(len(name_b))
        out += name_b
        out += _U64.pack(e.begin)
        out += _U64.pack(e.end)
        out += struct.pack("<B", kind)
        if kind != _K_COPY:
            out += _U64.pack(len(payload))
            out += payload

    return bytes(out)


def decompress(blob, base) -> bytes:
    """Reconstruct the byte-identical safetensors ``target`` from a ``Z4DM`` frame
    and the same ``base`` buffer used to build it (see :func:`compress`)."""
    blob = _as_bytes(blob)
    base = _as_bytes(base)
    if blob[:4] != MAGIC:
        raise ValueError("not a z4ai model-delta (Z4DM) frame")
    off = 4
    (version,) = struct.unpack_from("<B", blob, off)
    off += 1
    if version != _VERSION:
        raise ValueError(f"unsupported Z4DM version {version}")

    (hlen,) = _U64.unpack_from(blob, off)
    off += _U64.size
    t_header_bytes = blob[off : off + hlen]
    off += hlen
    (data_len,) = _U64.unpack_from(blob, off)
    off += _U64.size
    want_digest = blob[off : off + 8]
    off += 8
    (n_tensors,) = _U32.unpack_from(blob, off)
    off += _U32.size

    _b_hlen, _b_header, b_entries = _parse_header(base)
    b_data = base[8 + _b_hlen :]
    if _digest(b_data) != want_digest:
        raise ValueError(
            "base checkpoint does not match the one used to build this delta "
            "(digest mismatch); decoding would corrupt the weights"
        )
    base_by_name: Dict[str, bytes] = {
        e.name: b_data[e.begin : e.end] for e in b_entries
    }

    data = bytearray(data_len)
    for _ in range(n_tensors):
        (name_len,) = _U16.unpack_from(blob, off)
        off += _U16.size
        name = blob[off : off + name_len].decode("utf-8")
        off += name_len
        (begin,) = _U64.unpack_from(blob, off)
        off += _U64.size
        (end,) = _U64.unpack_from(blob, off)
        off += _U64.size
        (kind,) = struct.unpack_from("<B", blob, off)
        off += 1

        if kind == _K_COPY:
            tb = base_by_name[name]
        else:
            (frame_len,) = _U64.unpack_from(blob, off)
            off += _U64.size
            frame = blob[off : off + frame_len]
            off += frame_len
            if kind == _K_DELTA:
                tb = delta.decompress(frame, base_by_name[name])
            else:  # _K_FULL
                tb = codec.decompress(frame)
        data[begin:end] = tb

    return struct.pack("<Q", hlen) + bytes(t_header_bytes) + bytes(data)


def _plane_kwargs(threads: int) -> dict:
    """Mirror the public ``threads`` -> codec knob mapping (see z4ai.__init__)."""
    import os

    if threads == 1:
        return {"plane_workers": 1}
    if threads and threads > 1:
        return {"plane_workers": threads}
    return {"plane_workers": max(1, os.cpu_count() or 1)}


def compress_file(
    target_path: str,
    base_path: str,
    out_path: str,
    *,
    level: Optional[int] = None,
    threads: int = 0,
) -> dict:
    """Delta-compress the safetensors at ``target_path`` against ``base_path``.

    Writes a ``Z4DM`` frame to ``out_path`` and returns a small stats dict
    (``original``, ``stored``, ``ratio``).  Reconstruct with
    :func:`decompress_file` and the same base file.
    """
    with open(target_path, "rb") as f:
        target = f.read()
    with open(base_path, "rb") as f:
        base = f.read()
    blob = compress(target, base, level=level, threads=threads)
    with open(out_path, "wb") as f:
        f.write(blob)
    return {
        "original": len(target),
        "stored": len(blob),
        "ratio": len(target) / max(1, len(blob)),
    }


def decompress_file(delta_path: str, base_path: str, out_path: str) -> dict:
    """Reconstruct the byte-identical safetensors target from a ``Z4DM`` file and
    its base file (inverse of :func:`compress_file`)."""
    with open(delta_path, "rb") as f:
        blob = f.read()
    with open(base_path, "rb") as f:
        base = f.read()
    target = decompress(blob, base)
    with open(out_path, "wb") as f:
        f.write(target)
    return {"stored": len(blob), "restored": len(target)}
