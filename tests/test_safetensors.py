# SPDX-License-Identifier: Apache-2.0

"""Round-trip + ratio tests for the dependency-free safetensors adapter."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from z4ai import safetensors as zst


# safetensors numpy dtype -> declared string.
_NP_TO_ST = {
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "uint8": "U8",
}


def _build_safetensors(tensors: dict, metadata: dict | None = None) -> bytes:
    """Serialize ``{name: np.ndarray}`` into a valid safetensors buffer.

    bf16 tensors are passed as ``("BF16", uint16_array)`` tuples since NumPy has
    no native bf16.
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    buffers = []
    offset = 0
    for name, value in tensors.items():
        if isinstance(value, tuple):  # ("BF16", uint16 array)
            st_dtype, arr = value
            shape = list(arr.shape)
        else:
            arr = np.ascontiguousarray(value)
            st_dtype = _NP_TO_ST[arr.dtype.name]
            shape = list(arr.shape)
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        buffers.append(raw)
        offset += len(raw)
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + b"".join(buffers)


def _weights(n, scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32) * scale)


def _bf16(f32: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> 16).astype(np.uint16)


def test_roundtrip_mixed_dtypes():
    blob = _build_safetensors(
        {
            "embed": _weights(4096).reshape(64, 64),
            "ln.weight": _weights(64).astype(np.float16),
            "ids": np.arange(128, dtype=np.int64),
            "mask": np.ones(256, dtype=np.uint8),
            "bf": ("BF16", _bf16(_weights(2048))),
        },
        metadata={"format": "pt", "producer": "z4ai-test"},
    )
    frame = zst.compress_bytes(blob)
    assert frame[:4] == zst.MAGIC
    assert zst.decompress_bytes(frame) == blob  # byte-identical


def test_roundtrip_empty_metadata_only():
    blob = _build_safetensors({"w": _weights(1000)})
    assert zst.decompress_bytes(zst.compress_bytes(blob)) == blob


@pytest.mark.parametrize("level", [None, 1, 9, 19])
def test_roundtrip_levels(level):
    blob = _build_safetensors({"w": _weights(8000).reshape(80, 100)})
    assert zst.decompress_bytes(zst.compress_bytes(blob, level=level)) == blob


def test_shrinks_low_entropy_weights():
    # Realistic small-scale Gaussian fp32 weights compress (low-entropy exponent).
    blob = _build_safetensors({f"layer{i}.w": _weights(20000, seed=i) for i in range(4)})
    frame = zst.compress_bytes(blob)
    assert len(frame) < len(blob)
    assert zst.decompress_bytes(frame) == blob


def test_structured_beats_opaque_on_duplicated_tensor():
    # A tensor that repeats a block many times (tied/duplicated weights) should
    # compress dramatically thanks to per-tensor long-distance matching.
    base = _weights(50_000, seed=1)
    dup = np.tile(base, 20)  # 1M elems, repeats every 50k
    blob = _build_safetensors({"tied": dup})
    frame = zst.compress_bytes(frame_in := blob)
    ratio = len(blob) / len(frame)
    assert zst.decompress_bytes(frame) == blob
    assert ratio > 5.0, f"expected big win on duplicated tensor, got {ratio:.2f}x"


def test_file_roundtrip(tmp_path):
    blob = _build_safetensors(
        {"a": _weights(3000), "b": np.arange(500, dtype=np.int32)}
    )
    src = tmp_path / "model.safetensors"
    comp = tmp_path / "model.zstn"
    restored = tmp_path / "model.restored.safetensors"
    src.write_bytes(blob)

    stats = zst.compress_file(str(src), str(comp))
    assert stats["compressed_bytes"] == comp.stat().st_size
    assert stats["ratio"] > 1.0

    zst.decompress_file(str(comp), str(restored))
    assert restored.read_bytes() == blob


def test_tied_embedding_dedup():
    # embed_tokens and lm_head share byte-identical weights (tied) - extremely
    # common in LLMs. Cross-tensor dedup should make adding the second copy
    # nearly free, so the file with the tie compresses far better than if the
    # two large tensors were distinct.
    emb = _weights(200_000, seed=3)
    tied = _build_safetensors(
        {"model.embed_tokens.weight": emb, "lm_head.weight": emb}
    )
    distinct = _build_safetensors(
        {"model.embed_tokens.weight": emb, "lm_head.weight": _weights(200_000, seed=4)}
    )
    tied_frame = zst.compress_bytes(tied)
    distinct_frame = zst.compress_bytes(distinct)
    assert zst.decompress_bytes(tied_frame) == tied  # still byte-exact
    # The tied file's second tensor is deduped to a tiny reference, so its frame
    # is markedly smaller than the distinct-weights file.
    assert len(tied_frame) < 0.7 * len(distinct_frame)


def test_dedup_fingerprint_collision_is_not_deduped():
    # Cross-tensor dedup keys on a cheap length+sampled-head/middle/tail
    # fingerprint (not the whole tensor) and confirms a match with a full byte
    # compare.  Craft two tensors that SHARE the fingerprint (identical length and
    # identical sampled regions) but differ in an unsampled byte: they must stay
    # distinct, never falsely deduped to a reference (which would corrupt output).
    n = 50_000
    a = np.zeros(n, dtype=np.uint8)
    a[: 512] = 7      # head sample region
    a[n // 2 : n // 2 + 512] = 9   # middle sample region
    a[n - 512 :] = 11             # tail sample region
    b = a.copy()
    # Differ in a region NOT covered by the head/middle/tail samples.
    a[2000] = 1
    b[2000] = 2
    blob = _build_safetensors({"x": a, "y": b})
    frame = zst.compress_bytes(blob)
    assert zst.decompress_bytes(frame) == blob  # byte-exact: not falsely deduped


def test_rejects_non_zstn():
    with pytest.raises(ValueError):
        zst.decompress_bytes(b"NOPE" + b"\x00" * 32)


def test_rejects_malformed_safetensors():
    with pytest.raises(ValueError):
        zst.compress_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00not-json!")


# --- random-access reader (ZSTN v3) --------------------------------------


def test_reader_single_tensor_matches_full_decode():
    # A single tensor read through the lazy reader must equal the bytes that a
    # full decompress would place at that tensor's span.
    blob = _build_safetensors(
        {
            "a": _weights(3000).reshape(30, 100),
            "b": np.arange(500, dtype=np.int32),
            "c": _weights(7000, seed=2),
        }
    )
    frame = zst.compress_bytes(blob)
    full = zst.decompress_bytes(frame)
    hlen = struct.unpack_from("<Q", full, 0)[0]
    header = json.loads(full[8 : 8 + hlen].decode())
    data = full[8 + hlen :]

    r = zst.ZstnReader.from_bytes(frame)
    assert set(r.names()) == {"a", "b", "c"}
    for name in ("a", "b", "c"):
        b, e = header[name]["data_offsets"]
        assert r.read_raw(name) == data[b:e]


def test_reader_read_numpy_roundtrips_dtype_and_shape():
    a = _weights(2048).reshape(32, 64)
    ids = np.arange(300, dtype=np.int32)
    blob = _build_safetensors({"a": a, "ids": ids})
    r = zst.ZstnReader.from_bytes(zst.compress_bytes(blob))

    got_a = r.read_numpy("a")
    assert got_a.dtype == np.float32 and got_a.shape == (32, 64)
    assert np.array_equal(got_a, a)

    got_ids = r.read_numpy("ids")
    assert got_ids.dtype == np.int32 and np.array_equal(got_ids, ids)


def test_reader_resolves_tied_weight_reference():
    # lm_head is tied to embed_tokens: stored once, referenced. Reading either
    # name must return the identical bytes, transparently following the ref.
    emb = _weights(40_000, seed=7)
    blob = _build_safetensors(
        {"model.embed_tokens.weight": emb, "lm_head.weight": emb}
    )
    r = zst.ZstnReader.from_bytes(zst.compress_bytes(blob))
    raw_emb = r.read_raw("model.embed_tokens.weight")
    raw_head = r.read_raw("lm_head.weight")
    assert raw_emb == raw_head == np.ascontiguousarray(emb).tobytes()
    # The tied tensor carries no frame of its own (it is a reference).
    assert r.frame_size("lm_head.weight") == r.frame_size("model.embed_tokens.weight")


def test_reader_touches_only_the_requested_frame(tmp_path):
    # The point of the index: reading one tensor must not read the bytes of the
    # others. We prove it by counting bytes read from a tracking file wrapper.
    big = _weights(500_000, seed=1)        # large frame we will NOT read
    small = _weights(200, seed=2)          # small frame we WILL read
    blob = _build_safetensors({"big": big, "small": small})
    path = tmp_path / "m.zstn"
    zst.compress_file(  # write a real file
        str(_write(tmp_path / "m.safetensors", blob)), str(path)
    )

    reads = []
    real_open = open

    class _CountingFile:
        def __init__(self, f):
            self._f = f

        def seek(self, *a):
            return self._f.seek(*a)

        def tell(self):
            return self._f.tell()

        def read(self, n=-1):
            data = self._f.read(n)
            reads.append(len(data))
            return data

        def close(self):
            return self._f.close()

    with real_open(str(path), "rb") as raw:
        r = zst.ZstnReader(fh=_CountingFile(raw))
        reads.clear()
        out = r.read_raw("small")
    assert out == np.ascontiguousarray(small).tobytes()
    # We read only the small frame, never the ~2 MB "big" frame.
    big_frame_floor = big.nbytes // 2
    assert max(reads) < big_frame_floor, f"read a frame too large: {max(reads)}"


def test_reader_open_file_lazy(tmp_path):
    blob = _build_safetensors({"w": _weights(5000), "v": _weights(5000, seed=9)})
    src = _write(tmp_path / "m.safetensors", blob)
    comp = tmp_path / "m.zstn"
    zst.compress_file(str(src), str(comp))
    with zst.ZstnReader.open(str(comp)) as r:
        assert "w" in r and "v" in r
        assert r.read_numpy("w").shape == (5000,)


def test_reader_missing_tensor_raises():
    blob = _build_safetensors({"w": _weights(100)})
    r = zst.ZstnReader.from_bytes(zst.compress_bytes(blob))
    with pytest.raises(KeyError):
        r.read_raw("nope")


def _write(path, blob: bytes):
    path.write_bytes(blob)
    return path


# --------------------------------------------------------------------------- #
# Parallel (cross-tensor) compression/decompression must be byte-identical to
# the serial path.  ``compress_bytes`` fans the independent per-tensor codec
# calls across a thread pool by default (``workers=None`` -> all cores) for a
# large whole-model throughput win; the ON-DISK FRAME and the reconstructed
# buffer must not depend on how many workers ran.  These guard that invariant,
# including the tied-weight dedup path (a REF must still point at the right
# earlier FRAME regardless of compute order).
# --------------------------------------------------------------------------- #
def _mixed_model() -> bytes:
    tied = _bf16(_weights(4000, seed=1))
    return _build_safetensors(
        {
            "embed": ("BF16", tied),
            "block.0.w": _weights(3000, seed=2).astype(np.float32),
            "block.0.h": _weights(2500, seed=3).astype(np.float16),
            "block.1.w": _weights(3500, seed=4).astype(np.float32),
            "lm_head": ("BF16", tied),  # tied to embed -> exercises REF dedup
            "block.1.h": _weights(1500, seed=5).astype(np.float16),
        }
    )


@pytest.mark.parametrize("workers", [1, 2, 4, None])
def test_parallel_compress_is_byte_identical_to_serial(workers):
    blob = _mixed_model()
    serial = zst.compress_bytes(blob, workers=1)
    out = zst.compress_bytes(blob, workers=workers)
    assert out == serial, "parallel compress frame differs from serial"


@pytest.mark.parametrize("comp_workers", [1, None])
@pytest.mark.parametrize("dec_workers", [1, 2, None])
def test_parallel_roundtrip_is_lossless(comp_workers, dec_workers):
    blob = _mixed_model()
    frame = zst.compress_bytes(blob, workers=comp_workers)
    assert zst.decompress_bytes(frame, workers=dec_workers) == blob
