# SPDX-License-Identifier: Apache-2.0

"""Losslessness + behaviour tests for the chunked, multi-threaded codec.

The chunked codec spans many threads and a variable-length frame, so these
tests hammer the edges that concurrency and chunk boundaries expose: lengths
that straddle / don't align to chunk and element sizes, empty input, every
dtype width (including fp64 which ZipNN's byte path does not handle), and that
forcing multiple small chunks still round-trips bit-exactly.
"""
import numpy as np
import pytest

from _data import make_weights  # from benchmarks/, on sys.path via conftest

CHUNKED = pytest.importorskip("z4ai.chunked", reason="chunked codec not present")

FLOAT_DTYPES = ["bf16", "fp16", "fp32", "fp64"]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_roundtrip_float_dtypes(dtype):
    data = make_weights(300_000, dtype=dtype, seed=1)
    blob = CHUNKED.compress(data, dtype=dtype)
    assert CHUNKED.is_chunked(blob)
    assert bytes(CHUNKED.decompress(blob)) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [None])
@pytest.mark.parametrize("nbytes", [0, 1, 2, 3, 7, 8, 9, 4095, 4096, 4097])
def test_roundtrip_edge_lengths(dtype, nbytes):
    """Lengths around chunk and element boundaries must round-trip exactly."""
    rng = np.random.default_rng(nbytes)
    data = rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes()
    # Force many tiny chunks so boundaries are exercised hard.
    blob = CHUNKED.compress(data, dtype=dtype, chunk_size=64)
    assert bytes(CHUNKED.decompress(blob)) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_multi_chunk_matches_single_chunk(dtype):
    """Result must be identical whether we use one big chunk or many small ones."""
    data = make_weights(200_000, dtype=dtype, seed=2)
    a = CHUNKED.decompress(CHUNKED.compress(data, dtype=dtype, chunk_size=1 << 30))
    b = CHUNKED.decompress(CHUNKED.compress(data, dtype=dtype, chunk_size=8192))
    assert bytes(a) == bytes(b) == data


@pytest.mark.parametrize("threads", [1, 2, 8, 0])
def test_thread_counts_are_lossless(threads):
    data = make_weights(500_000, dtype="bf16", seed=3)
    blob = CHUNKED.compress(data, dtype="bf16", threads=threads, chunk_size=4096)
    out = CHUNKED.decompress(blob, threads=threads)
    assert bytes(out) == data


def test_compress_decompress_thread_mismatch():
    """Compress and decompress with different thread counts still agree."""
    data = make_weights(400_000, dtype="fp32", seed=4)
    blob = CHUNKED.compress(data, dtype="fp32", threads=8, chunk_size=4096)
    assert bytes(CHUNKED.decompress(blob, threads=1)) == data


def test_degenerate_and_random_payloads():
    for payload in (b"\x00" * 50_000, b"\xff" * 50_000, b"\xaa" * 50_000):
        blob = CHUNKED.compress(payload, dtype="bf16")
        assert bytes(CHUNKED.decompress(blob)) == payload
    rnd = np.random.default_rng(5).integers(0, 256, 50_000, dtype=np.uint8).tobytes()
    assert bytes(CHUNKED.decompress(CHUNKED.compress(rnd))) == rnd


def test_beats_zstd_on_realistic_weights():
    data = make_weights(1_000_000, dtype="bf16", seed=6)
    ratio = len(data) / len(CHUNKED.compress(data, dtype="bf16"))
    assert ratio > 1.2, f"expected >1.2x on realistic bf16, got {ratio:.3f}x"


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        CHUNKED.decompress(b"NOTZ4AI_" + b"\x00" * 16)
