# SPDX-License-Identifier: Apache-2.0

"""Losslessness + API-contract tests for the z4ai codec.

These are written against API_CONTRACT.md and skip automatically until the
core `z4ai` package is importable, so the suite is green throughout parallel
development and turns into real coverage the moment core lands.
"""
import numpy as np
import pytest

from _data import make_weights  # from benchmarks/, on sys.path via conftest

FLOAT_DTYPES = ["bf16", "fp16", "fp32", "fp64"]


# --- core losslessness --------------------------------------------------------

@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_roundtrip_float_dtypes(z4ai, dtype):
    data = make_weights(200_000, dtype=dtype, seed=1)
    blob = z4ai.compress(data, dtype=dtype)
    assert isinstance(blob, (bytes, bytearray))
    assert bytes(z4ai.decompress(blob)) == data


def test_roundtrip_opaque_bytes(z4ai):
    data = b"the quick brown fox" * 1000
    assert bytes(z4ai.decompress(z4ai.compress(data))) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n_extra", [0, 1, 2, 3])
def test_roundtrip_non_multiple_lengths(z4ai, dtype, n_extra):
    """Lengths not divisible by the element size must still round-trip."""
    base = make_weights(50_000, dtype=dtype, seed=2)
    data = base + bytes(range(n_extra))
    assert bytes(z4ai.decompress(z4ai.compress(data, dtype=dtype))) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [None])
def test_roundtrip_empty(z4ai, dtype):
    assert bytes(z4ai.decompress(z4ai.compress(b"", dtype=dtype))) == b""


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_roundtrip_single_element(z4ai, dtype):
    data = make_weights(1, dtype=dtype, seed=3)
    assert bytes(z4ai.decompress(z4ai.compress(data, dtype=dtype))) == data


@pytest.mark.parametrize("pattern", [b"\x00", b"\xff", b"\xaa"])
def test_roundtrip_degenerate(z4ai, pattern):
    data = pattern * 100_000
    assert bytes(z4ai.decompress(z4ai.compress(data))) == data


def test_roundtrip_random_bytes(z4ai):
    rng = np.random.default_rng(9)
    data = rng.integers(0, 256, size=300_000, dtype=np.uint8).tobytes()
    assert bytes(z4ai.decompress(z4ai.compress(data))) == data


# --- frame / contract ---------------------------------------------------------

def test_frame_has_magic(z4ai):
    blob = z4ai.compress(make_weights(1000, dtype="bf16"), dtype="bf16")
    assert bytes(blob[:4]) == b"Z4AI"


@pytest.mark.parametrize("level", [1, 3, 9, 19, 22])
def test_levels_are_lossless(z4ai, level):
    data = make_weights(80_000, dtype="bf16", seed=4)
    assert bytes(z4ai.decompress(z4ai.compress(data, dtype="bf16", level=level))) == data


@pytest.mark.parametrize("threads", [0, 1, 2])
def test_threads_are_lossless(z4ai, threads):
    data = make_weights(80_000, dtype="bf16", seed=5)
    out = z4ai.decompress(z4ai.compress(data, dtype="bf16", threads=threads))
    assert bytes(out) == data


def test_compresses_realistic_weights(z4ai):
    """Sanity: on realistic bf16 weights we should beat 1.0x comfortably."""
    data = make_weights(1_000_000, dtype="bf16", seed=6)
    blob = z4ai.compress(data, dtype="bf16")
    ratio = len(data) / len(blob)
    assert ratio > 1.2, f"expected >1.2x on realistic bf16, got {ratio:.3f}x"


# --- ndarray helpers (optional in v1) -----------------------------------------

def test_ndarray_helpers_if_present(z4ai):
    if not hasattr(z4ai, "compress_ndarray"):
        pytest.skip("ndarray helpers not implemented")
    arr = np.random.default_rng(7).standard_normal((128, 256)).astype(np.float32)
    blob = z4ai.compress_ndarray(arr)
    out = z4ai.decompress_ndarray(blob)
    assert out.dtype == arr.dtype and out.shape == arr.shape
    assert np.array_equal(out, arr)
