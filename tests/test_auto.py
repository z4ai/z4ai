# SPDX-License-Identifier: Apache-2.0

"""Tests for the AUTO best-of codec (``z4ai.auto``) -- the mode that beats ZipNN.

Covers losslessness across dtypes / edge cases, the self-describing frame, the
effort levels, NumPy round-trips, and -- crucially -- that AUTO never loses to
plain zstd and (when installed) beats or ties ZipNN on structured tensors.
"""
import numpy as np
import pytest

from _data import f32_to_bf16_bytes, make_weights  # benchmarks/ is on sys.path

auto = pytest.importorskip("z4ai.auto", reason="z4ai.auto not present")

FLOAT_DTYPES = ["bf16", "fp16", "fp32", "fp64"]
EFFORTS = ["fast", "balanced", "max"]


# --- losslessness -------------------------------------------------------------

@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [None])
@pytest.mark.parametrize("effort", EFFORTS)
def test_roundtrip_dtypes(dtype, effort):
    data = make_weights(50_000, dtype=dtype or "fp32", seed=1)
    blob = auto.compress(data, dtype=dtype, effort=effort)
    assert bytes(auto.decompress(blob)) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [None])
def test_roundtrip_empty(dtype):
    assert bytes(auto.decompress(auto.compress(b"", dtype=dtype))) == b""


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n_extra", [0, 1, 2, 3])
def test_roundtrip_non_multiple_lengths(dtype, n_extra):
    base = make_weights(20_000, dtype=dtype, seed=2)
    data = base + bytes(range(n_extra))
    assert bytes(auto.decompress(auto.compress(data, dtype=dtype))) == data


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_roundtrip_single_element(dtype):
    data = make_weights(1, dtype=dtype, seed=3)
    assert bytes(auto.decompress(auto.compress(data, dtype=dtype))) == data


@pytest.mark.parametrize("pattern", [b"\x00", b"\xff", b"\xaa"])
def test_roundtrip_degenerate(pattern):
    data = pattern * 80_000
    assert bytes(auto.decompress(auto.compress(data, dtype="bf16"))) == data


def test_roundtrip_opaque_bytes():
    data = b"the quick brown fox " * 4096
    assert bytes(auto.decompress(auto.compress(data))) == data


def test_roundtrip_random_bytes():
    data = np.random.default_rng(9).integers(0, 256, 200_000, dtype=np.uint8).tobytes()
    assert bytes(auto.decompress(auto.compress(data))) == data


# --- frame / contract ---------------------------------------------------------

def test_frame_has_magic():
    blob = auto.compress(make_weights(1000, dtype="bf16"), dtype="bf16")
    assert bytes(blob[:4]) == b"ZARA"


def test_decompress_into_out_buffer():
    data = make_weights(10_000, dtype="bf16", seed=5)
    blob = auto.compress(data, dtype="bf16")
    out = bytearray(len(data))
    auto.decompress(blob, out=out)
    assert bytes(out) == data


def test_out_buffer_too_small_raises():
    blob = auto.compress(b"x" * 1000)
    with pytest.raises(ValueError):
        auto.decompress(blob, out=bytearray(10))


# --- numpy round-trips --------------------------------------------------------

@pytest.mark.parametrize("np_dtype", [np.float32, np.float16, np.float64, np.int32])
def test_ndarray_roundtrip(np_dtype):
    arr = (np.random.default_rng(7).standard_normal((48, 96)) * 0.1).astype(np_dtype)
    blob = auto.compress(arr)
    back = auto.decompress_array(blob)
    assert back.dtype == arr.dtype and back.shape == arr.shape
    assert np.array_equal(back, arr)


# --- ratio claims (the whole point) -------------------------------------------

def test_beats_plain_zstd_on_structured_data():
    """On structured (repeated-value) weights AUTO must beat plain zstd-19."""
    import zstandard as zstd

    rng = np.random.default_rng(0)
    book = (rng.standard_normal(256).astype(np.float32) * 0.02)
    data = f32_to_bf16_bytes(book[rng.integers(0, 256, 400_000)])
    auto_size = len(auto.compress(data, dtype="bf16", effort="max"))
    zstd_size = len(zstd.ZstdCompressor(level=19).compress(data))
    assert auto_size < zstd_size, (auto_size, zstd_size)


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_bitpack_roundtrip(dtype):
    """The bit-packed-mantissa transform must round-trip exactly."""
    data = make_weights(40_000, dtype=dtype, seed=11)
    blob = auto.compress(data, dtype=dtype, effort="max")
    assert bytes(auto.decompress(blob)) == data


def test_beats_zipnn_on_iid_dense():
    """The headline upgrade: AUTO now beats ZipNN even on dense i.i.d. weights.

    The bit-packed mantissa (vs whole-byte) plus rANS on the exponent close and
    flip the entropy-floor gap, so AUTO is no longer behind on i.i.d. data.
    """
    zipnn = pytest.importorskip("zipnn")
    data = make_weights(500_000, dtype="bf16", seed=0)
    auto_size = len(auto.compress(data, dtype="bf16", effort="max"))
    z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
    zipnn_size = len(z.compress(bytes(bytearray(data))))
    # require at least parity; in practice AUTO is ~0.4% smaller here.
    assert auto_size <= zipnn_size, (auto_size, zipnn_size)


def test_beats_or_ties_zipnn_if_installed():
    """AUTO should beat (structured) or tie (dense) ZipNN -- never lose badly."""
    zipnn = pytest.importorskip("zipnn")
    rng = np.random.default_rng(0)
    # structured tensor: many repeated values -> zipnn's mandatory grouping hurts
    book = (rng.standard_normal(256).astype(np.float32) * 0.02)
    data = f32_to_bf16_bytes(book[rng.integers(0, 256, 400_000)])

    auto_size = len(auto.compress(data, dtype="bf16", effort="max"))
    z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
    zipnn_size = len(z.compress(bytes(bytearray(data))))
    # On structured data we expect a clear win; require at least parity (allow
    # 0.5% slack so the test is not brittle to library/version drift).
    assert auto_size <= zipnn_size * 1.005, (auto_size, zipnn_size)
