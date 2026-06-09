# SPDX-License-Identifier: Apache-2.0

"""Tests for the sparse zero-aware transform (:mod:`z4ai.sparse`).

Covers (1) standalone losslessness across widths / sparsity / edge cases,
(2) NumPy metadata round-trip, (3) automatic selection + delegation through the
public ``z4ai.compress``/``decompress`` API, and (4) the ratio win that is the
whole point of the mode.
"""
import numpy as np
import pytest

from _data import f32_to_bf16_bytes  # benchmarks/ on sys.path via conftest

WIDTH_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _sparse_bytes(width, n, frac, seed=0):
    """Random fixed-width buffer with ``frac`` of elements set to exact zero."""
    rng = np.random.default_rng(seed)
    dt = WIDTH_DTYPE[width]
    hi = 256 if width == 1 else 2 ** 31  # keep values in range, plenty of spread
    u = rng.integers(1, hi, size=n, dtype=np.uint64).astype(dt)
    u[rng.random(n) < frac] = 0
    return u.tobytes()


# --- standalone losslessness --------------------------------------------------

@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("frac", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_sparse_roundtrip(z4ai, width, frac):
    from z4ai import sparse

    data = _sparse_bytes(width, 50_000, frac, seed=width * 7 + int(frac * 10))
    blob = sparse.compress(data, width)
    assert sparse.peek_is_sparse(blob)
    assert sparse.decompress(blob) == data


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_sparse_edge_cases(z4ai, width):
    from z4ai import sparse

    cases = [
        b"",                              # empty
        bytes(width),                     # single zero element
        _sparse_bytes(width, 1, 0.0),     # single nonzero element
        bytes(width * 1000),              # all zeros
        b"\x01\x02\x03\x04\x05",          # length not a multiple of width (tail)
    ]
    for data in cases:
        blob = sparse.compress(data, width)
        assert sparse.decompress(blob) == data, (width, len(data))


def test_sparse_numpy_metadata(z4ai):
    from z4ai import sparse

    arr = np.zeros((32, 48), np.float32)
    arr[::5] = 1.5
    blob = sparse.compress(
        arr.tobytes(), 4, is_numpy=True, np_dtype=arr.dtype.str, shape=arr.shape
    )
    dtype_str, shape = sparse.read_numpy_metadata(blob)
    assert dtype_str == arr.dtype.str and shape == arr.shape
    assert sparse.decompress(blob) == arr.tobytes()


def test_sparse_no_numpy_metadata_returns_none(z4ai):
    from z4ai import sparse

    blob = sparse.compress(_sparse_bytes(2, 100, 0.5), 2)
    assert sparse.read_numpy_metadata(blob) is None


def test_unsupported_width_raises(z4ai):
    from z4ai import sparse

    assert not sparse.supported_width(3)
    with pytest.raises(ValueError):
        sparse.compress(b"abcabc", 3)


# --- automatic selection through the public API -------------------------------

def test_public_api_picks_sparse_and_roundtrips(z4ai):
    from z4ai import sparse

    rng = np.random.default_rng(1)
    f = (rng.standard_normal(400_000).astype(np.float32) * 0.02)
    f[rng.random(400_000) < 0.9] = 0.0
    data = f32_to_bf16_bytes(f)

    blob = z4ai.compress(data, dtype="bf16")
    assert sparse.peek_is_sparse(blob), "AUTO should choose the sparse frame here"
    assert bytes(z4ai.decompress(blob)) == data


def test_public_api_dense_stays_non_sparse(z4ai):
    """Dense weights must NOT be encoded as a sparse frame."""
    from z4ai import sparse

    rng = np.random.default_rng(2)
    data = f32_to_bf16_bytes(rng.standard_normal(200_000).astype(np.float32) * 0.02)
    blob = z4ai.compress(data, dtype="bf16")
    assert not sparse.peek_is_sparse(blob)
    assert bytes(blob[:4]) == b"Z4AI"
    assert bytes(z4ai.decompress(blob)) == data


def test_sparse_ndarray_roundtrip(z4ai):
    """A sparse array round-trips dtype/shape even when the sparse frame wins."""
    rng = np.random.default_rng(3)
    arr = np.zeros((500, 500), np.float32)
    arr[::4] = rng.standard_normal(arr[::4].shape).astype(np.float32)
    blob = z4ai.compress_array(arr)
    out = z4ai.decompress_array(blob)
    assert out.dtype == arr.dtype and out.shape == arr.shape
    assert np.array_equal(out, arr)


@pytest.mark.parametrize("frac", [0.5, 0.9, 0.95])
def test_sparse_beats_dense_path(z4ai, frac):
    """The sparse transform must shrink sparse data more than the dense path."""
    from z4ai import sparse

    rng = np.random.default_rng(int(frac * 100))
    f = (rng.standard_normal(1_000_000).astype(np.float32) * 0.02)
    f[rng.random(1_000_000) < frac] = 0.0
    data = f32_to_bf16_bytes(f)

    sparse_frame = sparse.compress(data, 2, level=1)
    dense_frame = z4ai.compress(data, dtype="bf16", auto=False)
    assert len(sparse_frame) < len(dense_frame)
    # And the public AUTO path is at least as good as the explicit sparse frame.
    assert len(z4ai.compress(data, dtype="bf16")) <= len(sparse_frame)
