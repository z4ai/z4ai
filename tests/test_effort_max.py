# SPDX-License-Identifier: Apache-2.0

"""Guards for the public ``effort="max"`` high-ratio tier and its chunk-parallel
brotli backend.

``effort="max"`` routes ``z4ai.compress`` to the AUTO codec (field split +
context-modeling backend) and emits ``b"ZARA"`` frames; ``z4ai.decompress`` must
transparently accept them.  The backend slices large streams into independent
1 MiB blocks compressed in parallel — these tests exercise multi-block streams
(so the block framing, parallel encode, and parallel decode are all covered) and
assert byte-exact round-trips, including the ratio win that is the whole point.
"""
import numpy as np
import pytest

import z4ai
from _data import f32_to_bf16_bytes  # benchmarks/ on sys.path (conftest)

brotli = pytest.importorskip("brotli", reason="effort=max needs the brotli backend")

DTYPES = ["bf16", "fp16", "fp32"]


def _structured_bf16(n=3_000_000, seed=7):
    """Realistic-ish weights: peaked exponent + some duplicate rows, so the
    context backend has structure to exploit (and several brotli blocks)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n // 4).astype(np.float32) * 0.05
    return f32_to_bf16_bytes(np.tile(base, 4))


@pytest.mark.parametrize("dtype", DTYPES)
def test_effort_max_bytes_roundtrip(dtype):
    data = f32_to_bf16_bytes(np.random.default_rng(1).standard_normal(500_000).astype(np.float32))
    blob = z4ai.compress(data, dtype=dtype, effort="max")
    # best-of(AUTO, default) -> a ZARA frame when brotli wins, else a Z4AI frame.
    assert bytes(blob[:4]) in (b"ZARA", b"Z4AI", b"ZSP1")
    assert bytes(z4ai.decompress(blob)) == data


@pytest.mark.parametrize("n_extra", [0, 1, 2, 3])
def test_effort_max_non_multiple_lengths(n_extra):
    base = f32_to_bf16_bytes(np.random.default_rng(2).standard_normal(40_000).astype(np.float32))
    data = base + bytes(range(n_extra))
    assert bytes(z4ai.decompress(z4ai.compress(data, dtype="bf16", effort="max"))) == data


def test_effort_max_empty_and_tiny():
    for d in (b"", b"\x01", b"\x01\x02\x03"):
        assert bytes(z4ai.decompress(z4ai.compress(d, dtype="bf16", effort="max"))) == d


def test_effort_max_multiblock_roundtrip():
    """A >2 MiB stream forces the chunk-parallel brotli path (multiple blocks)."""
    data = _structured_bf16()
    blob = z4ai.compress(data, dtype="bf16", effort="max")
    assert bytes(z4ai.decompress(blob)) == data


def test_effort_max_ndarray_roundtrip():
    arr = (np.random.default_rng(3).standard_normal((400, 500)).astype(np.float32))
    blob = z4ai.compress_ndarray(arr, effort="max")
    assert bytes(blob[:4]) in (b"ZARA", b"Z4AI", b"ZSP1")
    out = z4ai.decompress_ndarray(blob)
    assert np.array_equal(arr, out)


def _dense_bf16(n=3_000_000, seed=8):
    """Dense weights with NO long-range repeats — the case where the context
    backend (not LDM) is the lever and chunk-parallel brotli == whole brotli."""
    rng = np.random.default_rng(seed)
    return f32_to_bf16_bytes(rng.standard_normal(n).astype(np.float32) * 0.05)


def test_effort_max_never_worse_than_default():
    """``effort="max"`` keeps the smaller of {AUTO, fast-default}, so it must
    NEVER produce a larger frame than the default — including on duplicate-heavy
    data where the default's whole-buffer LDM beats chunk-parallel brotli."""
    for data in (_dense_bf16(), _structured_bf16()):
        default = len(z4ai.compress(data, dtype="bf16"))
        mx = len(z4ai.compress(data, dtype="bf16", effort="max"))
        assert mx <= default, (mx, default)


def test_effort_max_wins_on_dense():
    """On dense weights (no duplicates), the context backend is the lever and
    max must be STRICTLY smaller than the default — proof brotli is engaged."""
    data = _dense_bf16()
    default = len(z4ai.compress(data, dtype="bf16"))
    mx = len(z4ai.compress(data, dtype="bf16", effort="max"))
    assert mx < default, (mx, default)


def test_chunk_parallel_brotli_matches_whole():
    """On dense data (brotli's win is local), the chunk-parallel backend must
    round-trip and stay within a hair of whole-buffer brotli."""
    from z4ai import auto

    data = _dense_bf16()
    payload = auto._brotli_blocks_compress(data, quality=11)
    assert auto._brotli_blocks_decompress(payload, len(data)) == data
    whole = brotli.compress(data, quality=11)
    assert len(payload) <= 1.02 * len(whole), (len(payload), len(whole))
