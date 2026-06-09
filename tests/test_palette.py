# SPDX-License-Identifier: Apache-2.0

"""Tests for the palette (dictionary) transform and its AUTO best-of wiring.

The palette frame targets *dequantised low-precision* tensors (INT4/INT8/FP8
stored in a wide float container), which carry only a small set of distinct
values.  These tests assert (1) exact lossless round-trip across widths, value
counts, shapes and partial tails, (2) that the cheap distinct-count gate fires on
quantised data and skips genuine full-entropy data, and (3) that the codec keeps
the palette frame only when it actually wins and never regresses a native tensor.
"""

from __future__ import annotations

import numpy as np
import pytest

import z4ai
from z4ai import codec, palette


def _bf16(f32: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(f32.astype(np.float32))
    return (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def _quantize(f32: np.ndarray, levels: int) -> np.ndarray:
    s = np.abs(f32).max() / levels
    return (np.round(f32 / s) * s).astype(np.float32)


# --------------------------------------------------------------------------- #
# Module-level round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("n", [0, 1, 2, 7, 257, 10_000])
@pytest.mark.parametrize("k", [1, 2, 16, 256, 300, 5000])
def test_palette_roundtrip(width, n, k):
    rng = np.random.default_rng(n * 131 + k * 7 + width)
    dt = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    hi = min(k * 17 + 1, np.iinfo(dt).max)
    if n == 0:
        raw = b""
    else:
        vals = rng.choice(np.arange(1, hi, dtype=dt), size=min(k, hi - 1), replace=False)
        raw = rng.choice(vals, size=n).astype(dt).tobytes()
    blob = palette.compress(raw, width)
    assert palette.peek_is_palette(blob)
    assert palette.decompress(blob) == raw


@pytest.mark.parametrize("width", [2, 4, 8])
def test_palette_roundtrip_with_tail(width):
    rng = np.random.default_rng(width)
    dt = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    body = rng.integers(0, 50, size=5000, dtype=dt).tobytes()
    tail = b"\x01\x02\x03\x04\x05\x06\x07"[: width - 1]  # not a whole element
    raw = body + tail
    blob = palette.compress(raw, width)
    assert palette.decompress(blob) == raw


def test_palette_constant_tensor():
    raw = np.full(10_000, 1234, dtype=np.uint16).tobytes()
    blob = palette.compress(raw, 2)
    assert palette.decompress(blob) == raw
    # one distinct value -> the frame is dominated by metadata, far below input
    assert len(blob) < len(raw) // 4


# --------------------------------------------------------------------------- #
# Gate behaviour
# --------------------------------------------------------------------------- #
def test_gate_fires_on_quantized_skips_native():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(2_000_000).astype(np.float32)
    native = _bf16(w)
    int8 = _bf16(_quantize(w, 127))
    assert palette.should_use(int8, 2) is True
    assert palette.should_use(native, 2) is False


def test_gate_rejects_unsupported_width_and_empty():
    assert palette.should_use(b"", 2) is False
    assert palette.supported_width(1) is False
    assert palette.should_use(b"\x00" * 100, 1) is False


# --------------------------------------------------------------------------- #
# End-to-end via the public codec (best-of selection + ndarray restore)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("levels", [7, 127])
def test_codec_picks_palette_on_quantized(dtype, levels):
    rng = np.random.default_rng(levels)
    w = rng.standard_normal(1_500_000).astype(np.float32)
    q = _quantize(w, levels)
    if dtype == "bf16":
        raw = _bf16(q)
    elif dtype == "fp16":
        raw = q.astype(np.float16).tobytes()
    else:
        raw = q.astype(np.float32).tobytes()
    blob = z4ai.compress(raw, dtype=dtype)
    assert bytes(z4ai.decompress(blob)) == raw  # lossless
    assert palette.peek_is_palette(blob)        # palette won the best-of
    assert len(blob) < len(raw)


def test_codec_does_not_regress_native_bf16():
    rng = np.random.default_rng(1)
    raw = _bf16(rng.standard_normal(2_000_000))
    blob = z4ai.compress(raw, dtype="bf16")
    assert bytes(z4ai.decompress(blob)) == raw
    assert not palette.peek_is_palette(blob)  # native must not use the palette


def test_ndarray_roundtrip_quantized():
    rng = np.random.default_rng(2)
    arr = _quantize(rng.standard_normal((1500, 1000)), 127).astype(np.float32)
    blob = z4ai.compress_ndarray(arr)
    back = z4ai.decompress_ndarray(blob)
    assert np.array_equal(arr, back)
    assert back.dtype == arr.dtype and back.shape == arr.shape
    assert palette.peek_is_palette(blob)


def test_palette_beats_byteplane_on_int4():
    rng = np.random.default_rng(3)
    raw = _bf16(_quantize(rng.standard_normal(2_000_000), 7))
    pal = z4ai.compress(raw, dtype="bf16")
    plain = codec.compress(raw, dtype="bf16", auto=False)  # forces the transform path
    assert len(pal) < len(plain)
    assert bytes(z4ai.decompress(pal)) == raw
