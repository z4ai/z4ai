# SPDX-License-Identifier: Apache-2.0

"""Tests for the synthetic data generator (runnable without the core codec)."""
import numpy as np

from _data import bytes_per_element, f32_to_bf16_bytes, make_weights


def test_sizes():
    for dtype, bpe in [("bf16", 2), ("fp16", 2), ("fp32", 4), ("fp64", 8)]:
        data = make_weights(1000, dtype=dtype)
        assert len(data) == 1000 * bpe
        assert bytes_per_element(dtype) == bpe


def test_deterministic():
    assert make_weights(5000, dtype="bf16", seed=42) == make_weights(5000, dtype="bf16", seed=42)


def test_bf16_truncation_matches_high_bits():
    # For values without rounding carry, bf16 must equal the top 16 bits of fp32.
    f32 = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
    bf = np.frombuffer(f32_to_bf16_bytes(f32), dtype="<u2")
    top = (f32.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(bf, top)


def test_bf16_exponent_low_entropy():
    """The realistic generator should yield a concentrated exponent distribution."""
    u16 = np.frombuffer(make_weights(200_000, dtype="bf16", seed=0), dtype="<u2")
    exp = (u16 >> 7) & 0xFF  # 8 exponent bits of bf16
    # most mass should sit in a handful of exponent values
    counts = np.bincount(exp, minlength=256)
    top8_fraction = np.sort(counts)[-8:].sum() / counts.sum()
    assert top8_fraction > 0.6
