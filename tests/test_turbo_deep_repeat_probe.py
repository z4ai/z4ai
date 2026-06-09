# SPDX-License-Identifier: Apache-2.0

"""Regression guard for the turbo per-plane store-raw probe.

History: the probe used to compress only the *first* 128 KiB of a plane to
decide whether to store it verbatim. A plane whose head is high-entropy but
which has long-range repeats deeper in (structured / duplicated weights - the
case z4ai markets as its big win) was therefore wrongly stored raw, discarding
the entire plane-transpose ratio win. The bug was masked whenever the opaque
whole-buffer fallback happened to win; this test constructs a case where it does
NOT, so the probe regression fails loudly.
"""
import numpy as np
import pytest

from z4ai import turbo


def _deep_repeat_plus_noise():
    """width-2 buffer: plane0 = a block repeated 12x (head looks random, deep
    repeats); plane1 = pure noise. Interleaved so an opaque pass cannot win."""
    rng = np.random.default_rng(0)
    block = rng.integers(0, 256, 256 * 1024, dtype=np.uint8)
    plane0 = np.tile(block, 12)
    plane1 = rng.integers(0, 256, plane0.size, dtype=np.uint8)
    inter = np.empty(plane0.size * 2, dtype=np.uint8)
    inter[0::2] = plane0
    inter[1::2] = plane1
    return inter.tobytes()


def test_turbo_catches_deep_repeats_in_plane():
    """The plane with deep long-range repeats must be compressed, not stored.

    Ideal ratio is ~1.85 (plane0 ~12x, plane1 incompressible). The old
    head-only probe stored plane0 raw and produced ratio ~1.0.
    """
    data = _deep_repeat_plus_noise()
    blob = turbo.compress(data, width=2, level=9)
    assert turbo.decompress(blob) == data            # lossless first
    ratio = len(data) / len(blob)
    assert ratio > 1.5, f"deep-repeat plane was not compressed (ratio {ratio:.2f})"


def test_turbo_still_stores_pure_noise():
    """The probe must still cheaply store genuinely incompressible planes raw
    (never expanding the input)."""
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, 6_000_000, dtype=np.uint8).tobytes()
    blob = turbo.compress(noise, width=2, level=9)
    assert turbo.decompress(blob) == noise
    # never expands: frame is the data plus a small constant header overhead.
    assert len(blob) <= len(noise) + 256
