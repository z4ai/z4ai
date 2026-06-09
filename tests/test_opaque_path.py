# SPDX-License-Identifier: Apache-2.0

"""Guards for the opaque (non-float / ``dtype=None``) compression path.

Non-float tensors — token-id buffers, attention masks, int/bool buffers — are
common in real checkpoints and take z4ai's opaque path.  That path used to
compress at level-1 only and could fall *below* plain zstd-3 (measured: a uint8
mask 762x vs zstd-3 1416x), violating the "never lose to plain Zstd" invariant.
``codec._opaque_primary_frame`` now keeps the best of {plain high-level,
whole-window+LDM, store}.  These tests lock that in.
"""

from __future__ import annotations

import numpy as np
import pytest
import zstandard as zstd

import z4ai
from z4ai import codec


def _zstd3(data: bytes) -> int:
    return len(zstd.ZstdCompressor(level=3).compress(data))


# On data that compresses thousands-fold, z4ai's small frame header is a
# measurable fraction of the output, so we allow a combined multiplicative +
# additive slack rather than demanding exact parity with bare zstd.
def _within(frame_len: int, zstd_len: int) -> bool:
    return frame_len <= 1.03 * zstd_len + 128


def _tiled_u8():
    return np.tile(np.arange(256, dtype=np.uint8), 200_000).tobytes()


def _int32_ramp():
    return np.arange(3_000_000, dtype=np.int32).tobytes()


def _triangular_mask():
    return np.tril(np.ones((4096, 4096), dtype=np.uint8)).tobytes()


def _repeated_block():
    return b"the quick brown fox " * 500_000


# Factories (not the bytes themselves) keep pytest's parametrize ids small.
@pytest.mark.parametrize(
    "make",
    [_tiled_u8, _int32_ramp, _triangular_mask, _repeated_block],
    ids=["tiled_u8", "int32_ramp", "triangular_mask", "repeated_block"],
)
def test_opaque_never_loses_to_zstd3(make):
    data = make()
    frame = z4ai.compress(data, dtype=None)
    assert z4ai.decompress(frame) == data, "not lossless"
    z3 = _zstd3(data)
    assert _within(len(frame), z3), (
        f"z4ai opaque {len(frame)} > zstd-3 {z3} (+slack) — "
        f"opaque path regressed below plain Zstd"
    )


def test_opaque_incompressible_never_expands():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, 4_000_000, dtype=np.uint8).tobytes()
    frame = z4ai.compress(noise, dtype=None)
    assert z4ai.decompress(frame) == noise
    assert len(frame) <= len(noise) + 256  # store fallback


def test_opaque_long_range_redundancy_caught():
    # A repeat whose period (4 MiB) exceeds zstd's default window: only the
    # whole-window+LDM candidate in the opaque best-of can dedup it.
    block = np.random.default_rng(1).integers(0, 256, 4 << 20, dtype=np.uint8).tobytes()
    data = block * 4
    frame = z4ai.compress(data, dtype=None)
    assert z4ai.decompress(frame) == data
    assert (
        len(data) / len(frame) > 3.0
    ), "opaque LDM candidate failed to dedup far repeats"


def test_opaque_escalate_roundtrips_and_helps_when_monotonic():
    # max-ratio mode (codec.escalate_level) raises the opaque build level. zstd is
    # famously NON-monotonic (a higher level can be worse on some structured
    # inputs), so we don't assert it beats the default on adversarial data — only
    # that it stays lossless, and that on ordinary text it does at least as well.
    mask = _triangular_mask()
    assert codec.decompress(codec.compress(mask, dtype=None, escalate_level=19)) == mask

    text = _repeated_block()
    default = codec.compress(text, dtype=None)
    maxr = codec.compress(text, dtype=None, escalate_level=19)
    assert codec.decompress(maxr) == text
    assert len(maxr) <= len(default) + 128
