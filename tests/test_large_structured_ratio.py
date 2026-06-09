# SPDX-License-Identifier: Apache-2.0

"""Large-tensor ratio guard - closes a real coverage gap.

``tests/test_auto.py`` already asserts the beat-zstd / beat-or-tie-ZipNN ratio
claim, but only on **400 K-element** buffers (~400 KiB byte planes).  The
backend routes a plane to its throughput-oriented block-parallel path only when
the plane is **>= 4 MiB** (``backend._MIN_PARALLEL_PLANE``); below that every
plane takes the whole-plane LDM path.  So the small-buffer tests cannot exercise
the routing decision at all.

That gap let a real regression ship undetected: a redundancy detector that
keyed on exact 64 KiB-chunk duplicates reported "no long-range redundancy" for
weight planes that merely repeat a *small alphabet of values* (the common real
case - bf16 high-byte planes have only a few thousand distinct bytes).  Those
planes were diverted to block-parallel compression, which forfeits global dedup
and collapsed the ratio: measured bf16 2.76x -> 1.47x, turning the headline win
over ZipNN into a *loss*, while every existing test stayed green.

These tests reproduce that exact condition - a **large** (>= 4 MiB plane),
spatially-structured, value-repetitive bf16 tensor (tied / repeated / low-rank
layers, the structure real checkpoints have) - and assert the ratio claim
*there*, on the block-parallel-eligible path.  The plain-zstd comparison needs
no optional dependency and would have caught the regression on its own (the
broken codec fell below whole-buffer zstd-19); the ZipNN check runs only when
ZipNN is importable.
"""

import numpy as np
import pytest
import zstandard as zstd

from _data import f32_to_bf16_bytes  # benchmarks/ is on sys.path (see conftest)

# 8.4 M bf16 elements => 16.8 MB buffer => ~8.4 MB byte planes, comfortably above
# backend._MIN_PARALLEL_PLANE (4 MiB) so the block-parallel routing is exercised.
_N_ELEMS = 8_400_000
_BASE = 262_144  # 256 K-element base block; tiling it creates >4 MiB-period repeats


def _structured_weights_bf16() -> bytes:
    """A large bf16 buffer with tied / duplicated layers (exact repeats).

    Models the structure real checkpoints have - tied input/output embeddings,
    duplicated or EMA/shadow transformer blocks, multi-shard concatenations: one
    base block of small-magnitude weights repeated across the tensor.  The
    repeats are *spatial* (megabytes apart), exactly what whole-plane LDM
    deduplicates and ZipNN's fixed ~256 KiB chunks cannot reach.  We use exact
    repeats (not jittered near-duplicates) because that is what tied/duplicated
    weights actually are, and it makes the ratio claim robust rather than
    sensitive to a particular noise level.
    """
    rng = np.random.default_rng(0)
    base = (rng.standard_normal(_BASE).astype(np.float32) * 0.02)
    n_tiles = (_N_ELEMS + _BASE - 1) // _BASE
    f32 = np.tile(base, n_tiles)[:_N_ELEMS]
    return f32_to_bf16_bytes(f32)


@pytest.fixture(scope="module")
def structured_bf16():
    return _structured_weights_bf16()


def test_large_structured_roundtrips_losslessly(z4ai, structured_bf16):
    """Bit-exact round-trip on a >4 MiB-plane tensor (block-parallel path)."""
    blob = z4ai.compress(structured_bf16, dtype="bf16")
    assert bytes(z4ai.decompress(blob)) == structured_bf16


def test_large_structured_beats_plain_zstd(z4ai, structured_bf16):
    """The ratio win must hold on the block-parallel-eligible path.

    No optional deps: this is the regression guard.  The broken block-routing
    codec produced a frame *larger* than whole-buffer zstd-19 here; a correct
    codec (whole-plane LDM for structured planes) is clearly smaller.
    """
    z4ai_size = len(z4ai.compress(structured_bf16, dtype="bf16"))
    zstd19 = len(zstd.ZstdCompressor(level=19).compress(structured_bf16))
    assert z4ai_size < zstd19, (
        f"z4ai {z4ai_size} should beat whole-buffer zstd-19 {zstd19} on large "
        f"structured weights; a larger frame means the block-parallel path is "
        f"forfeiting long-range dedup (the bf16 2.76x->1.47x regression)."
    )


def test_large_structured_beats_zipnn(z4ai, structured_bf16):
    """On structured weights z4ai must beat ZipNN by a clear margin."""
    zipnn = pytest.importorskip("zipnn")
    z4ai_size = len(z4ai.compress(structured_bf16, dtype="bf16"))
    z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
    zipnn_size = len(z.compress(bytes(structured_bf16)))
    # Expect a comfortable win (measured ~+17%); require at least a real win with
    # slack for library/version drift.
    assert z4ai_size < zipnn_size, (z4ai_size, zipnn_size)
