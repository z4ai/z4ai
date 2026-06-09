# SPDX-License-Identifier: Apache-2.0

"""Regression guard for the long-distance-matching (LDM) best-of win.

This is the headline result in ``DECISIONS.md``: on real checkpoints
(``prajjwal1/bert-tiny`` fp32) z4ai's **default** settings reach ~2.24x vs
ZipNN's 1.20x because a whole-buffer Zstd pass with LDM deduplicates the
repeated multi-element runs (tied embeddings, duplicated/low-rank layers) that
real models are full of and that ZipNN's independent 256 KiB chunks cannot see.

That win has silently regressed at least once: when the opaque/byte-plane LDM
best-of was disabled "for speed", real-model ratio collapsed from ~2.24x to the
bit-field/byte-plane floor of ~1.19x (a *loss* to ZipNN) -- and the entire test
suite still passed, because the existing structured-data tests use extreme
*tiling* (>10x) that survives even a degraded path. This test reproduces the
realistic **moderate-redundancy** regime (block-level duplication, like tied
layers) where the LDM lever is the whole story, so a future regression fails
loudly here instead of shipping.

See ``DECISIONS.md`` -> "The one number that matters".
"""

import numpy as np
import pytest

import z4ai

# The bit-field/byte-plane-only floor on fp32 weights is ~1.17-1.19x (exponent
# decorrelation only). Anything well above that is necessarily the LDM win. The
# block-duplicated buffer below compresses to >3x with LDM, so 1.8x is a
# comfortable trip-wire that the ~1.19x regression cannot pass.
_LDM_FLOOR = 1.8


def _tied_layer_weights(
    seed: int = 0,
    n_unique_blocks: int = 3,
    n_total_blocks: int = 12,
    block_elems: int = 262_144,
):
    """fp32 bytes whose redundancy is *far-apart repeated runs*, like tied layers.

    A pool of a few distinct **large** (~1 MiB) random blocks is sampled with
    replacement and concatenated, so exact block repeats land *megabytes* apart
    -- beyond a generic compressor's sliding window. Only a whole-buffer pass
    with long-distance matching deduplicates them; plain zstd-3 sees ~1.08x here
    (verified), so this data genuinely isolates the LDM lever. (A small-block
    variant fits inside zstd's window and would NOT detect an LDM regression.)
    """
    rng = np.random.default_rng(seed)
    pool = (rng.standard_normal((n_unique_blocks, block_elems)) * 0.05).astype(
        np.float32
    )
    order = rng.integers(0, n_unique_blocks, size=n_total_blocks)
    return pool[order].reshape(-1).tobytes()


def test_ldm_win_is_lossless_and_large():
    """Default compress must dedup repeated runs (LDM) and stay byte-exact."""
    raw = _tied_layer_weights()
    blob = z4ai.compress(raw, dtype="fp32")  # DEFAULT settings only
    back = z4ai.decompress(blob)
    assert bytes(back) == raw, "LDM path must remain bit-exact"

    ratio = len(raw) / len(blob)
    assert ratio > _LDM_FLOOR, (
        f"real-model LDM win regressed: got {ratio:.3f}x, expected > {_LDM_FLOOR}x. "
        "The opaque/byte-plane LDM best-of has likely been disabled -- see "
        "DECISIONS.md (this is the 2.24x -> 1.19x regression)."
    )


def test_ldm_beats_plain_zstd_on_repeated_runs():
    """LDM best-of must beat generic zstd-3 *by a large margin* on far-apart
    repeats -- a self-scaling guard. Plain zstd-3's bounded window misses the
    megabyte-distant repeats (~1.6x here); z4ai's whole-buffer LDM gets >4x. If
    the LDM path regresses, z4ai collapses toward the zstd-3 number and this
    2x-margin assertion fails.
    """
    zstd = pytest.importorskip("zstandard")
    raw = _tied_layer_weights()
    z4ai_size = len(z4ai.compress(raw, dtype="fp32"))
    zstd_size = len(zstd.ZstdCompressor(level=3).compress(raw))
    assert z4ai_size * 2 < zstd_size, (
        f"LDM win regressed: z4ai={z4ai_size} must be >2x smaller than "
        f"plain zstd-3={zstd_size} on megabyte-distant repeats (see DECISIONS.md)"
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_ldm_win_across_float_dtypes(dtype):
    """The win is not fp32-specific: bf16/fp16 tied-layer weights win too."""
    raw = _tied_layer_weights(seed=1)
    if dtype != "fp32":
        # reinterpret the same redundant structure at 2-byte width
        a = np.frombuffer(raw, dtype=np.float32)
        if dtype == "fp16":
            raw = a.astype(np.float16).tobytes()
        else:  # bf16: truncate top 16 bits (round-to-nearest-even)
            u = a.view(np.uint32)
            raw = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)).tobytes()
    blob = z4ai.compress(raw, dtype=dtype)
    assert bytes(z4ai.decompress(blob)) == raw
    ratio = len(raw) / len(blob)
    assert ratio > _LDM_FLOOR, f"{dtype}: LDM win regressed ({ratio:.3f}x)"
