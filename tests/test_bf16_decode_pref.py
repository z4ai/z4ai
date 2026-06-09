# SPDX-License-Identifier: Apache-2.0

"""Guard the bf16 decode-aware candidate preference.

For bf16, the byte-plane frame decodes ~5x faster than the bit-field frame at a
near-identical ratio, so :func:`z4ai.codec.compress` prefers byte-plane when it
is within ``_BYTEPLANE_DECODE_TOLERANCE`` of the smallest candidate.  These tests
pin that behaviour (and its losslessness) so a future refactor cannot silently
regress bf16 decode speed back to the slow bit-field path.
"""

from __future__ import annotations

import numpy as np
import pytest

import z4ai
from z4ai import codec
from z4ai import format as zfmt


def _bf16_bytes(f32: np.ndarray) -> bytes:
    return (f32.view(np.uint32) >> 16).astype("<u2").tobytes()


def _transform(frame: bytes) -> int:
    return zfmt.deserialize(frame).transform


@pytest.mark.parametrize("mixture", [True, False])
def test_bf16_dense_transform_choice(mixture: bool) -> None:
    """Dense bf16 must pick whichever frame the decode-aware best-of selects.

    The selection rule is unchanged: the byte-plane frame is preferred when it is
    within ``_BYTEPLANE_DECODE_TOLERANCE`` of the smallest candidate (it decodes
    several× faster).  What changed is the *ratio gap*: now that the order-0 rANS
    entropy coder runs BY DEFAULT, the bit-field split — which isolates a clean
    exponent byte AND a clean 7-bit mantissa byte that rANS codes to the entropy
    floor — pulls meaningfully (>1%) ahead of the byte-plane frame (whose middle
    byte mixes exponent and mantissa bits).  So pure-dense bf16 is now outside the
    decode tolerance and is kept as the smaller BIT-FIELD frame (the ratio that
    beats ZipNN), while a wider-scale mixture stays a near-tie and keeps the
    faster-decoding byte-plane frame.  Either way the frame must be lossless.
    """
    rng = np.random.default_rng(0)
    n = 8_000_000
    if mixture:
        scales = np.array([0.006, 0.02, 0.05, 0.15], dtype=np.float32)
        which = rng.integers(0, len(scales), size=n)
        f32 = rng.standard_normal(n).astype(np.float32) * scales[which]
    else:
        f32 = rng.standard_normal(n).astype(np.float32) * np.float32(0.02)
    raw = _bf16_bytes(f32)

    frame = codec.compress(raw, dtype="bf16")
    assert z4ai.decompress(frame) == raw  # lossless
    # Whatever is chosen, it must be the smallest candidate (the best-of invariant)
    # and never larger than the byte-plane-only or bit-field-only frame.
    bitfield_only = codec.compress(raw, dtype="bf16", auto=False)
    assert len(frame) <= len(bitfield_only) * codec._BYTEPLANE_DECODE_TOLERANCE
    expected = zfmt.TRANSFORM_BYTEPLANE if mixture else zfmt.TRANSFORM_BITFIELD
    assert _transform(frame) == expected


def test_bf16_decode_preference_costs_little_ratio() -> None:
    """The byte-plane choice must stay within tolerance of the bit-field frame."""
    rng = np.random.default_rng(1)
    f32 = rng.standard_normal(8_000_000).astype(np.float32) * np.float32(0.03)
    raw = _bf16_bytes(f32)

    auto = codec.compress(raw, dtype="bf16")  # decode-aware AUTO
    bitfield = codec.compress(raw, dtype="bf16", auto=False)  # bit-field only

    # AUTO (byte-plane) gives up at most _BYTEPLANE_DECODE_TOLERANCE on size.
    assert len(auto) <= codec._BYTEPLANE_DECODE_TOLERANCE * len(bitfield)


def test_bf16_structured_still_takes_ldm_win() -> None:
    """Duplicated-layer bf16 must keep the whole-buffer LDM ratio win (not byte-plane)."""
    rng = np.random.default_rng(2)
    base = _bf16_bytes(rng.standard_normal(1_000_000).astype(np.float32))
    raw = base * 16  # 16 identical layers -> long-range redundancy

    frame = codec.compress(raw, dtype="bf16")
    assert z4ai.decompress(frame) == raw
    # The LDM/opaque candidate must crush this far beyond any per-element split.
    assert len(raw) / len(frame) > 5.0
