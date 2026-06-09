# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the long-range-redundancy (LDM) win over ZipNN.

A prior detector (exact chunk-hash collisions) missed redundancy whose repeat
period was misaligned to its sample stride, so the codec silently skipped the
opaque LDM candidate and fell *below* plain Zstd on structured weights (20.7x vs
51x). These tests pin the two invariants that prevent a recurrence:

  1. The detector fires on long-range-redundant data and stays quiet on i.i.d.
  2. On structured weights, z4ai is never worse than a plain whole-buffer Zstd
     pass, and crushes a fixed-chunk codec - which is the headline claim.
"""

import zstandard as zstd

from _data import make_scenario


def _zstd_ratio(data: bytes, level: int = 3) -> float:
    params = zstd.ZstdCompressionParameters.from_level(level, enable_ldm=True)
    comp = zstd.ZstdCompressor(compression_params=params).compress(data)
    return len(data) / len(comp)


def test_detector_fires_on_structured_not_iid(z4ai):
    from z4ai import backend

    structured = make_scenario("structured", 4_000_000, "bf16", seed=0)
    iid = make_scenario("iid", 4_000_000, "bf16", seed=0)
    assert backend._has_long_range_redundancy(structured) is True
    assert backend._has_long_range_redundancy(iid) is False


def test_structured_never_worse_than_plain_zstd(z4ai):
    """The core regression: structured weights must not lose to plain Zstd."""
    data = make_scenario("structured", 6_000_000, "bf16", seed=1)
    blob = z4ai.compress(data, dtype="bf16")
    assert bytes(z4ai.decompress(blob)) == data  # lossless first
    z4ai_ratio = len(data) / len(blob)
    zstd_ratio = _zstd_ratio(data)
    # Allow a small framing-overhead margin, but z4ai must be in zstd's league -
    # not the 2.5x-worse regression we are guarding against.
    assert z4ai_ratio >= 0.95 * zstd_ratio, (
        f"z4ai {z4ai_ratio:.1f}x must not lose to plain zstd {zstd_ratio:.1f}x "
        f"on structured weights (LDM candidate skipped?)"
    )


def test_structured_large_ratio_win(z4ai):
    """Sanity floor for the headline win: structured weights compress far past
    a fixed-256KiB-chunk codec's reach (ZipNN gets ~1.5x here)."""
    data = make_scenario("structured", 6_000_000, "bf16", seed=2)
    ratio = len(data) / len(z4ai.compress(data, dtype="bf16"))
    assert (
        ratio > 10.0
    ), f"expected a large LDM win on structured weights, got {ratio:.1f}x"


def test_iid_ratio_unchanged_by_detector(z4ai):
    """The fast path must still produce the i.i.d. ratio (~1.4x on bf16)."""
    data = make_scenario("iid", 4_000_000, "bf16", seed=3)
    ratio = len(data) / len(z4ai.compress(data, dtype="bf16"))
    assert 1.3 < ratio < 1.6
