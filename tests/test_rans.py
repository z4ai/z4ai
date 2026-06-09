# SPDX-License-Identifier: Apache-2.0

"""Round-trip and model tests for the native rANS entropy coder.

Skipped automatically if no C compiler is available (e.g. minimal CI images),
since the codec falls back to Zstd in that case.
"""

from __future__ import annotations

import numpy as np
import pytest

from z4ai import rans

pytestmark = pytest.mark.skipif(
    not rans.available(), reason="native rANS unavailable (no C compiler)"
)


@pytest.mark.parametrize(
    "buf",
    [
        b"",
        b"\x00",
        b"\x07",
        b"\x05" * 1000,  # single symbol
        b"\x00\xff" * 5000,  # two symbols
        bytes(range(256)) * 37,  # uniform alphabet
    ],
)
def test_roundtrip_fixed(buf):
    assert rans.decompress(rans.compress(buf)) == buf


@pytest.mark.parametrize("alphabet", [2, 8, 64, 256])
@pytest.mark.parametrize("size", [1, 2, 100, 100_000])
def test_roundtrip_random(alphabet, size):
    rng = np.random.default_rng(alphabet * 1000 + size)
    buf = rng.integers(0, alphabet, size, dtype=np.uint8).tobytes()
    out = rans.decompress(rans.compress(buf))
    assert out == buf


def test_skewed_distribution_compresses():
    # A low-entropy distribution must compress well below the raw size.
    rng = np.random.default_rng(0)
    buf = rng.integers(0, 4, 1_000_000, dtype=np.uint8).tobytes()
    comp = rans.compress(buf)
    assert len(comp) < len(buf) * 0.5  # ~2 bits/symbol vs 8


def test_uniform_distribution_does_not_underflow():
    # Near-incompressible input: the coder must still be exactly reversible.
    rng = np.random.default_rng(1)
    buf = rng.integers(0, 256, 500_000, dtype=np.uint8).tobytes()
    assert rans.decompress(rans.compress(buf)) == buf


def test_python_fallback_matches_native(monkeypatch):
    """A rANS frame produced with the native coder must decode byte-for-byte
    through the pure-Python fallback, so frames stay portable to hosts without
    a C compiler."""
    rng = np.random.default_rng(3)
    for buf in (
        rng.integers(0, 8, 200_000, dtype=np.uint8).tobytes(),
        rng.integers(0, 256, 100_000, dtype=np.uint8).tobytes(),
        b"\x09" * 4000,
        b"\x01\x02\x03",
    ):
        blob = rans.compress(buf)
        native = rans.decompress(blob)
        assert native == buf
        # Force the fallback path by pretending the native lib is unloadable.
        monkeypatch.setattr(rans, "_load", lambda: None)
        monkeypatch.setattr(rans, "_warned_fallback", False)
        with pytest.warns(RuntimeWarning):
            fallback = rans.decompress(blob)
        monkeypatch.undo()
        assert fallback == buf


def test_decode_is_portable():
    assert rans.decode_is_portable() is True


def test_normalize_sums_to_scale():
    rng = np.random.default_rng(2)
    for _ in range(20):
        hist = rng.integers(0, 1000, 256).astype(np.int64)
        hist[rng.random(256) < 0.5] = 0  # make it sparse
        freq = rans._normalize(hist)
        if hist.sum() == 0:
            assert int(freq.sum()) == 0
            continue
        assert int(freq.sum()) == rans.PROB_SCALE
        # every present symbol keeps a nonzero frequency
        assert np.all(freq[hist > 0] >= 1)
