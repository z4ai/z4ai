# SPDX-License-Identifier: Apache-2.0

"""Tests for the parallel Turbo codec (:mod:`z4ai.turbo`)."""
import numpy as np
import pytest

turbo = pytest.importorskip("z4ai.turbo")


WIDTHS = [1, 2, 4, 8]


def _bf16_like(n, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n).astype(np.float32)
    return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()


# --- losslessness across widths, sizes, and worker counts --------------------

@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("nbytes", [0, 1, 2, 7, 8, 1000, 100_003])
def test_roundtrip_small(width, nbytes):
    data = bytes((i * 37 + 11) & 0xFF for i in range(nbytes))
    blob = turbo.compress(data, width=width)
    assert bytes(turbo.decompress(blob)) == data


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("workers", [None, 1, 4])
def test_roundtrip_large(width, workers):
    data = _bf16_like(2_000_000)  # 4 MB
    blob = turbo.compress(data, width=width, workers=workers)
    assert bytes(turbo.decompress(blob, workers=workers)) == data


def test_roundtrip_non_aligned_tail():
    # Length not a multiple of width: falls back to opaque, still exact.
    data = _bf16_like(50_000) + b"\x01\x02\x03"
    blob = turbo.compress(data, width=2)
    assert bytes(turbo.decompress(blob)) == data


@pytest.mark.parametrize("pattern", [b"\x00", b"\xff", b"\xaa\x55"])
def test_roundtrip_degenerate(pattern):
    data = pattern * 100_000
    assert bytes(turbo.decompress(turbo.compress(data, width=2))) == data


def test_roundtrip_random_bytes():
    data = np.random.default_rng(3).integers(0, 256, 500_000, np.uint8).tobytes()
    assert bytes(turbo.decompress(turbo.compress(data, width=1))) == data


# --- frame + guarantees -------------------------------------------------------

def test_frame_magic():
    assert turbo.compress(_bf16_like(10_000), width=2)[:4] == turbo.MAGIC == b"ZRT1"


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        turbo.decompress(b"NOPE" + b"\x00" * 32)


def test_never_much_larger_than_input():
    data = np.random.default_rng(1).integers(0, 256, 1_000_000, np.uint8).tobytes()
    blob = turbo.compress(data, width=2)
    assert len(blob) <= len(data) + 4096  # only a tiny fixed header overhead


def test_workers_are_lossless_regardless_of_count():
    # Zstd's multi-threaded framing may differ across thread counts, so frames
    # need not be byte-identical - but every count must round-trip exactly.
    data = _bf16_like(1_500_000)
    for w in (None, 1, 2, 8):
        blob = turbo.compress(data, width=2, workers=w)
        assert bytes(turbo.decompress(blob)) == data


# --- competitive behaviours we must not regress ------------------------------

def test_never_worse_than_raw_zstd():
    """The opaque fallback guarantees Turbo is never materially worse than a
    plain Zstd pass - even on element-repeating data where the byte-plane
    transpose would hurt."""
    import zstandard as zstd

    base = _bf16_like(200_000)          # 400 KB base block
    data = base * 16                    # 6.4 MB, repeats every 400 KB
    blob = turbo.compress(data, width=2, workers=1)
    raw = zstd.ZstdCompressor(level=turbo.DEFAULT_LEVEL).compress(data)
    assert bytes(turbo.decompress(blob)) == data
    assert len(blob) <= len(raw) * 1.02   # opaque fallback ~= raw zstd + header
    assert len(blob) < len(data) // 8     # and it compressed a lot


def test_compresses_iid_float_weights():
    data = _bf16_like(2_000_000)
    blob = turbo.compress(data, width=2)
    assert bytes(turbo.decompress(blob)) == data
    assert len(data) / len(blob) > 1.2    # exponent-plane win


def test_picks_planes_for_float_and_opaque_for_repeats():
    # Float weights -> the byte-plane transform should win.
    floats = _bf16_like(1_000_000)
    fblob = turbo.compress(floats, width=2, workers=1)
    # Element-repeating data -> opaque should win (transpose would split runs).
    reps = (_bf16_like(50_000)) * 20
    rblob = turbo.compress(reps, width=2, workers=1)
    # Both lossless; the repeat case must compress dramatically better.
    assert bytes(turbo.decompress(fblob)) == floats
    assert bytes(turbo.decompress(rblob)) == reps
    assert len(reps) / len(rblob) > len(floats) / len(fblob)
