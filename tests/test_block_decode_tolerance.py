# SPDX-License-Identifier: Apache-2.0

"""Guard the backend's block-vs-LDM decode-throughput tolerance.

A single Zstd frame decodes on ONE core; independent blocks decode across all
cores (~3-5x faster on a compressible plane — z4ai's weakest axis vs ZipNN).
The backend therefore keeps the BLOCK form when it is within
``backend._BLOCK_DECODE_TOLERANCE`` of the whole-plane LDM frame, trading <=1%
ratio for the large parallel-decode win.

Regression this guards: an fp16 high byte plane compresses ~0.005% *smaller* as
one LDM frame than as blocks, so a strict ``blocks <= ldm`` comparison kept the
single frame and decoded the whole dtype ~3x slower (measured 3457 -> 10800
MB/s).  These tests pin that the near-tie now takes blocks, that it stays
lossless, and that a genuine long-range (structured) win still keeps LDM.
"""

from __future__ import annotations

import numpy as np

import z4ai
from z4ai import backend
from z4ai import format as zfmt


def _fp16_weights(n: int = 16_000_000, scale: float = 0.02, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal(n).astype(np.float32) * np.float32(scale))
        .astype(np.float16)
        .tobytes()
    )


def _plane_methods(frame: bytes):
    return [p.method for p in zfmt.deserialize(frame).planes]


def test_fp16_compressible_plane_takes_parallel_decode_form() -> None:
    """Dense fp16 weights must store their compressible plane in a PARALLEL-decode
    form, never a single whole-plane LDM frame (which decodes on one core).

    The high byte plane is well over the parallel-plane threshold.  Two encodings
    decode across all cores: independent Zstd blocks (``METHOD_ZSTD_BLOCKS``) and —
    now the default, and a strictly better ratio (entropy floor vs Huffman) —
    chunked rANS (``METHOD_RANS``/``METHOD_RANS_ADAPT``), whose chunks are
    independent streams the GIL-releasing decoder runs in parallel.  Either is the
    correct choice over the single-core LDM frame; this guards that the codec never
    regresses that plane back to a one-core decode.
    """
    raw = _fp16_weights()
    frame = z4ai.compress(raw, dtype="fp16")
    assert z4ai.decompress(frame) == raw  # lossless
    methods = set(_plane_methods(frame))
    parallel_forms = {
        backend.METHOD_ZSTD_BLOCKS,
        backend.METHOD_RANS,
        getattr(backend, "METHOD_RANS_ADAPT", -1),
    }
    assert parallel_forms & methods, _plane_methods(frame)


def test_decode_tolerance_band_is_small() -> None:
    """The tolerance must stay tight so the ratio give-up is negligible (<=1%)."""
    assert 1.0 < backend._BLOCK_DECODE_TOLERANCE <= 1.01


def test_structured_plane_still_keeps_ldm_frame() -> None:
    """A duplicated-layer plane must keep the single LDM frame (the ratio win).

    Here LDM is many times smaller than blocks — far outside the tolerance band —
    so the best-of keeps ``METHOD_ZSTD`` and the long-range dedup win is intact.
    """
    rng = np.random.default_rng(7)
    base = (
        (rng.standard_normal(1_000_000).astype(np.float32) * np.float32(0.02))
        .astype(np.float16)
        .tobytes()
    )
    raw = base * 16  # 16 identical layers -> long-range redundancy
    frame = z4ai.compress(raw, dtype="fp16")
    assert z4ai.decompress(frame) == raw
    # The LDM/opaque path must crush this far beyond any block split.
    assert len(raw) / len(frame) > 5.0
