# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused multithreaded native codec (``z4ai._native_codec``).

The native codec produces and consumes the *identical* ``Z4AIMF01`` frame as the
pure-Python :mod:`z4ai.chunked` pipeline, so the contract is:

* native -> native round-trips byte-exactly, and
* native and pure-Python are byte-interoperable in BOTH directions.

Every test is skipped cleanly when the extension is not built, so CI without a C
compiler / zstd headers still passes (the chunked path falls back to Python).
"""
from __future__ import annotations

import os
import random

import numpy as np
import pytest

import z4ai.chunked as ck

nc = pytest.importorskip(
    "z4ai._native_codec", reason="native codec extension not built"
)

CPU = max(1, os.cpu_count() or 1)
WIDTHS = {"bf16": 2, "fp32": 4, "fp64": 8, "uint16": 2, "opaque": 1}


def _py_compress(raw, dtype, level, chunk):
    """Force the pure-Python chunked pipeline regardless of the native ext."""
    saved = ck._native
    ck._native = None
    try:
        return ck.compress(raw, dtype=dtype, level=level, chunk_size=chunk)
    finally:
        ck._native = saved


def _py_decompress(blob):
    saved = ck._native
    ck._native = None
    try:
        return ck.decompress(blob)
    finally:
        ck._native = saved


@pytest.mark.parametrize("dtype,width", list(WIDTHS.items()))
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 15, 16, 17, 255, 4096, 4097, 1_000_003])
def test_native_roundtrip(dtype, width, n):
    raw = bytes(np.random.randint(0, 256, n, np.uint8))
    blob = nc.compress(raw, width, 1, CPU, 2 << 20)
    assert bytes(nc.decompress(blob, CPU)) == raw


@pytest.mark.parametrize("dtype,width", list(WIDTHS.items()))
def test_cross_native_to_python(dtype, width):
    raw = bytes(np.random.randint(0, 256, 500_003, np.uint8))
    blob = nc.compress(raw, width, 1, CPU, 1 << 20)
    assert bytes(_py_decompress(blob)) == raw  # native frame -> python decoder


@pytest.mark.parametrize("dtype,width", list(WIDTHS.items()))
def test_cross_python_to_native(dtype, width):
    raw = bytes(np.random.randint(0, 256, 500_003, np.uint8))
    blob = _py_compress(raw, dtype if dtype != "opaque" else None, 1, 1 << 20)
    assert bytes(nc.decompress(blob, CPU)) == raw  # python frame -> native decoder


def test_realistic_float_weights_roundtrip():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(2_000_000) * 0.02).astype(np.float32)
    for dtype, width, raw in [
        ("fp32", 4, w.tobytes()),
        ("bf16", 2, (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()),
    ]:
        blob = nc.compress(raw, width, 1, CPU, 2 << 20)
        # actually shrinks (exponent is low-entropy) and decodes exactly
        assert len(blob) < len(raw)
        assert bytes(nc.decompress(blob, CPU)) == raw


def test_levels_and_threads_roundtrip():
    raw = (np.random.default_rng(1).standard_normal(300_000)
           .astype(np.float32).tobytes())
    for level in (-1, 1, 3, 9):
        for threads in (1, 2, CPU):
            blob = nc.compress(raw, 4, level, threads, 1 << 20)
            assert bytes(nc.decompress(blob, threads)) == raw


def test_single_threaded_matches_multi():
    raw = (np.random.default_rng(2).standard_normal(400_000)
           .astype(np.float32).tobytes())
    b1 = nc.compress(raw, 4, 1, 1, 1 << 20)
    bN = nc.compress(raw, 4, 1, CPU, 1 << 20)
    # different thread counts may pick different chunk boundaries only if chunk
    # size changes; here chunk size is fixed so frames are identical.
    assert b1 == bN
    assert bytes(nc.decompress(b1, CPU)) == raw
    assert bytes(nc.decompress(bN, 1)) == raw


def test_corrupt_frames_raise_not_crash():
    good = nc.compress(b"hello world" * 1000, 2, 1, CPU, 1 << 20)
    for bad in (b"", b"Z4AIMF01", b"NOTMAGIC" + good[8:], good[: len(good) // 2]):
        with pytest.raises((ValueError, RuntimeError)):
            nc.decompress(bad, CPU)


def test_fuzz_cross_validation():
    rng = random.Random(12345)
    for _ in range(120):
        n = rng.choice([0, 1, 5, 63, 64, 65, 4096, rng.randint(0, 2_000_000)])
        raw = bytes(np.random.randint(0, 256, n, np.uint8))
        dtype = rng.choice(list(WIDTHS))
        width = WIDTHS[dtype]
        level = rng.choice([1, 1, 3, -1])
        chunk = rng.choice([512, 4096, 1 << 16, 2 << 20])
        nb = nc.compress(raw, width, level, CPU, chunk)
        assert bytes(nc.decompress(nb, CPU)) == raw
        assert bytes(_py_decompress(nb)) == raw
