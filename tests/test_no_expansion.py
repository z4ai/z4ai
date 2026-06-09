# SPDX-License-Identifier: Apache-2.0

"""Regression guards for two production-critical invariants.

These exist because both invariants were violated by earlier revisions and the
breakage was silent (still lossless, just useless / pathologically slow):

* A compressor must NEVER expand its input by more than a tiny constant header.
  The bit-field path once applied to fp16 expanded data to ratio 0.817.
* The bit-field path must only be used where it actually helps (bf16), and the
  default config must stay fast.  These tests will fail loudly if someone
  re-introduces a mantissa-misfitting bit-field dtype or a slow default level.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import z4ai

# Maximum bytes of container overhead we tolerate for an incompressible input.
_MAX_HEADER_SLACK = 256


def _gaussian(dtype, n=600 * 600):
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(dtype)


@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(1)
        .integers(0, 256, size=500_000, dtype=np.uint8)
        .tobytes(),
        _gaussian(np.float16),
        _gaussian(np.float32),
        _gaussian(np.float64),
        b"",
        b"\x00",
        b"abc",
    ],
    ids=["random-bytes", "fp16", "fp32", "fp64", "empty", "one-byte", "tiny"],
)
def test_never_expands(data):
    """Every input round-trips and never grows beyond a small header slack."""
    raw = data if isinstance(data, (bytes, bytearray)) else data.tobytes()
    blob = z4ai.compress(data)
    assert bytes(z4ai.decompress(blob)) == raw
    assert (
        len(blob) <= len(raw) + _MAX_HEADER_SLACK
    ), f"compressor EXPANDED {len(raw)} -> {len(blob)} bytes"


def test_fp16_compresses_not_expands():
    """fp16 gaussians must compress (ratio > 1), not expand (the 0.817 bug)."""
    w = _gaussian(np.float16, 1000 * 1000)
    raw = w.tobytes()
    blob = z4ai.compress(w)
    ratio = len(raw) / len(blob)
    assert ratio > 1.0, f"fp16 ratio {ratio:.3f} <= 1.0 (regression!)"


def test_default_is_fast():
    """The default config must not be pathologically slow (no escalate-by-default).

    A generous ceiling: escalation-by-default made this ~1.2 s; the fast default
    is ~10 ms.  1 second leaves huge headroom while still catching a regression.
    """
    w = _gaussian(np.float32, 1500 * 1500)
    start = time.perf_counter()
    z4ai.compress(w)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"default compress took {elapsed:.3f}s (escalate-by-default?)"
