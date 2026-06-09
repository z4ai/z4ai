# SPDX-License-Identifier: Apache-2.0

"""Round-trip and correctness tests for :mod:`z4ai.bitfield`.

The central contract is *lossless, byte-exact* reconstruction.  We hammer it
with edge cases (empty, single element, denormals, inf/nan, signed zero) plus
randomized fuzzing across every supported format.
"""

from __future__ import annotations

import numpy as np
import pytest

from z4ai import bitfield as bf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip_raw(raw: bytes, spec_name: str) -> bytes:
    res = bf.split_fields(raw, dtype=spec_name)
    return bf.join_fields(res)


def _bf16_from_float32(arr32: np.ndarray) -> np.ndarray:
    """Truncate float32 to bfloat16 bit pattern (uint16, round-toward-zero)."""
    u32 = arr32.astype("<f4").view("<u4")
    return (u32 >> 16).astype("<u2")


# ---------------------------------------------------------------------------
# float32 / float16 native round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_roundtrip_native_random(dtype):
    rng = np.random.default_rng(1234)
    arr = (rng.standard_normal(4096) * 3.0).astype(dtype)
    res = bf.split_fields(arr)
    out = bf.join_fields(res)
    assert out == np.ascontiguousarray(arr).tobytes()
    back = bf.to_array(out, dtype)
    # NaN-safe comparison on the raw bits.
    assert back.view(np.uint8).tobytes() == arr.view(np.uint8).tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_special_values(dtype):
    specials = np.array(
        [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan,
         np.finfo(dtype).tiny, np.finfo(dtype).max, -np.finfo(dtype).max],
        dtype=dtype,
    )
    out = bf.join_fields(bf.split_fields(specials))
    assert out == specials.view(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# bfloat16 (raw uint16) round-trips
# ---------------------------------------------------------------------------

def test_roundtrip_bfloat16_random():
    rng = np.random.default_rng(99)
    arr32 = (rng.standard_normal(4096) * 2.0).astype("<f4")
    bf16 = _bf16_from_float32(arr32)
    raw = bf16.tobytes()
    res = bf.split_fields(raw, dtype="bfloat16")
    assert res.spec_name == "bfloat16"
    assert bf.join_fields(res) == raw


def test_bfloat16_via_uint16_array():
    bf16 = np.array([0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F80, 0xFFC0], dtype="<u2")
    res = bf.split_fields(bf16, dtype="bfloat16")
    assert bf.join_fields(res) == bf16.tobytes()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_empty(dtype):
    raw = b""
    res = bf.split_fields(raw, dtype=dtype)
    assert res.count == 0
    assert bf.join_fields(res) == b""


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_single_element(dtype):
    spec = bf.SPECS[dtype]
    raw = (b"\xAB" * spec.itemsize)
    assert _roundtrip_raw(raw, dtype) == raw


def test_unaligned_buffer_rejected():
    with pytest.raises(ValueError):
        bf.split_fields(b"\x00\x01\x02", dtype="float32")  # 3 bytes, not /4


def test_unsupported_dtype():
    with pytest.raises(KeyError):
        bf.resolve_spec("float64")


def test_raw_bytes_requires_dtype():
    with pytest.raises(ValueError):
        bf.split_fields(b"\x00\x00\x00\x00")


# ---------------------------------------------------------------------------
# Stream shape invariants - the sign stream really is 8x smaller, etc.
# ---------------------------------------------------------------------------

def test_stream_sizes():
    n = 1000
    arr = np.ones(n, dtype="float32")
    res = bf.split_fields(arr)
    assert len(res.exponent) == n                 # one byte per element
    assert len(res.sign) == (n + 7) // 8          # bit-packed
    assert len(res.mantissa) == n * 3             # 23 mantissa bits -> 3 bytes


def test_aliases_resolve():
    for alias, canon in [("fp32", "float32"), ("bf16", "bfloat16"), ("half", "float16")]:
        assert bf.resolve_spec(alias).name == canon


# ---------------------------------------------------------------------------
# Fuzz: many random buffers, all formats, must be byte-exact.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("seed", range(8))
def test_fuzz_random_bits(dtype, seed):
    rng = np.random.default_rng(seed)
    spec = bf.SPECS[dtype]
    n = int(rng.integers(0, 500))
    raw = rng.integers(0, 256, size=n * spec.itemsize, dtype=np.uint8).tobytes()
    assert _roundtrip_raw(raw, dtype) == raw
