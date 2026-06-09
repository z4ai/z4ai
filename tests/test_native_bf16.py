# SPDX-License-Identifier: Apache-2.0

"""Guard the *native* bf16 field split/join against the NumPy reference.

``z4ai/bitfield.py`` has two interchangeable bf16 implementations: a pure-NumPy
path and a one-pass C path (``_native_shuffle.bf16_split``/``bf16_join`` via
:mod:`z4ai._accel`).  The codec's round-trip tests only assert
``join(split(x)) == x``, which passes even if *both* paths share an identical
bug.  These tests instead assert the native path is **byte-for-byte identical to
the NumPy reference** for the three streams, so a divergence in the C bit math
(exponent/mantissa masks, MSB-first sign packing, little-endian element order)
fails loudly — and only on hosts where the extension is actually built.

Skipped cleanly when the C extension is not present (NumPy fallback only).
"""

from __future__ import annotations

import numpy as np
import pytest

from z4ai import _accel, bitfield as bf

pytestmark = pytest.mark.skipif(
    not getattr(_accel, "HAVE_NATIVE", False),
    reason="native _native_shuffle extension not built on this host",
)


def _numpy_reference(raw: bytes):
    """The exact NumPy split the native path must reproduce, for bf16."""
    u = np.frombuffer(raw, dtype="<u2")
    sign = np.packbits((u >> 15).astype(np.uint8)).tobytes()
    exponent = ((u >> 7) & 0xFF).astype(np.uint8).tobytes()
    mantissa = (u & 0x7F).astype(np.uint8).tobytes()
    return sign, exponent, mantissa


# Counts chosen to exercise the partial-final-sign-byte arithmetic (packbits
# pads the last byte) and the empty/singleton edges.
_COUNTS = [0, 1, 2, 7, 8, 9, 15, 16, 17, 255, 256, 1000, 1 << 16]


@pytest.mark.parametrize("count", _COUNTS)
def test_native_split_matches_numpy_reference(count):
    rng = np.random.default_rng(count + 1)
    raw = rng.integers(0, 1 << 16, size=count, dtype=np.uint16).tobytes()

    s_n, e_n, m_n = _accel.bf16_split(raw)
    s_ref, e_ref, m_ref = _numpy_reference(raw)

    assert bytes(s_n) == s_ref, "native sign plane diverged from packbits reference"
    assert bytes(e_n) == e_ref, "native exponent plane diverged from reference"
    assert bytes(m_n) == m_ref, "native mantissa plane diverged from reference"
    # Stream lengths follow the documented contract.
    assert len(e_n) == count and len(m_n) == count
    assert len(s_n) == (count + 7) // 8


@pytest.mark.parametrize("count", _COUNTS)
def test_native_join_inverts_split(count):
    rng = np.random.default_rng(count + 7)
    raw = rng.integers(0, 1 << 16, size=count, dtype=np.uint16).tobytes()

    sign, exponent, mantissa = _accel.bf16_split(raw)
    rebuilt = _accel.bf16_join(sign, exponent, mantissa, count)
    assert bytes(rebuilt) == raw


def test_bitfield_module_native_path_is_byte_exact():
    """The high-level bitfield API round-trips byte-exactly via the native path."""
    rng = np.random.default_rng(123)
    raw = rng.integers(0, 1 << 16, size=50_000, dtype=np.uint16).tobytes()
    result = bf.split_fields(raw, dtype="bfloat16")
    # The three streams must equal the NumPy reference regardless of which path ran.
    s_ref, e_ref, m_ref = _numpy_reference(raw)
    assert bytes(result.sign) == s_ref
    assert bytes(result.exponent) == e_ref
    assert bytes(result.mantissa) == m_ref
    assert bf.join_fields(result) == raw


def test_native_split_covers_all_sign_bit_patterns():
    """Every one of the 8 sign-bit positions in a packed byte is exercised."""
    # Alternating and walking sign bits stress the MSB-first packing shift.
    for pattern in (0x8000, 0x0000, 0xFFFF, 0x7FFF):
        u = np.full(64, pattern, dtype=np.uint16)
        # Flip individual signs so packed bytes are non-trivial.
        u[::3] ^= 0x8000
        raw = u.tobytes()
        s_n, e_n, m_n = _accel.bf16_split(raw)
        s_ref, e_ref, m_ref = _numpy_reference(raw)
        assert (bytes(s_n), bytes(e_n), bytes(m_n)) == (s_ref, e_ref, m_ref)
        assert _accel.bf16_join(s_n, e_n, m_n, len(u)) == raw
