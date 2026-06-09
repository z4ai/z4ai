# SPDX-License-Identifier: Apache-2.0

"""Round-trip fuzz for the adaptive backend's two routing branches.

The backend chooses, per plane, between a whole-plane LDM frame (structured
data) and a block-parallel frame (i.i.d. data), with a raw-store guard on top.
The block path frames N independent Zstd blocks inside the plane payload, so its
correctness hinges on exact behaviour at block boundaries.  These tests pin
round-trip exactness at sizes straddling the block boundary, for both routing
branches (forced via structured vs random content) and every public dtype.
"""

from __future__ import annotations

import numpy as np
import pytest

import z4ai
from z4ai import backend

# The size at which the codec starts framing a plane into parallel blocks.
_BS = backend._DEFAULT_BLOCK_SIZE

# Sizes that straddle the block boundary and the small-input edges.  Kept to a
# couple of multiples of the (multi-MiB) block so the suite stays fast.
_SIZES = [
    0,
    1,
    3,
    16,
    4096,
    4097,
    _BS - 1,
    _BS,
    _BS + 1,
    2 * _BS + 7,
]


def _make(kind: str, size: int) -> bytes:
    rng = np.random.default_rng(0xC0FFEE ^ size ^ hash(kind) & 0xFFFF)
    if kind == "random":  # forces the block-parallel branch (no repeats)
        return rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    if kind == "structured":  # forces the whole-plane LDM branch (long repeats)
        if size == 0:
            return b""
        block = rng.integers(0, 256, size=min(size, 4096), dtype=np.uint8).tobytes()
        return (block * (size // len(block) + 1))[:size]
    if kind == "zeros":
        return b"\x00" * size
    raise ValueError(kind)


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("kind", ["random", "structured", "zeros"])
@pytest.mark.parametrize("dtype", [None, "fp32", "bf16", "uint8"])
def test_block_path_roundtrip(kind, size, dtype):
    raw = _make(kind, size)
    blob = z4ai.compress(raw, dtype=dtype)
    assert bytes(z4ai.decompress(blob)) == raw
    # Never expand beyond a small header (the store guard must hold here too).
    assert len(blob) <= len(raw) + 256


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_block_path_preserves_numpy_roundtrip(dtype):
    """Multi-block arrays restore dtype and shape exactly."""
    np_dtype = np.float32 if dtype == "fp32" else np.uint16
    n = (2 * _BS) // np.dtype(np_dtype).itemsize + 5
    arr = np.random.default_rng(7).integers(0, 60000, size=n).astype(np_dtype)
    blob = z4ai.compress(arr)
    back = z4ai.decompress_ndarray(blob)
    assert back.dtype == arr.dtype and back.shape == arr.shape
    assert np.array_equal(back, arr)
