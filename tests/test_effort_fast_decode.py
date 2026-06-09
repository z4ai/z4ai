# SPDX-License-Identifier: Apache-2.0

"""Guard the decode-speed contract of ``effort="fast"``.

``effort="fast"`` must keep the block-parallel Zstd plane form (METHOD_ZSTD_BLOCKS),
which decodes across all cores, rather than the order-1 rANS form used by the
ratio-first default tier (rANS reaches the entropy floor but its streams decode
~6x slower).  This pins that contract: a regression that lets rANS leak back into
the fast tier (as happened once when rANS was made on-by-default) fails here.
"""

import numpy as np
import pytest

import z4ai
from z4ai import backend, format as fmt

_RANS_METHODS = {backend.METHOD_RANS, backend.METHOD_RANS_O1}


def _bf16_exponent_heavy(mb=8):
    """A buffer whose compressible plane is large enough to take the block path."""
    n = mb * 1024 * 1024
    w = np.random.randn(n // 2).astype(np.float32)
    return (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_fast_tier_is_lossless_and_uses_no_rans(dtype):
    if dtype == "fp32":
        data = np.random.randn(4 * 1024 * 1024).astype(np.float32).tobytes()
    else:
        data = _bf16_exponent_heavy()

    blob = z4ai.compress(data, dtype=dtype, effort="fast")
    assert z4ai.decompress(blob) == data, "fast tier must be lossless"

    # Streaming (Z4AI) frame expected for the fast tier; inspect its plane methods.
    container = fmt.deserialize(blob)
    methods = {p.method for p in container.planes}
    assert not (
        methods & _RANS_METHODS
    ), f"effort='fast' must not use rANS (slow serial decode); got methods {methods}"
    # At least one plane should take the block-parallel form on this large,
    # compressible input — that is the whole point of the fast tier.
    assert (
        backend.METHOD_ZSTD_BLOCKS in methods
    ), f"expected a block-parallel plane in the fast tier; got {methods}"


def test_fast_tier_decodes_at_least_as_fast_as_default():
    """Sanity: the fast tier never decodes slower than the ratio-first default."""
    import time

    data = _bf16_exponent_heavy(8)
    blobs = {e: z4ai.compress(data, dtype="bf16", effort=e) for e in ("fast", None)}
    for b in blobs.values():
        assert z4ai.decompress(b) == data

    def best_decode_s(b):
        best = float("inf")
        for _ in range(5):
            t = time.perf_counter()
            z4ai.decompress(b)
            best = min(best, time.perf_counter() - t)
        return best

    fast_s = best_decode_s(blobs["fast"])
    default_s = best_decode_s(blobs[None])
    # Generous margin (10%) to absorb scheduler noise on a loaded CI box; the real
    # gap is multiples.  The point is fast must not be *slower* to decode.
    assert fast_s <= default_s * 1.10, (
        f"fast tier decode {fast_s*1e3:.1f}ms should not exceed default "
        f"{default_s*1e3:.1f}ms"
    )
