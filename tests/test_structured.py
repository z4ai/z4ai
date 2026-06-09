# SPDX-License-Identifier: Apache-2.0

"""Coverage for the long-range / highly-compressible path - z4ai's headline win.

The big win over ZipNN is on *structured* weights (tied embeddings, repeated
layers, EMA/optimizer copies): whole-plane long-distance matching deduplicates
redundancy megabytes apart, which ZipNN's chunked Huffman cannot.

This path is distinct from the i.i.d. path: it activates the
"plane IS compressible" branch (LDM, escalation, per-plane probe). A regression
there once produced a `NameError` that the i.i.d. tests did not catch because
they never hit a strongly-compressible plane. These tests pin that path down:
losslessness AND that structured data really does compress far better than noise.
"""

import numpy as np
import pytest

# The structured generator lives in benchmarks/_data.py (on sys.path via conftest).
_data = pytest.importorskip("_data")
make_weights = _data.make_weights
make_scenario = getattr(_data, "make_scenario", None)

FLOAT_DTYPES = ["bf16", "fp16", "fp32", "fp64"]


def _structured(n_elem, dtype, seed=0):
    """Structured payload: prefer the shared generator, else build one locally."""
    if make_scenario is not None:
        return make_scenario("structured", n_elem, dtype=dtype, seed=seed)
    # Fallback: tile a single random block so it repeats far apart (LDM bait).
    block = make_weights(max(1, n_elem // 16), dtype=dtype, seed=seed)
    reps = (n_elem * _data.bytes_per_element(dtype)) // len(block) + 1
    return (block * reps)[: n_elem * _data.bytes_per_element(dtype)]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_structured_roundtrip_is_lossless(z4ai, dtype):
    """The compressible/LDM branch must stay byte-exact for every dtype."""
    data = _structured(2_000_000, dtype, seed=1)
    blob = z4ai.compress(data, dtype=dtype)
    assert bytes(z4ai.decompress(blob)) == data


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_structured_beats_iid_by_a_lot(z4ai, dtype):
    """Redundant weights must compress dramatically better than i.i.d. noise.

    This is the property that makes z4ai beat ZipNN on real checkpoints; if a
    refactor silently disables long-range matching, this ratio collapses.
    """
    n = 2_000_000
    iid = make_weights(n, dtype=dtype, seed=2)
    structured = _structured(n, dtype, seed=2)
    iid_ratio = len(iid) / len(z4ai.compress(iid, dtype=dtype))
    str_ratio = len(structured) / len(z4ai.compress(structured, dtype=dtype))
    assert str_ratio > 5 * iid_ratio, (
        f"structured {str_ratio:.1f}x not >> iid {iid_ratio:.1f}x - "
        "long-range matching may be disabled"
    )


def test_highly_redundant_plane_path(z4ai):
    """A trivially-repeating buffer must hit the compressible branch and shrink hard."""
    data = (np.arange(256, dtype=np.uint8).tobytes()) * 40_000  # ~10 MB, period 256
    blob = z4ai.compress(data, dtype="fp32")
    assert bytes(z4ai.decompress(blob)) == data
    assert len(data) / len(blob) > 20, "expected a large ratio on periodic data"
