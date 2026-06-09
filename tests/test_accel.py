# SPDX-License-Identifier: Apache-2.0

"""The native SIMD shuffle must be byte-identical to the NumPy reference.

If the `_native_shuffle` extension is built (`python -m z4ai.build_native`),
these tests cross-check it against the pure-NumPy transpose for every width and
tail length. If it is not built, they verify the fallback path is self
consistent. Either way, the codec's own round-trip is exercised.
"""

import os

import pytest

z4ai = pytest.importorskip("z4ai")

from z4ai import _accel, transforms  # noqa: E402

WIDTHS = [1, 2, 3, 4, 8]
TAILS = [0, 1, 2, 5]


def _numpy_split(buf, width):
    import numpy as np

    n = len(buf) // width
    arr = np.frombuffer(buf, dtype=np.uint8, count=n * width).reshape(n, width)
    return [np.ascontiguousarray(arr[:, j]).tobytes() for j in range(width)]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("tail", TAILS)
def test_accel_matches_numpy_reference(width, tail):
    buf = os.urandom(777 * width + tail)
    n = len(buf) // width
    body = buf[: n * width]

    planes = [bytes(p) for p in _accel.split_planes(buf, width)]
    assert planes == _numpy_split(body, width)

    # round-trip through the accel join, including the unaligned tail
    rebuilt = _accel.join_planes(planes, width, buf[n * width :])
    assert rebuilt == buf


def test_native_built_flag_is_bool():
    assert isinstance(_accel.HAVE_NATIVE, bool)


@pytest.mark.parametrize("width", WIDTHS)
def test_transforms_delegates_consistently(width):
    buf = os.urandom(1024 * width + 3)
    n = len(buf) // width
    planes = transforms.split_planes(buf, width)
    out = transforms.join_planes(planes, width, buf[n * width :])
    assert out == buf


def test_empty_and_single():
    for width in WIDTHS:
        assert all(len(bytes(p)) == 0 for p in _accel.split_planes(b"", width))
        one = os.urandom(width)
        planes = _accel.split_planes(one, width)
        assert _accel.join_planes(planes, width) == one


@pytest.mark.skipif(not _accel.HAVE_NATIVE, reason="native extension not built")
def test_codec_roundtrip_uses_native():
    """End-to-end codec round-trip while the native path is live."""
    import numpy as np

    data = (
        np.random.default_rng(0).integers(0, 256, 100_000, dtype=np.uint8)
    ).tobytes()
    for dtype in ("uint16", "uint32", "fp32"):
        blob = z4ai.compress(data, dtype=dtype)
        assert bytes(z4ai.decompress(blob)) == data
