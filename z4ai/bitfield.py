# SPDX-License-Identifier: Apache-2.0

"""Bit-exact float *field* splitting - z4ai's ratio edge over byte-grouping.

ZipNN (and :mod:`z4ai.transforms`) split a tensor into *byte* planes.  That is
already much better than a raw Zstd pass, but it is coarse: a byte boundary
rarely lines up with the sign/exponent/mantissa field boundaries of a float.

For ``float32`` the damage is concrete.  Its layout is::

    [ s | eeeeeeee | mmmmmmmmmmmmmmmmmmmmmmm ]
      1      8                 23                (bits, MSB-first)

Little-endian byte planes therefore come out as:

    byte3 = s + 7 high exponent bits      (clean-ish)
    byte2 = 1 low exponent bit + 7 high mantissa bits   (CONTAMINATED)
    byte1 = 8 mantissa bits               (noise)
    byte0 = 8 mantissa bits               (noise)

The exponent - the most compressible field - is *split across two planes* and
plane ``byte2`` is polluted with random mantissa bits.  The entropy backend
cannot fully exploit the exponent's low entropy.

Field splitting fixes this exactly: we pull the sign bit, the exponent integer,
and the mantissa integer into three separate, statistically homogeneous
streams using vectorized NumPy bit ops:

    * **sign**     - 1 bit / element, bit-packed (``np.packbits``)
    * **exponent** - the raw exponent integer in its own ``uint8`` plane
                     (every float format in use has <= 8 exponent bits)
    * **mantissa** - the mantissa integer, byte-split into its minimal planes

The exponent stream is now *pure*, so a general compressor models it far better
than it can model a contaminated byte plane.  This is the same idea ZipNN uses,
taken one resolution finer.  The split is fully reversible and byte-exact.

The module is deliberately self-contained (only depends on NumPy) so the core
codec can adopt it as a high-ratio mode without coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Optional SIMD byte-shuffle accelerator (shared with z4ai.transforms).  It de-
# interleaves / interleaves byte planes at memory bandwidth; the bf16 fast paths
# below use it to avoid NumPy's slow strided [:, j] access.  Import is always
# safe - `_accel` transparently falls back to NumPy when the C extension is
# absent, so the bf16 fast paths stay correct everywhere, just slower.
try:
    from . import _accel
except Exception:  # noqa: BLE001 - never let an accel import break the codec
    _accel = None

# ---------------------------------------------------------------------------
# Format table.  ``bfloat16`` has no native NumPy dtype, so it is addressed by
# name and handled as a raw ``uint16`` bit pattern.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatSpec:
    """Bit geometry of an IEEE-like floating point format."""

    name: str
    total_bits: int
    exp_bits: int
    mantissa_bits: int
    uint_dtype: str  # NumPy unsigned integer view used for bit ops

    @property
    def sign_bits(self) -> int:
        return self.total_bits - self.exp_bits - self.mantissa_bits  # always 1

    @property
    def itemsize(self) -> int:
        return self.total_bits // 8

    @property
    def mantissa_bytes(self) -> int:
        """Minimal whole bytes needed to hold the mantissa integer."""
        return (self.mantissa_bits + 7) // 8


SPECS: Dict[str, FloatSpec] = {
    "float32": FloatSpec("float32", 32, 8, 23, "uint32"),
    "float16": FloatSpec("float16", 16, 5, 10, "uint16"),
    "bfloat16": FloatSpec("bfloat16", 16, 8, 7, "uint16"),
}

# Aliases accepted by :func:`resolve_spec`.
_ALIASES = {
    "fp32": "float32",
    "f32": "float32",
    "float": "float32",
    "fp16": "float16",
    "f16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
}


def resolve_spec(dtype) -> FloatSpec:
    """Resolve a NumPy dtype / string into a :class:`FloatSpec`.

    Raises:
        KeyError: if the format is not a supported float layout.
    """
    if isinstance(dtype, FloatSpec):
        return dtype
    key = str(getattr(dtype, "name", dtype)).lower()
    key = _ALIASES.get(key, key)
    if key not in SPECS:
        raise KeyError(
            f"unsupported float format {dtype!r}; "
            f"supported: {sorted(SPECS)} (+ aliases)"
        )
    return SPECS[key]


# ---------------------------------------------------------------------------
# Core split / join
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Container for the three field streams plus the metadata to invert them."""

    spec_name: str
    count: int          # number of elements
    sign: bytes         # bit-packed sign bits, ceil(count/8) bytes
    exponent: bytes     # count bytes, one exponent integer each
    mantissa: bytes     # count * mantissa_bytes bytes, byte-split planes

    def as_streams(self) -> Dict[str, bytes]:
        return {"sign": self.sign, "exponent": self.exponent, "mantissa": self.mantissa}


def _to_uint(raw: bytes, spec: FloatSpec) -> np.ndarray:
    """Interpret ``raw`` little-endian bytes as the format's unsigned integers."""
    arr = np.frombuffer(raw, dtype=f"<{np.dtype(spec.uint_dtype).char}")
    return arr


def split_fields(data, dtype=None) -> SplitResult:
    """Split float ``data`` into sign / exponent / mantissa streams.

    Args:
        data: a NumPy array of a supported float dtype, **or** raw
            little-endian ``bytes``/``bytearray`` (in which case ``dtype`` must
            be given, e.g. ``"bfloat16"``).
        dtype: format name/dtype; required when ``data`` is raw bytes, ignored
            when ``data`` is a typed NumPy float array (its dtype wins, unless
            you pass ``"bfloat16"`` explicitly for a ``uint16`` buffer).

    Returns:
        A :class:`SplitResult`.  Fully reversible via :func:`join_fields`.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        if dtype is None:
            raise ValueError("dtype is required when splitting raw bytes")
        spec = resolve_spec(dtype)
        raw = bytes(data)
    else:
        arr = np.asarray(data)
        if dtype is not None and resolve_spec(dtype).name == "bfloat16":
            spec = SPECS["bfloat16"]
        else:
            spec = resolve_spec(arr.dtype)
        raw = np.ascontiguousarray(arr).view(np.uint8).tobytes()

    itemsize = spec.itemsize
    count = len(raw) // itemsize
    aligned = count * itemsize
    if aligned != len(raw):
        raise ValueError(
            f"buffer of {len(raw)} bytes is not a whole multiple of "
            f"{itemsize}-byte {spec.name} elements"
        )

    # bf16 fast path: `_accel.bf16_split` builds all three streams (sign
    # bit-packed, exponent byte, mantissa byte) in two vectorizable C passes at
    # ~9.8 GB/s - RE-MEASURED 2.08x faster than the NumPy path below and
    # byte-exact (an earlier note claimed native was slower; that measured the
    # old single-pass scalar routine - see COORDINATION.md iter7).  `_accel`
    # transparently falls back to NumPy when the C extension is not built.
    if spec.name == "bfloat16" and _accel is not None:
        sign_packed, exp_bytes, mant_bytes = _accel.bf16_split(raw[:aligned])
        return SplitResult(
            spec_name=spec.name,
            count=count,
            sign=sign_packed,
            exponent=exp_bytes,
            mantissa=mant_bytes,
        )

    # Operate in the format's *native* unsigned width (uint16/uint32), not
    # uint64.  Every field of every supported format fits, so widening to
    # uint64 only burned memory bandwidth (8 bytes/elem instead of 2-4) and
    # made the split ~4.7x slower for byte-identical output.  Shift/mask
    # operands are plain Python ints, which NumPy treats as "weak" and so keeps
    # the array's native dtype (NEP 50).  See NOTES/profiling-throughput.md.
    u = _to_uint(raw[:aligned], spec)

    mant_mask = (1 << spec.mantissa_bits) - 1
    exp_mask = (1 << spec.exp_bits) - 1

    sign = (u >> (spec.total_bits - 1)).astype(np.uint8)
    exponent = ((u >> spec.mantissa_bits) & exp_mask).astype(np.uint8)
    mantissa = u & mant_mask

    sign_packed = np.packbits(sign).tobytes()
    exp_bytes = exponent.tobytes()

    # Byte-split the mantissa into its minimal planes: plane k holds byte k of
    # every mantissa integer, low byte first.  Homogeneous planes compress
    # marginally better and let the backend skip the noisiest one cheaply.
    mb = spec.mantissa_bytes
    if count == 0:
        mant_bytes = b""
    else:
        planes = [
            ((mantissa >> (8 * k)) & 0xFF).astype(np.uint8).tobytes()
            for k in range(mb)
        ]
        mant_bytes = b"".join(planes)

    return SplitResult(
        spec_name=spec.name,
        count=count,
        sign=sign_packed,
        exponent=exp_bytes,
        mantissa=mant_bytes,
    )


def join_fields(result: SplitResult) -> bytes:
    """Inverse of :func:`split_fields`; returns the byte-exact original buffer.

    The result is raw little-endian bytes.  For ``float32``/``float16`` you can
    wrap it back into an array with ``np.frombuffer(buf, dtype=...)``; for
    ``bfloat16`` reinterpret as ``uint16``.
    """
    spec = SPECS[result.spec_name]
    count = result.count
    if count == 0:
        return b""

    # bf16 fast path: rebuild the whole little-endian buffer in a single
    # contiguous C pass (sign-unpack + exp/mant recombine + interleave fused),
    # at memory bandwidth.  This supersedes the earlier NumPy low/high
    # construction (several full-width uint8 passes) + separate interleave that
    # was the bf16 *decompress* bottleneck.  `_accel.bf16_join` is byte-exact
    # and itself falls back to that NumPy path when the C extension is absent.
    if spec.name == "bfloat16" and _accel is not None:
        return _accel.bf16_join(result.sign, result.exponent, result.mantissa, count)

    # Reassemble in the format's *native* unsigned width (uint16/uint32), not
    # uint64 - symmetric with split_fields (see its comment).  Widening to
    # uint64 here burned 8 bytes/elem of memory bandwidth instead of 2-4 and was
    # the bf16 *decompress* bottleneck (566 MB/s vs fp16/fp32's 2.5-6.5 GB/s on
    # the byte-plane path).  Shift amounts are plain Python ints, which NumPy
    # treats as "weak" (NEP 50) and so keeps the array's native dtype; every
    # field fits in that width so no value is truncated.
    udt = np.dtype(spec.uint_dtype)

    sign = np.unpackbits(
        np.frombuffer(result.sign, dtype=np.uint8), count=count
    ).astype(udt)
    exponent = np.frombuffer(result.exponent, dtype=np.uint8, count=count).astype(udt)

    mb = spec.mantissa_bytes
    mant = np.zeros(count, dtype=udt)
    if mb:
        flat = np.frombuffer(result.mantissa, dtype=np.uint8)
        if flat.size != count * mb:
            raise ValueError("mantissa stream length does not match element count")
        for k in range(mb):
            plane = flat[k * count : (k + 1) * count].astype(udt)
            mant |= plane << (8 * k)

    u = (
        (sign << (spec.total_bits - 1))
        | (exponent << spec.mantissa_bits)
        | mant
    )
    out = u.astype(f"<{udt.char}")
    return out.tobytes()


def to_array(buf: bytes, spec_name: str) -> np.ndarray:
    """Reinterpret a reconstructed buffer as a NumPy array where possible.

    ``bfloat16`` is returned as ``uint16`` (NumPy has no native bf16).
    """
    spec = SPECS[spec_name]
    if spec_name == "bfloat16":
        return np.frombuffer(buf, dtype="<u2")
    return np.frombuffer(buf, dtype=spec_name)


__all__ = [
    "FloatSpec",
    "SplitResult",
    "SPECS",
    "resolve_spec",
    "split_fields",
    "join_fields",
    "to_array",
]
