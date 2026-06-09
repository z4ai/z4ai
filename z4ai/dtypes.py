# SPDX-License-Identifier: Apache-2.0

"""Element dtype registry.

Every value z4ai can compress is described by a small integer ``dtype`` code
that is serialized into the frame header.  The code fixes three things:

* ``itemsize`` -- bytes per element, used to align the buffer before transform;
* ``transform`` -- which decorrelating transform to apply (see
  :mod:`z4ai.transforms` and :mod:`z4ai.bitfield`);
* the human-readable string code used by the public API and CLI.

bfloat16 is deliberately handled as a *raw-buffer* dtype: NumPy has no native
bf16, so bf16 weights arrive as opaque 2-byte little-endian buffers tagged with
``dtype="bf16"``.  This keeps ``import z4ai`` free of the ``ml_dtypes`` dependency.
"""

from __future__ import annotations

from typing import Dict, Optional

# --- transform codes (serialized in the frame header) -----------------------
TRANSFORM_NONE = 0       # single opaque stream
TRANSFORM_BYTEPLANE = 1  # byte-grouping / transpose (z4ai.transforms)
TRANSFORM_BITFIELD = 2   # sign/exp/mantissa field split (z4ai.bitfield)

# --- dtype codes (serialized in the frame header) ---------------------------
RAW = 0          # opaque uint8, no transform
FP8_E4M3 = 1     # 1 byte, no transform
FP8_E5M2 = 2     # 1 byte, no transform
FP16 = 3         # IEEE binary16, byte-plane split
BF16 = 4         # bfloat16, bit-field split (the headline edge over ZipNN)
FP32 = 5         # IEEE binary32, byte-plane split
FP64 = 6         # IEEE binary64, byte-plane split
INT8 = 7         # 1 byte, no transform
INT16 = 8        # byte-plane split
INT32 = 9        # byte-plane split
INT64 = 10       # byte-plane split


class DTypeInfo:
    """Static description of one dtype code."""

    __slots__ = ("code", "name", "itemsize", "transform")

    def __init__(self, code: int, name: str, itemsize: int, transform: int):
        self.code = code
        self.name = name
        self.itemsize = itemsize
        self.transform = transform

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DTypeInfo(name={self.name!r}, itemsize={self.itemsize})"


_INFOS = [
    DTypeInfo(RAW, "uint8", 1, TRANSFORM_NONE),
    DTypeInfo(FP8_E4M3, "fp8_e4m3", 1, TRANSFORM_NONE),
    DTypeInfo(FP8_E5M2, "fp8_e5m2", 1, TRANSFORM_NONE),
    DTypeInfo(FP16, "fp16", 2, TRANSFORM_BITFIELD),
    DTypeInfo(BF16, "bf16", 2, TRANSFORM_BITFIELD),
    DTypeInfo(FP32, "fp32", 4, TRANSFORM_BITFIELD),
    DTypeInfo(FP64, "fp64", 8, TRANSFORM_BYTEPLANE),
    DTypeInfo(INT8, "int8", 1, TRANSFORM_NONE),
    DTypeInfo(INT16, "int16", 2, TRANSFORM_BYTEPLANE),
    DTypeInfo(INT32, "int32", 4, TRANSFORM_BYTEPLANE),
    DTypeInfo(INT64, "int64", 8, TRANSFORM_BYTEPLANE),
]

BY_CODE: Dict[int, DTypeInfo] = {i.code: i for i in _INFOS}
BY_NAME: Dict[str, DTypeInfo] = {i.name: i for i in _INFOS}

# Float dtype code -> spec name understood by :mod:`z4ai.bitfield`.  Only the
# field-split (TRANSFORM_BITFIELD) dtypes appear here.
BITFIELD_SPEC_NAME: Dict[int, str] = {
    FP16: "float16",
    BF16: "bfloat16",
    FP32: "float32",
}

# A few friendly aliases accepted from the public API / CLI.
_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "double": "fp64",
    "u8": "uint8",
    "raw": "uint8",
    "bytes": "uint8",
}


def info_for_name(name: Optional[str]) -> DTypeInfo:
    """Resolve a dtype string (or ``None``) to a :class:`DTypeInfo`.

    ``None`` and unknown-but-aliased names resolve to the opaque ``uint8`` type.
    Unknown names raise ``ValueError`` so typos surface loudly.
    """
    if name is None:
        return BY_NAME["uint8"]
    key = name.lower()
    key = _ALIASES.get(key, key)
    if key not in BY_NAME:
        raise ValueError(
            f"unknown dtype {name!r}; known: {sorted(BY_NAME)} "
            f"(aliases: {sorted(_ALIASES)})"
        )
    return BY_NAME[key]


def info_for_code(code: int) -> DTypeInfo:
    """Resolve a serialized dtype code to a :class:`DTypeInfo`."""
    if code not in BY_CODE:
        raise ValueError(f"unknown dtype code {code}")
    return BY_CODE[code]


# NumPy dtype name -> z4ai dtype string, for ndarray inference.  bf16 is absent
# on purpose (NumPy has no native bfloat16).
NUMPY_NAME_TO_CODE = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "uint16": "int16",  # treated as 2-byte byte-plane stream
    "int32": "int32",
    "uint32": "int32",
    "int64": "int64",
    "uint64": "int64",
}
