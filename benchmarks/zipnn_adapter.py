# SPDX-License-Identifier: Apache-2.0

"""An OPTIONAL lossless ZipNN adapter driving ZipNN's numpy/torch path.

NOTE (corrected): ZipNN's ``input_format="byte"`` path **is** lossless - but it
**mutates its input buffer in place**, so you must hand it a fresh mutable
``bytearray`` (not immutable ``bytes``) and keep a pristine copy for the equality
check. ``benchmarks/benchmark.py::run_one`` already does exactly that, so the main
benchmark's ZipNN numbers are valid. (An earlier draft of this file wrongly
claimed the byte path was non-lossless; that was a harness bug - passing ``bytes``
instead of ``bytearray``.)

This adapter is therefore not a bug fix; it is a convenience that drives ZipNN
through its **numpy** path (``input_format="numpy"``, matching element dtype:
fp32 1.172, fp16 1.124, byte-exact) and its **torch** path for bf16
(``input_format="torch"`` with a real ``torch.bfloat16`` tensor - the numpy path
raises "Support only uint32 with NumPy format"). Useful when you'd rather not rely
on in-place mutation. Returns a ``(compress, decompress, available)`` triple that
self-tests round-trip and reports ``available=False`` if ZipNN can't losslessly
handle the dtype in this environment.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

# dtype code -> (numpy dtype or None, zipnn bytearray_dtype string)
_NUMPY_DTYPE = {"fp32": np.float32, "fp16": np.float16, "fp64": np.float64}
_BF16 = "bf16"


def make_zipnn_codec(
    dtype: str,
) -> Tuple[
    Optional[Callable[[bytes], object]], Optional[Callable[[object], bytes]], bool
]:
    """Return ``(compress, decompress, available)`` for a lossless ZipNN run.

    ``compress`` takes the original raw little-endian ``bytes`` and returns an
    opaque compressed object; ``decompress`` returns the original ``bytes``,
    byte-identical. If ZipNN cannot losslessly handle ``dtype`` in this
    environment, ``available`` is ``False`` and the callables are ``None``.
    """
    try:
        import zipnn  # noqa: F401
    except Exception:  # noqa: BLE001
        return None, None, False

    if dtype in _NUMPY_DTYPE:
        npdt = _NUMPY_DTYPE[dtype]

        def comp(raw: bytes):
            import zipnn

            arr = np.frombuffer(raw, dtype=npdt)
            z = zipnn.ZipNN(input_format="numpy")
            return (z.compress(arr.copy()), len(raw))

        def decomp(obj) -> bytes:
            import zipnn

            blob, _ = obj
            z = zipnn.ZipNN(input_format="numpy")
            out = z.decompress(blob)
            return out.tobytes() if hasattr(out, "tobytes") else bytes(out)

        if _selftest(comp, decomp, dtype):
            return comp, decomp, True
        return None, None, False

    if dtype == _BF16:
        # bf16 requires the torch path; only offer it if torch is importable.
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return None, None, False

        def comp_bf16(raw: bytes):
            import torch, zipnn

            u16 = np.frombuffer(raw, dtype=np.uint16)
            t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
            z = zipnn.ZipNN(input_format="torch", bytearray_dtype="bfloat16")
            return (z.compress(t), len(raw))

        def decomp_bf16(obj) -> bytes:
            import torch, zipnn

            blob, _ = obj
            z = zipnn.ZipNN(input_format="torch", bytearray_dtype="bfloat16")
            t = z.decompress(blob)
            return t.view(torch.uint16).numpy().tobytes()

        if _selftest(comp_bf16, decomp_bf16, dtype):
            return comp_bf16, decomp_bf16, True
        return None, None, False

    return None, None, False


def _selftest(comp, decomp, dtype: str) -> bool:
    """Confirm the chosen ZipNN path is byte-exact on a small sample."""
    try:
        rng = np.random.default_rng(0)
        f = rng.standard_normal(512).astype(np.float32) * 0.02
        if dtype == "fp32":
            raw = f.astype("<f4").tobytes()
        elif dtype == "fp16":
            raw = f.astype("<f2").tobytes()
        elif dtype == "fp64":
            raw = f.astype("<f8").tobytes()
        elif dtype == "bf16":
            u32 = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
            bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
            raw = ((u32 + bias) >> 16).astype("<u2").tobytes()
        else:
            return False
        return decomp(comp(raw)) == raw
    except Exception:  # noqa: BLE001
        return False
