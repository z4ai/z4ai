# SPDX-License-Identifier: Apache-2.0
"""HuggingFace / safetensors integration for z4ai.

Two layers, smallest-footprint first:

* :func:`load_file` / :func:`save_file` - drop-in replacements for
  ``safetensors.torch.load_file`` / ``save_file`` that transparently (de)compress
  a z4ai ``ZSTN`` container.  Use these when you load checkpoints yourself.

* :func:`enable_hf` - monkey-patches ``safetensors`` so that
  ``transformers`` / ``vllm`` ``from_pretrained`` loads z4ai-compressed weights
  with no other code change.  Mirrors ZipNN's ``zipnn_safetensors()``: it must be
  called **before** ``transformers`` / ``vllm`` are imported, because those
  libraries bind ``from safetensors import safe_open`` at import time.

``torch`` and ``safetensors`` are optional dependencies; they are imported lazily
so importing :mod:`z4ai` never pulls them in.  Install with ``pip install
z4ai[hf]`` (or ``pip install torch safetensors``).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from . import safetensors as _zstn

__all__ = ["load_file", "save_file", "enable_hf", "is_enabled"]

# z4ai's ZSTN container starts with this 4-byte magic; a plain .safetensors file
# starts with a little-endian u64 header length, which never collides with it.
_MAGIC = _zstn.MAGIC


def _missing(pkg: str) -> "RuntimeError":
    return RuntimeError(
        f"z4ai HuggingFace integration needs {pkg!r}; install with "
        f"`pip install z4ai[hf]` (or `pip install torch safetensors`)."
    )


def _import_torch_st():
    try:
        import safetensors.torch as st  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - exercised via _missing
        raise _missing("safetensors") from exc
    try:
        import torch  # noqa: F401, WPS433
    except ImportError as exc:  # pragma: no cover
        raise _missing("torch") from exc
    return st


def _is_zstn_path(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def _decompressed_safetensors_bytes(path: str) -> bytes:
    """Read a file and return raw ``.safetensors`` bytes, decompressing if it is
    a ZSTN container and passing it through unchanged otherwise."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[: len(_MAGIC)] == _MAGIC:
        return _zstn.decompress_bytes(blob)
    return blob


def load_file(path: str, device: str = "cpu") -> Dict[str, Any]:
    """Load a ``ZSTN`` (or plain ``.safetensors``) file into a torch state dict.

    Drop-in for ``safetensors.torch.load_file``: a z4ai-compressed file is
    decompressed in memory first; an uncompressed ``.safetensors`` file is loaded
    directly.
    """
    st = _import_torch_st()
    raw = _decompressed_safetensors_bytes(path)
    tensors = st.load(raw)  # bytes -> {name: cpu tensor}
    if device != "cpu":
        tensors = {k: v.to(device) for k, v in tensors.items()}
    return tensors


def save_file(
    tensors: Dict[str, Any],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
    *,
    level: Optional[int] = None,
) -> dict:
    """Serialize a torch state dict and compress it to a ``ZSTN`` file.

    Inverse of :func:`load_file`; the file round-trips byte-identically through
    ``safetensors`` on decompress.  Returns the compression stats.
    """
    st = _import_torch_st()
    raw = st.save(tensors, metadata=metadata)  # state dict -> .safetensors bytes
    frame = _zstn.compress_bytes(raw, level=level)
    with open(path, "wb") as fh:
        fh.write(frame)
    orig, comp = len(raw), len(frame)
    return {
        "original_bytes": orig,
        "compressed_bytes": comp,
        "ratio": orig / comp if comp else float("inf"),
        "saved_pct": (1 - comp / orig) * 100 if orig else 0.0,
    }


# --- transparent from_pretrained patch ------------------------------------

# Saved originals so enable_hf() is idempotent and reversible.
_orig: Dict[str, Any] = {}


class _TempBackedHandle:
    """Wraps a real ``safe_open`` handle opened on a decompressed temp file.

    Delegates every attribute to the real handle (so ``keys`` / ``get_tensor`` /
    ``get_slice`` / ``metadata`` behave exactly as safetensors does) and deletes
    the temp file when the context closes.
    """

    def __init__(self, real, tmp_path: str) -> None:
        self._real = real
        self._tmp_path = tmp_path

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._real.__exit__(*exc)
        finally:
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass


def _make_patched_safe_open(orig_safe_open):
    def _patched_safe_open(filename, framework="pt", device="cpu", **kwargs):
        if _is_zstn_path(str(filename)):
            raw = _decompressed_safetensors_bytes(str(filename))
            fd, tmp = tempfile.mkstemp(suffix=".safetensors")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
            except BaseException:
                os.unlink(tmp)
                raise
            real = orig_safe_open(tmp, framework=framework, device=device, **kwargs)
            return _TempBackedHandle(real, tmp)
        return orig_safe_open(filename, framework=framework, device=device, **kwargs)

    return _patched_safe_open


def _make_patched_load_file(orig_load_file):
    def _patched_load_file(filename, device="cpu"):
        if _is_zstn_path(str(filename)):
            return load_file(str(filename), device=device)
        return orig_load_file(filename, device=device)

    return _patched_load_file


def is_enabled() -> bool:
    """True if :func:`enable_hf` has patched safetensors in this process."""
    return bool(_orig)


def enable_hf() -> None:
    """Patch ``safetensors`` so ``from_pretrained`` transparently loads z4ai files.

    Compress your weights to ``ZSTN`` and name them with a variant suffix
    (``model.z4ai.safetensors``), then::

        from z4ai.integrations import enable_hf
        enable_hf()                                  # before importing transformers
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, variant="z4ai")

    Idempotent; call once at process start, before importing transformers/vllm.
    """
    if _orig:
        return
    try:
        import safetensors  # noqa: WPS433
        import safetensors.torch as st  # noqa: WPS433
    except ImportError as exc:
        raise _missing("safetensors") from exc

    _orig["safe_open"] = safetensors.safe_open
    _orig["torch.safe_open"] = st.safe_open
    _orig["torch.load_file"] = st.load_file

    patched_open = _make_patched_safe_open(_orig["safe_open"])
    patched_load = _make_patched_load_file(_orig["torch.load_file"])

    safetensors.safe_open = patched_open
    st.safe_open = patched_open
    st.load_file = patched_load


def disable_hf() -> None:
    """Undo :func:`enable_hf`."""
    if not _orig:
        return
    import safetensors
    import safetensors.torch as st

    safetensors.safe_open = _orig["safe_open"]
    st.safe_open = _orig["torch.safe_open"]
    st.load_file = _orig["torch.load_file"]
    _orig.clear()
