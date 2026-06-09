# SPDX-License-Identifier: Apache-2.0

"""Whole-checkpoint, name-aligned delta compression (z4ai.model_delta)."""

import numpy as np
import pytest

from z4ai import model_delta
from z4ai import safetensors as zst

from test_safetensors import _build_safetensors  # benchmarks/ + tests/ on sys.path


def _base_model(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "embed": (rng.standard_normal((512, 128)).astype(np.float32) * 0.02),
        "layer.0.w": (rng.standard_normal((128, 256)).astype(np.float32) * 0.02),
        "layer.0.b": (rng.standard_normal(256).astype(np.float32) * 0.02),
        "layer.1.w": (rng.standard_normal((256, 128)).astype(np.float32) * 0.02),
    }


def test_reupload_is_byte_identical_and_tiny():
    base = _build_safetensors(_base_model())
    blob = model_delta.compress(base, base)  # target == base
    assert model_delta.decompress(blob, base) == base
    # An identical re-upload is almost entirely COPY records -> far smaller than
    # compressing the model standalone.
    assert len(blob) < len(zst.compress_bytes(base)) // 2


def test_finetune_roundtrip_and_beats_full():
    tensors = _base_model()
    base = _build_safetensors(tensors)
    # "fine-tune": nudge a few tensors, leave the (large) embedding untouched.
    ft = dict(tensors)
    rng = np.random.default_rng(7)
    ft["layer.0.w"] = (
        ft["layer.0.w"]
        + rng.standard_normal(ft["layer.0.w"].shape).astype(np.float32) * 0.001
    )
    ft["layer.1.w"] = (
        ft["layer.1.w"]
        + rng.standard_normal(ft["layer.1.w"].shape).astype(np.float32) * 0.001
    )
    target = _build_safetensors(ft)

    blob = model_delta.compress(target, base)
    assert model_delta.decompress(blob, base) == target
    # The unchanged embedding (the bulk) costs ~nothing, so the delta must beat a
    # from-scratch compression of the target.
    assert len(blob) < len(zst.compress_bytes(target))


def test_added_removed_reordered_tensors_align_by_name():
    base = _build_safetensors(_base_model())
    tensors = _base_model()
    # Drop one tensor, add a new one, and rely on name (not offset) alignment.
    del tensors["layer.0.b"]
    tensors["layer.2.new"] = (
        np.random.default_rng(3).standard_normal((64, 64)).astype(np.float32) * 0.02
    )
    # Reorder dict insertion to shift byte offsets vs base.
    target = _build_safetensors({k: tensors[k] for k in reversed(list(tensors))})

    blob = model_delta.compress(target, base)
    assert model_delta.decompress(blob, base) == target


def test_wrong_base_is_rejected():
    base = _build_safetensors(_base_model(seed=0))
    other = _build_safetensors(_base_model(seed=99))
    blob = model_delta.compress(base, base)
    with pytest.raises(ValueError, match="base checkpoint does not match"):
        model_delta.decompress(blob, other)


def test_bf16_finetune_roundtrip():
    # bf16 path: pass uint16-backed tensors via the ("BF16", arr) convention.
    rng = np.random.default_rng(0)
    base_arr = rng.integers(0, 65536, size=(256, 128), dtype=np.uint16)
    base = _build_safetensors({"w": ("BF16", base_arr)})
    ft_arr = base_arr.copy()
    flip = rng.random(ft_arr.shape) < 0.02
    ft_arr[flip] ^= 1  # flip a low bit in 2% of elements
    target = _build_safetensors({"w": ("BF16", ft_arr)})

    blob = model_delta.compress(target, base)
    assert model_delta.decompress(blob, base) == target
    assert len(blob) < len(zst.compress_bytes(target))


def test_file_roundtrip(tmp_path):
    base = _build_safetensors(_base_model())
    target = _build_safetensors(_base_model(seed=1))
    bp = tmp_path / "base.safetensors"
    tp = tmp_path / "target.safetensors"
    dp = tmp_path / "delta.z4dm"
    op = tmp_path / "restored.safetensors"
    bp.write_bytes(base)
    tp.write_bytes(target)
    stats = model_delta.compress_file(str(tp), str(bp), str(dp))
    assert stats["stored"] == dp.stat().st_size
    model_delta.decompress_file(str(dp), str(bp), str(op))
    assert op.read_bytes() == target
