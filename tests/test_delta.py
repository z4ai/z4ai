# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-checkpoint delta compression (:mod:`z4ai.delta`)."""

from __future__ import annotations

import numpy as np
import pytest

import z4ai
from z4ai import delta
from benchmarks._data import f32_to_bf16_bytes, make_weights


def _fine_tuned(base_f32: np.ndarray, frac_changed: float, seed: int) -> np.ndarray:
    """Return a copy of ``base_f32`` with a fraction of weights nudged slightly,
    mimicking one fine-tuning step (most weights unchanged, a few perturbed)."""
    rng = np.random.default_rng(seed)
    out = base_f32.copy()
    mask = rng.random(out.size) < frac_changed
    out[mask] += (rng.standard_normal(int(mask.sum())) * 1e-4).astype(np.float32)
    return out


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_delta_roundtrip_is_lossless(dtype):
    ref = make_weights(2_000_000, dtype=dtype, seed=1)
    cur = make_weights(2_000_000, dtype=dtype, seed=2)
    blob = delta.compress(cur, ref, dtype=dtype)
    assert delta.peek_is_delta(blob)
    restored = delta.decompress(blob, ref)
    assert bytes(restored) == cur


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_delta_beats_from_scratch_on_similar_checkpoints(dtype):
    """The whole point: a near-identical checkpoint stores far smaller as a delta."""
    base = (np.random.default_rng(0).standard_normal(2_000_000).astype(np.float32)
            * np.float32(0.02))
    cur_f32 = _fine_tuned(base, frac_changed=0.05, seed=7)
    if dtype == "bf16":
        ref = f32_to_bf16_bytes(base)
        cur = f32_to_bf16_bytes(cur_f32)
    else:
        ref = base.astype("<f4").tobytes()
        cur = cur_f32.astype("<f4").tobytes()

    scratch = z4ai.compress(cur, dtype=dtype)
    delta_blob = delta.compress(cur, ref, dtype=dtype)
    # 95% of weights unchanged -> delta must be dramatically smaller.
    assert len(delta_blob) < 0.5 * len(scratch)
    assert bytes(delta.decompress(delta_blob, ref)) == cur


def test_identical_checkpoint_compresses_to_almost_nothing():
    ref = make_weights(1_000_000, dtype="bf16", seed=3)
    blob = delta.compress(ref, ref, dtype="bf16")  # current == reference
    # XOR of identical buffers is all zeros -> compresses to a tiny frame.
    assert len(blob) < 0.01 * len(ref)
    assert bytes(delta.decompress(blob, ref)) == ref


def test_numpy_array_input_roundtrips():
    rng = np.random.default_rng(5)
    ref = (rng.standard_normal(500_000).astype(np.float32) * 0.02)
    cur = ref.copy()
    cur[:1000] += 1e-3
    blob = delta.compress(cur.astype("<f4"), ref.astype("<f4"), dtype="fp32")
    out = delta.decompress(blob, ref.astype("<f4"))
    assert np.array_equal(np.frombuffer(out, dtype="<f4"), cur.astype("<f4"))


def test_length_mismatch_rejected():
    a = make_weights(1000, dtype="bf16")
    b = make_weights(2000, dtype="bf16")
    with pytest.raises(ValueError):
        delta.compress(a, b, dtype="bf16")


def test_wrong_reference_rejected_at_decompress():
    ref = make_weights(500_000, dtype="bf16", seed=1)
    cur = make_weights(500_000, dtype="bf16", seed=2)
    wrong = make_weights(500_000, dtype="bf16", seed=99)
    blob = delta.compress(cur, ref, dtype="bf16")
    with pytest.raises(ValueError):
        delta.decompress(blob, wrong)  # digest mismatch must be caught


def test_empty_buffer():
    blob = delta.compress(b"", b"", dtype="bf16")
    assert bytes(delta.decompress(blob, b"")) == b""
