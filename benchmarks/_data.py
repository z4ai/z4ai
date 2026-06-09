# SPDX-License-Identifier: Apache-2.0

"""Synthetic model-weight generators with realistic byte distributions.

Real transformer weights are roughly Normal with a small standard deviation, so
the float exponent field is highly concentrated (low entropy) while the mantissa
is close to uniform. Reproducing that distribution is what makes a benchmark
representative - uniform-random bytes would understate every codec's ratio.

Shared by ``tests/`` and ``benchmarks/`` so both exercise identical inputs.
"""
from __future__ import annotations

import numpy as np


def f32_to_bf16_bytes(f32: np.ndarray) -> bytes:
    """Convert float32 -> bfloat16 raw bytes (little-endian, round-to-nearest-even).

    NumPy has no native bfloat16, so we truncate the top 16 bits of the float32
    bit pattern with proper RNE rounding - exactly what hardware bf16 casts do.
    """
    u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even bias: 0x7FFF + (LSB of the surviving mantissa)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    u16 = ((u32 + bias) >> 16).astype(np.uint16)
    return u16.astype("<u2").tobytes()


def make_weights(
    n_elements: int = 4_000_000,
    dtype: str = "bf16",
    seed: int = 0,
    mixture: bool = True,
) -> bytes:
    """Return ``n_elements`` of synthetic weights encoded as ``dtype`` raw bytes.

    ``mixture=True`` blends several Gaussian scales (like different layers /
    tensors in a real checkpoint) so the exponent distribution is realistic
    rather than a single sharp spike.
    """
    rng = np.random.default_rng(seed)
    if mixture:
        scales = np.array([0.006, 0.02, 0.05, 0.15], dtype=np.float32)
        which = rng.integers(0, len(scales), size=n_elements)
        f32 = (rng.standard_normal(n_elements).astype(np.float32) * scales[which])
    else:
        f32 = rng.standard_normal(n_elements).astype(np.float32) * np.float32(0.02)

    if dtype == "bf16":
        return f32_to_bf16_bytes(f32)
    if dtype == "fp16":
        return f32.astype(np.float16).astype("<f2").tobytes()
    if dtype == "fp32":
        return f32.astype("<f4").tobytes()
    if dtype == "fp64":
        return f32.astype("<f8").tobytes()
    raise ValueError(f"unsupported dtype for synthetic data: {dtype!r}")


def bytes_per_element(dtype: str) -> int:
    return {"bf16": 2, "fp16": 2, "fp32": 4, "fp64": 8}[dtype]


def make_scenario(scenario: str, n_elements: int, dtype: str = "bf16", seed: int = 0) -> bytes:
    """Synthetic weights with a chosen *structure*, to probe codec behaviour.

    Scenarios model real-checkpoint phenomena that distinguish whole-plane Zstd
    (z4ai) from 256 KB-chunk Huffman (ZipNN):

    * ``"iid"``        - independent Gaussian mixture (baseline; codecs tie).
    * ``"structured"`` - a base block repeated across the tensor (tied embeddings,
      duplicated/low-rank layers, padding). Long-range redundancy -> favours z4ai.
    * ``"sparse"``     - 50% exact zeros (pruned models) -> favours run-friendly codecs.
    """
    rng = np.random.default_rng(seed)
    if scenario == "iid":
        return make_weights(n_elements, dtype=dtype, seed=seed)

    if scenario == "structured":
        block = max(1, n_elements // 40)
        base = rng.standard_normal(block).astype(np.float32) * np.float32(0.02)
        reps = -(-n_elements // block)  # ceil
        f32 = np.tile(base, reps)[:n_elements]
    elif scenario == "sparse":
        f32 = rng.standard_normal(n_elements).astype(np.float32) * np.float32(0.02)
        f32[rng.random(n_elements) < 0.5] = 0.0
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if dtype == "bf16":
        return f32_to_bf16_bytes(f32)
    if dtype == "fp16":
        return f32.astype(np.float16).astype("<f2").tobytes()
    if dtype == "fp32":
        return f32.astype("<f4").tobytes()
    if dtype == "fp64":
        return f32.astype("<f8").tobytes()
    raise ValueError(f"unsupported dtype: {dtype!r}")
