<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quickstart

## Compress a buffer

```python
import z4ai

# `dtype` tells z4ai the element width so it can split the float planes.
blob = z4ai.compress(weights_bytes, dtype="bf16")   # -> bytes, self-describing
data = z4ai.decompress(blob)                          # -> bytes, byte-identical
assert data == weights_bytes
```

The frame is self-describing: `decompress` needs no side-channel metadata, and the
output is never larger than the input.

## Compress a NumPy array (dtype + shape preserved)

```python
import numpy as np
import z4ai

w    = np.random.randn(1024, 1024).astype(np.float16)
blob = z4ai.compress_array(w)
back = z4ai.decompress_array(blob)
assert np.array_equal(w, back) and back.dtype == w.dtype
```

## Store a sequence of checkpoints as deltas

The single largest win: consecutive checkpoints (fine-tuning, RLHF, continued
pre-training) are ~95-99% identical, so each is stored as a tiny delta.

```python
import z4ai

# store checkpoint N as the bit-exact delta from checkpoint N-1
delta    = z4ai.compress_delta(step_2000, reference=step_1000, dtype="bf16")
restored = z4ai.decompress_delta(delta, reference=step_1000)
assert bytes(restored) == step_2000        # lossless
```

## The three mental tiers of the API

| You have... | Use | Notes |
|-------------|-----|-------|
| a raw buffer or NumPy array | `compress` / `decompress`, `compress_array` / `decompress_array` | the core codec |
| a checkpoint similar to one you already stored | `compress_delta` / `decompress_delta` (aligned buffers) or `z4ai.model_delta` (`.safetensors`, tensor-aware) | stores only what changed |
| a `.safetensors` file you want smaller, with per-tensor random access | `z4ai.safetensors` (`compress_file`, `ZstnReader`) | keeps lazy single-tensor loads |

See {doc}`usage` for the detailed recipes and {doc}`cli` for the command line.
