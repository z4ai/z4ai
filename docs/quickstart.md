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

## Shrink a `.safetensors` file, keep per-tensor random access

`z4ai.safetensors` compresses a `.safetensors` file into a `ZSTN` container that
stays byte-identical on the round trip - but unlike a plain zip, it keeps a
per-tensor index, so you can lazily read a single tensor without touching the
rest of the file (the same access pattern `safetensors` gives you).

```python
from z4ai import safetensors

# whole-file: ZSTN is byte-identical to the original .safetensors on decompress
stats = safetensors.compress_file("model.safetensors", "model.zstn")
print(f"{stats['ratio']:.2f}x  ({stats['saved_pct']:.1f}% smaller)")
safetensors.decompress_file("model.zstn", "restored.safetensors")

# random access: read just one tensor, seeking past the rest of the file
with safetensors.ZstnReader.open("model.zstn") as r:
    print(r.names()[:3])                              # tensor names, no decode
    w = r.read_numpy("model.embed_tokens.weight")     # decodes only this tensor
```

## From the command line

The CLI is the generic byte codec (one frame in, one frame out). For the
tensor-aware `ZSTN` container with per-tensor random access, use the
`z4ai.safetensors` API above.

```bash
z4ai compress   weights.bin  -o weights.z4ai
z4ai decompress weights.z4ai -o weights.bin
z4ai info       weights.z4ai          # ratio + per-plane breakdown
```

See {doc}`cli` for every flag (effort tiers, threads, dtype).

## Load into PyTorch / transformers

With the `hf` extra (`pip install "z4ai[hf]"`), load compressed weights straight
into a torch state dict, or patch `transformers` to do it transparently:

```python
import z4ai

z4ai.save_file(model.state_dict(), "model.z4ai.safetensors")
state = z4ai.load_file("model.z4ai.safetensors")     # -> {name: torch.Tensor}

# ...or transparent from_pretrained (call before importing transformers):
from z4ai import enable_hf
enable_hf()
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-org/model", variant="z4ai")
```

See {doc}`usage` for the full HuggingFace integration.

## The three mental tiers of the API

| You have... | Use | Notes |
|-------------|-----|-------|
| a raw buffer or NumPy array | `compress` / `decompress`, `compress_array` / `decompress_array` | the core codec |
| a checkpoint similar to one you already stored | `compress_delta` / `decompress_delta` (aligned buffers) or `z4ai.model_delta` (`.safetensors`, tensor-aware) | stores only what changed |
| a `.safetensors` file you want smaller, with per-tensor random access | `z4ai.safetensors` (`compress_file`, `ZstnReader`) | keeps lazy single-tensor loads |

See {doc}`usage` for the detailed recipes and {doc}`cli` for the command line.
