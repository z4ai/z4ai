<!-- SPDX-License-Identifier: Apache-2.0 -->

# Usage

All examples are lossless and round-trip byte-for-byte.

## Buffers and dtypes

`dtype` selects the element width so z4ai can split the float planes. Supported
codes include `bf16`, `fp16`, `fp32`, `fp64`, and integer types (`int8`/`uint8`
... `int64`/`uint64`). `None` treats the input as opaque bytes (still compressed,
no float-plane split).

```python
import z4ai

blob = z4ai.compress(buffer, dtype="fp32")
data = z4ai.decompress(blob)                 # byte-identical to `buffer`

# Decode into a pre-allocated buffer (no extra allocation):
out = bytearray(len(data))
z4ai.decompress(blob, out=out)
```

## NumPy arrays (dtype + shape preserved)

`compress_array` records the array's dtype and shape, so the decode restores an
array identical in value, dtype, and shape — no need to pass `dtype` yourself.

```python
import numpy as np, z4ai

w    = np.random.randn(1024, 1024).astype(np.float16)
blob = z4ai.compress_array(w)
back = z4ai.decompress_array(blob)
assert np.array_equal(w, back) and back.dtype == w.dtype and back.shape == w.shape
```

## Effort tiers: speed vs size

`effort` is a single dial trading decode speed against file size. All three tiers
are lossless and decode through the same {func}`z4ai.decompress`.

```python
# fastest to decode (serve / load-often paths)
blob = z4ai.compress(buffer, dtype="bf16", effort="fast")

# balanced default
blob = z4ai.compress(buffer, dtype="bf16")

# smallest file (offline encode; for the artifact you upload)
blob = z4ai.compress(buffer, dtype="bf16", effort="max")
```

`effort="max"` adds a chunk-parallel context-modeling backend (brotli) and keeps
the smaller of it and the fast default - so it is never worse than the default.
Install it with `pip install "z4ai[maxratio]"`; without brotli, `max` falls back
to the default.

## Checkpoint deltas (aligned buffers)

When two checkpoints are positionally aligned (same shapes, same order),
`compress_delta` stores only the bytes that changed.

```python
import z4ai

blob = z4ai.compress_delta(current_ckpt_bytes, reference=prev_ckpt_bytes, dtype="bf16")
restored = z4ai.decompress_delta(blob, reference=prev_ckpt_bytes)
assert bytes(restored) == current_ckpt_bytes      # lossless
```

The reference's length and a digest are recorded in the frame, so passing the
wrong reference is rejected loudly rather than producing a corrupt result.

## Model deltas (`.safetensors`, tensor-aware)

For real `.safetensors` checkpoints whose layout may have shifted (a fine-tune
that added, dropped, renamed, or re-ordered tensors), `z4ai.model_delta` aligns
each tensor to the base tensor of the **same name and shape** before delta-coding
it - so unchanged and slightly-changed tensors still delta cleanly regardless of
how the surrounding bytes moved.

```python
from z4ai import model_delta

# in memory (.safetensors bytes)
blob     = model_delta.compress(finetuned_bytes, base_bytes)
restored = model_delta.decompress(blob, base_bytes)
assert restored == finetuned_bytes                 # lossless

# or straight from / to files
stats = model_delta.compress_file("finetuned.safetensors", "base.safetensors",
                                  "finetuned.z4dm")
print(f"{stats['ratio']:.1f}x smaller")
```

Each tensor is stored as the smallest of a zero-cost `COPY` (bit-identical to the
base), a per-tensor XOR-delta, or a from-scratch frame - so a delta archive is
never larger than compressing the target alone.

## safetensors with random access

Compress a `.safetensors` checkpoint to a smaller `.zstn` file, and still load a
single tensor lazily without decompressing the rest.

```python
from z4ai.safetensors import compress_file, decompress_file, ZstnReader

stats = compress_file("model.safetensors", "model.zstn")
print(f"{stats['saved_pct']:.0f}% smaller ({stats['ratio']:.2f}x)")

# Restore a byte-identical .safetensors file:
decompress_file("model.zstn", "model.safetensors")

# Or read just the tensors you need, lazily:
with ZstnReader.open("model.zstn") as r:
    print(r.names())                                  # list tensors, no decode
    w = r.read_numpy("model.embed_tokens.weight")     # decode only this tensor
```

The `ZSTN` container carries a per-tensor index + footer for the random-access
reads and deduplicates tied weights so they are stored once.

## High-throughput native path

For the load-often path, `z4ai.chunked` is backed by a fused, multithreaded C
codec that runs the whole split -> zstd -> interleave loop in C with the GIL
released. The frame is byte-interoperable with the default path.

```python
import z4ai.chunked as ck

blob         = ck.compress(weights, dtype="bf16")
weights_back = ck.decompress(blob)                    # byte-exact, multithreaded
```
