<!-- SPDX-License-Identifier: Apache-2.0 -->

# z4ai

<p class="z4ai-hero">
A <span class="z4ai-grad">lossless</span> storage-and-distribution layer for AI
model checkpoints — bit-for-bit reversible, with per-tensor random access. Most
useful on <strong>collections of related checkpoints</strong> (training runs,
fine-tune families, model registries) and in environments the Hugging Face Hub's
Xet backend doesn't cover: self-hosted registries, internal MLOps, plain object
storage.
</p>

```python
import z4ai

blob = z4ai.compress(weights_bytes, dtype="bf16")   # smaller, self-describing
data = z4ai.decompress(blob)                          # byte-identical original
assert data == weights_bytes
```

It is a small, pure-Python package (NumPy + `zstandard`; no PyTorch, no compiled
toolchain required) with optional native acceleration.

::::{grid} 1 1 2 2
:gutter: 3
:margin: 2 0 0 0

:::{grid-item-card} 🚀 Install
:link: installation
:link-type: doc
`pip install z4ai` and you're ready — pure-Python, no build step.
:::

:::{grid-item-card} ⚡ Quickstart
:link: quickstart
:link-type: doc
Compress a buffer, an ndarray, or a whole `.safetensors` file in a few lines.
:::

:::{grid-item-card} 📦 Usage
:link: usage
:link-type: doc
Files, per-tensor random access, sparse / quantized weights, checkpoint deltas.
:::

:::{grid-item-card} 🧠 How it works
:link: howitworks
:link-type: doc
Field decorrelation, whole-tensor matching, and the best-of candidate selector.
:::

:::{grid-item-card} 🖥️ CLI
:link: cli
:link-type: doc
`z4ai compress / decompress / info` — pipe-friendly, self-describing frames.
:::

:::{grid-item-card} 📚 API reference
:link: api
:link-type: doc
Every public function, generated from the source docstrings.
:::

::::

## Honest framing

```{admonition} What lossless can — and cannot — do for weights
:class: note

On a *dense* checkpoint a trained float's mantissa is near-random and its
exponent carries only **~2.6 bits**, capping *any* lossless codec at **~1.5×**
(bf16) / **~1.2×** (fp32). z4ai cannot meaningfully out-*ratio* that wall on dense
weights — and says so.
```

The large wins come from redundancy the entropy bound assumes away, which z4ai
auto-detects and exploits:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Reduced precision
fp32 files carrying fp16/bf16-origin values have dead low mantissa bits →
**2.3–3.0×**, automatically.
:::

:::{grid-item-card} Quantized weights
INT4/INT8/FP8 dequantised into a wide container → a lossless palette transform →
**2.4–10.8×** (the common *deployed* format).
:::

:::{grid-item-card} Structure & sparsity
Tied embeddings, duplicated layers, pruned zeros → whole-tensor long-distance
matching and a zero-aware path.
:::

:::{grid-item-card} Checkpoint deltas
Consecutive checkpoints are ~95–99% identical → store each as a tiny delta →
**10–180×**.
:::

::::

See {doc}`howitworks` for the mechanism and {doc}`background` for prior art.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

installation
quickstart
usage
cli
howitworks
background
api
```
