<!-- SPDX-License-Identifier: Apache-2.0 -->

# How it works

A float is `[ sign | exponent | mantissa ]`. In trained weights the **exponent
bits repeat heavily** (values cluster near zero) while the **mantissa bits look
like noise**, and the two are interleaved byte-by-byte. A general-purpose zip
can't separate the signal from the noise, so plain `zstd` on a raw fp32 tensor
barely reaches ~1.06x.

z4ai pulls the bytes apart, matches redundancy across the whole tensor, then
entropy-codes each part near its floor:

```text
            float tensor bytes (bf16 / fp16 / fp32 / fp64)
                              |
                  split by dtype (plane / bit-field)
                   /                              \
        exponent / sign plane                mantissa plane
           (low entropy)                       (noise-like)
                |                                   |
        entropy coding                       store verbatim, or zstd
        (rANS / zstd)                              |
                \                                  /
            whole-tensor long-distance matching
           (dedups tied / repeated weights)
                              |
              best-of selection: keep the smallest
              (never larger than the input)
                              |
            self-describing container (.z4ai / .zstn)
```

Decoding is the exact inverse, driven entirely by the self-describing header - no
side-channel metadata. Pruned weights take a zero-aware path (a nonzero bitmap +
densified values); the safetensors/ZSTN container adds a per-tensor index for
random-access reads and stores tied weights once.

## Why dense weights hit a wall

A trained float tensor decomposes into three fields with very different
compressibility:

- **Mantissa** - essentially random, incompressible.
- **Sign** - close to a fair coin (~1 bit).
- **Exponent** - the only compressible field; for real weights it carries only
  **~2.6 bits** of entropy.

That bounds *any* lossless codec at about **1.51x for BF16** / **~1.2x for FP32**.
ZipNN already reaches that wall, so z4ai cannot meaningfully out-*ratio* it on a
dense checkpoint - it wins only by a hair, via an order-1 context exponent model
(rANS) that reaches below the order-0 floor a plain Huffman coder stops at
(distilgpt2 +1.6-8.2%, pythia-70m +11-29% vs ZipNN on real models; pure synthetic
i.i.d. weights stay a tie because their exponent has no spatial correlation to
exploit).

## Where the real wins come from

The large ratios come from redundancy the entropy bound assumes away:

- **Reduced effective precision** - fp32 files carrying fp16/bf16-origin values
  have dead low mantissa bits; the byte-plane transpose sends them to ~zero
  (2.3-3.0x), automatically.
- **Quantized weights in a wide container** - INT4/INT8/FP8 dequantised into
  bf16/fp16/fp32 compress via a lossless palette transform.
- **Sparsity / pruning** - a 1-bit nonzero bitmap + densified values.
- **Structure within a checkpoint** - tied embeddings, duplicated/repeated layers,
  multi-shard concatenations, caught by whole-tensor long-distance matching.
- **Similarity between checkpoints** - the biggest lever; see `compress_delta` and
  `model_delta` in {doc}`usage`.

The building blocks (float field decorrelation, LZ matching, rANS) are
well-studied; what is specific to z4ai is applying them to the byte structure of
model weights and matching across the whole tensor - and across checkpoints. See
{doc}`background`.

## Where it pays off

z4ai optimizes for **artifact size**, so it earns its keep wherever a checkpoint
is written once and read many times:

- **Model registries & hubs** - compress once, store and serve many times.
- **Long-term checkpoint archives** - training snapshots, EMA / optimizer-shadow
  copies, fine-tune lineages.
- **Distribution over metered or slow links** - bytes saved dominate transfer time.

How much you save tracks how much *structure* the weights carry: a freshly
pretrained dense layer sits near the entropy floor (≈ tie with ZipNN), while
reduced precision, quantization, sparsity, tied/duplicated tensors, and
checkpoint similarity each unlock a many-× win.
