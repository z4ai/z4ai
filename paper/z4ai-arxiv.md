---
title: "z4ai: Lossless, Random-Access Compression for Neural-Network Model Checkpoints"
author:
  - Hyukjin Kwon
date: June 2026
bibliography: paper.bib
abstract: |
  The storage and distribution of neural-network checkpoints has become a
  first-order systems cost, yet general-purpose compressors handle weight tensors
  poorly and weight-specific codecs leave large, practically important sources of
  redundancy untouched. We present z4ai, a lossless codec for model checkpoints
  that operates on a single file and preserves per-tensor random access. We first
  make explicit the information-theoretic ceiling that binds *every* lossless codec
  on dense trained weights — the exponent of a trained float carries only
  ~2.6 bits and the mantissa is near-random, capping ratios at ~1.51x for bf16 and
  ~1.2x for fp32 — and show that z4ai ties the state of the art (ZipNN) there, as
  it must. The contribution is not a new entropy coder but the integration of
  whole-tensor long-distance matching, a lossless palette transform for quantized
  weights shipped in wide float containers, and a single-reference cross-checkpoint
  delta into one self-describing, random-access container that is never worse than
  plain zstd. On the redundancy regimes the entropy bound assumes away, this yields
  large lossless gains over ZipNN: 4.72x vs 1.94x on INT8-in-fp32, 10.8x vs 4.6x on
  INT4-in-fp32, 2.93x vs 1.67x on a realistic structured checkpoint, and ~30–184x
  on cross-checkpoint deltas. z4ai is complementary to repository-scale storage
  systems such as ZipLLM, targeting the pipelines and object stores that have no
  corpus-wide deduplication service to build on.
---

# Introduction

Model registries, fine-tune families, and training runs produce large numbers of
checkpoints that are highly *related* to one another, and storing or transferring
them at full size is increasingly expensive. General-purpose compressors are a poor
fit: in an IEEE-754 float the low-entropy exponent and the high-entropy mantissa are
interleaved byte-by-byte, so a generic `zstd` pass over raw `fp32` weights reaches
only ~1.06x. Weight-specific lossless codecs — ZipNN [@zipnn], NeuZip [@neuzip],
DFloat11 [@dfloat11], DietGPU [@dietgpu], and Unweight [@unweight] — improve on this
by separating the float fields and entropy-coding the skewed exponent.

These codecs are, however, *order-0* and operate within fixed-size chunks, which
leaves two practically large sources of redundancy unaddressed. First, structure
that spans a chunk boundary or a whole tensor — tied `embed_tokens`/`lm_head`
matrices, duplicated transformer blocks, multi-shard concatenations, and pruned
layers — is invisible to a chunked, LZ-free coder. Second, the format that models
are *actually shipped in*: quantized weights (INT4/INT8/FP8 from GPTQ, AWQ, or
`compressed-tensors`) are routinely dequantized into a wide float container for
deployment, leaving only a small set of distinct values that a byte-grouping codec
cannot exploit. Finally, consecutive checkpoints in a training run differ in only a
few percent of their bytes, a redundancy exploited at *repository* scale by recent
systems such as ZipLLM [@zipllm] but absent from drop-in codecs.

We present **z4ai**, a lossless codec that targets exactly these regimes while
remaining usable as a single-file library and command-line tool with per-tensor
random-access reads. Our contributions are:

1. An explicit, reproducible statement of the **entropy ceiling** on dense trained
   weights, and a demonstration that z4ai ties ZipNN there within 0.3% — clarifying
   what no lossless codec can do.
2. The **integration** of whole-tensor matching, a lossless palette transform for
   dequantized quantized weights, and a single-reference cross-checkpoint delta into
   one self-describing container with a *never-worse-than-zstd* guarantee.
3. An **evaluation** across dense, structured, sparse, quantized, real-checkpoint,
   and checkpoint-delta workloads, fully reproducible from the open-source
   repository.

# Background: the entropy ceiling

A float is laid out as `[sign | exponent | mantissa]`. In trained weights the sign
is approximately a fair coin (~1 bit), the mantissa is near-random (entropy close to
its full width, ~7 bits for bf16), and only the exponent is strongly skewed, with a
measured entropy of roughly 2.57–2.74 bits against its allocated 8 [@dfloat11;
@ckptdelta]. Entropy-coding the exponent and storing the rest is therefore the whole
game for a *dense* tensor, and it caps the achievable lossless ratio at approximately
**1.51x for bf16 and 1.2x for fp32**, regardless of the coder. ZipNN already sits at
that wall. This is an information-theoretic bound, not an engineering gap: no
lossless codec — z4ai included — can meaningfully exceed it on dense, independently
distributed weights. Larger ratios are only available where the data carries
redundancy the bound assumes away: reduced precision, sparsity, repeated structure,
or near-identical neighbours in a checkpoint sequence. z4ai is designed entirely
around capturing that redundancy losslessly, and around *not regressing* on dense
data where none exists.

# Design

z4ai decomposes each float tensor into sign, exponent, and mantissa planes. The
low-entropy exponent and sign planes are entropy-coded with an interleaved range-ANS
coder [@rans; @ryg_rans], or with `zstd` [@zstd] when that is smaller for a given
stream; the noise-like mantissa is stored verbatim or `zstd`-compressed. A
**whole-tensor long-distance-matching** pass [@lz77] then deduplicates repeated and
tied weights across the entire buffer — the redundancy ZipNN's 256 KiB chunking
cannot see. For every stream, a **best-of selection** keeps the smallest encoding,
which guarantees the output is never larger than a plain `zstd` pass.

Three transforms extend the codec beyond dense data:

- **Palette transform.** When a tensor in a wide float container holds only a small
  set of distinct values — the signature of quantized weights dequantized for
  deployment — z4ai relabels the values to a dense index plus a codebook. The
  transform is bijective and therefore lossless, and it recovers the original low
  bit-width that the wide container wasted.
- **Sparsity path.** Pruned weights take a zero-aware encoding rather than paying to
  store runs of zero bytes.
- **Cross-checkpoint delta.** `compress_delta` / `model_delta` implement
  name-aligned, per-tensor lossless deltas (copy / XOR-delta / full) against a
  reference checkpoint, robust to reordered or added tensors, so that checkpoint *N*
  costs only the bytes that changed relative to *N−1*.

The container is self-describing — decoding is the exact inverse driven entirely by
the header, with no side-channel metadata — and carries a per-tensor index for
random-access reads, storing tied tensors once. An optional `effort="max"` tier adds
a chunk-parallel context-modeling backend [@brotli] that reaches *below* the order-0
floor on real transformer weights, falling back to the fast default on genuinely
incompressible data. General-purpose float codecs such as ALP [@alp] and Pcodec
[@pcodec] were evaluated but target smooth or decimal-origin numeric data, not the
full-entropy mantissa of trained weights, and were not adopted.

# Evaluation

All results below are measured on this repository's benchmark suite (16-core
machine, Python 3.14, `zstandard` 0.25.0, latest `zipnn`, 32 MB per dtype,
best-of-3 timing) and reproducible with a single command per row. Every z4ai output
is verified byte-exact. Compression *ratios* are load-independent and the primary
metric; throughput figures are reported for completeness and are sensitive to
machine load.

## Compression ratio

| Scenario | dtype | z4ai | ZipNN | zstd-3 |
|---|---|---:|---:|---:|
| Dense / i.i.d. | bf16 | 1.413 | **1.417** | 1.227 |
| Dense / i.i.d. | fp32 | 1.171 | **1.172** | 1.061 |
| Structured (repeated) | bf16 | **58.1** | 1.51 | 16.97 |
| Structured (repeated) | fp32 | **47.3** | 1.20 | 14.24 |
| Sparse (50% zeros) | bf16 | **2.47** | 2.20 | 1.88 |
| Sparse (50% zeros) | fp32 | **2.21** | 1.86 | 1.79 |
| Quantized INT8 → wide | bf16 | **2.39** | 2.07 | 1.79 |
| Quantized INT8 → wide | fp32 | **4.72** | 1.94 | 3.07 |
| Quantized INT4 → wide | bf16 | **5.41** | 3.87 | 3.91 |
| Quantized INT4 → wide | fp32 | **10.77** | 4.59 | 5.13 |

On dense, i.i.d. weights z4ai ties ZipNN within 0.3%, as the entropy bound requires.
The large gains appear exactly where redundancy exists. (Caveat: on the quantized
rows several ZipNN entries were *not* byte-exact on the tested build; the palette
benchmark additionally reports a stronger lossless `zstd-19` baseline, against which
z4ai's margin is more conservative than the table above.)

## Real and production checkpoints

| Workload | z4ai | ZipNN | Note |
|---|---:|---:|---|
| bert-tiny, 17.7 MB fp32 (downloaded) | 1.188 | **1.202** | −1.2%: a single dense checkpoint is ~i.i.d. |
| Production .safetensors, 201 MB bf16, tied embedding | **1.525** | 1.510 | +1.0%: dedups the tied `embed_tokens`/`lm_head` |
| Realistic full checkpoint, 107 MB bf16 (tied embeddings, shared blocks, optimizer state, 50%-pruned layer) | **2.93** | 1.67 | +75.7%: whole-buffer dedup of real structure |
| Checkpoint delta, bert-tiny bf16, 5% of weights changed | **51.1** | ~1.7 | ~30x smaller than from-scratch; 1% → 184x, 20% → 18x |

## Throughput

| Codec | compress (MB/s) | decompress (MB/s) |
|---|---:|---:|
| z4ai (i.i.d. bf16) | 1420 | 16700 |
| ZipNN | **8125** | **20020** |

z4ai compresses several times slower than ZipNN's compiled-C core and decompresses
competitively — a deliberate trade for the write-once, read-many lifecycle of a
stored checkpoint.

# Related work

**Weight-specific lossless codecs.** ZipNN [@zipnn] splits float fields and applies
Huffman coding to the exponent in 256 KiB chunks; it is the closest dense baseline
and, being chunked and LZ-free, is exactly why it misses cross-tensor structure.
NeuZip [@neuzip] and DFloat11 [@dfloat11] entropy-code the exponent with ANS and
dynamic-length codes respectively, primarily for memory-efficient training and GPU
inference; DietGPU [@dietgpu] provides a GPU-native rANS float coder; Unweight
[@unweight] targets MLP weights for inference. All are order-0 on the float fields
and confirm the entropy decomposition of Section 2.

**Repository-scale storage.** ZipLLM [@zipllm] is the closest *system*: it
deduplicates tensors across an entire model corpus using content-defined chunking
and model-family clustering, and stores fine-tunes as lossless XOR deltas against
their base, reporting large whole-corpus savings. z4ai is **complementary**, not
competing: ZipLLM optimizes a managed repository as a service, whereas z4ai is a
self-contained per-file codec with random-access reads, an explicit single-reference
delta, and the palette transform — for the many pipelines and object stores that
have no corpus-wide deduplication service to build on. The broader landscape of
low-precision and delta compression for weights, checkpoints, and K/V caches is
surveyed by @ckptdelta.

**General float codecs.** ALP [@alp] and Pcodec [@pcodec] are strong on smooth or
decimal-origin numeric sequences but target a different data shape than the
full-entropy mantissa of trained weights.

# Limitations

z4ai cannot exceed the lossless entropy ceiling on dense weights, and does not try
to; its wins are conditional on redundancy being present. Its pure-Python core
compresses several times slower than compiled-C codecs. As a per-file codec it does
not capture cross-model redundancy at corpus scale, where a dedicated system such as
ZipLLM is the better tool. The high-ratio `effort="max"` tier trades substantial
decode throughput for a few percent of additional ratio and is therefore opt-in.

# Conclusion

The lossless compressibility of dense trained weights is bounded by physics, and
that wall is already met by existing codecs. The practically valuable lossless wins
for model storage and distribution come instead from structure, reduced precision,
sparsity, and cross-checkpoint redundancy. z4ai packages the transforms that capture
these — whole-tensor matching, a lossless palette transform, and single-reference
deltas — into one self-describing, random-access, never-worse-than-zstd codec usable
without any repository-scale infrastructure. It is open source and every result here
is reproducible from the repository.

# References
