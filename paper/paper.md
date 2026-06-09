---
title: 'z4ai: Lossless, random-access compression for neural-network model checkpoints'
tags:
  - Python
  - data compression
  - machine learning
  - model checkpoints
  - lossless
  - MLOps
authors:
  - name: Hyukjin Kwon
    orcid: 0009-0000-8876-1506
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 9 June 2026
bibliography: paper.bib
---

# Summary

`z4ai` is a lossless compression codec for neural-network model checkpoints. It
reduces the size of stored and transferred model weights bit-for-bit reversibly,
while preserving per-tensor random access so a single weight can be read back
without decompressing the whole file. Rather than treat a checkpoint as an opaque
byte stream, `z4ai` exploits the structure of IEEE-754 float tensors and of model
*collections*: it splits each float into its sign, exponent, and mantissa fields
and entropy-codes them separately [@fpzip; @rans]; it matches redundancy across the
*whole* tensor and across tensors to deduplicate tied embeddings, shared blocks, and
repeated layers [@lz77; @zstd]; it recovers the low cardinality of quantized weights
that ship dequantized into a wide float container via a bijective palette transform;
and it stores a checkpoint as a compact lossless delta against a reference checkpoint
[@zipllm]. The result is a self-describing container with no side-channel metadata,
implemented in pure Python over NumPy and `zstandard` with optional native (C) and
`brotli` acceleration, and exposed through both a Python API and a command-line tool.

# Statement of need

The storage and distribution of model weights has become a first-order systems
cost. Model registries, fine-tune families, and training runs produce many large,
highly *related* checkpoints, and general-purpose compressors handle them poorly:
the exponent and mantissa of a float are interleaved byte-by-byte, so a generic
`zstd` pass on raw `fp32` weights reaches only ~1.06x. Weight-specific codecs such
as ZipNN [@zipnn], NeuZip [@neuzip], DFloat11 [@dfloat11], DietGPU [@dietgpu], and
Unweight [@unweight] improve on this by separating the float fields and
entropy-coding the low-entropy exponent. They are, however, *order-0* and operate within fixed-size chunks. This
leaves two practically large sources of redundancy on the table.

First, **structure that spans a chunk boundary or a whole tensor** — tied
`embed_tokens`/`lm_head` matrices, duplicated transformer blocks, multi-shard
concatenations, and pruned (sparse) layers — is invisible to a chunked, LZ-free
coder. Second, the **format models are actually shipped in**: quantized weights
(INT4/INT8/FP8 produced by GPTQ, AWQ, or `compressed-tensors`) are routinely
*dequantized into a wide float container* for deployment, leaving only a small set
of distinct values that a byte-grouping codec cannot exploit. Finally, the
**lossless cross-checkpoint redundancy** that consecutive checkpoints carry — they
differ in only a few percent of their bytes — has been exploited at *repository*
scale by recent systems such as ZipLLM [@zipllm], which deduplicates tensors across
a corpus and stores fine-tunes as a lossless XOR delta against their base, and
characterized for low-precision formats by @ckptdelta. What is missing is the same
idea packaged as a *drop-in, self-describing codec with per-tensor random access*,
adoptable by a single pipeline or object store without standing up a corpus-wide
deduplication service.

`z4ai` targets exactly these cases while remaining a drop-in lossless codec. On
*dense, independently-distributed* weights it deliberately makes no claim to beat
the state of the art: a trained float's mantissa is near-random and its exponent
carries only ~2.6 bits, capping *every* lossless codec at ~1.51x for `bf16` and
~1.2x for `fp32` [@zipserv; @ecf8]. ZipNN already sits at that information-theoretic
wall, and `z4ai` ties it there (within 0.3%). The contribution is not a new
compression primitive — the field split [@zipnn; @dfloat11], whole-tensor matching
[@zstd], palette coding, and XOR delta [@zipllm] are each established. It is their
integration into a single lossless, self-describing container that selects the best
encoding per stream (never worse than plain `zstd`), preserves per-tensor random
access, and ships as a drop-in Python library and CLI rather than a corpus-scale
storage service. This puts the structural and cross-checkpoint redundancy the
entropy bound assumes away within reach of the settings where a repository-wide
deduplication system is not an option. On benchmarks reproducible from the
repository, this yields large lossless gains over ZipNN where redundancy exists:
INT8-in-`fp32` 4.72x vs 1.94x, INT4-in-`fp32` 10.8x vs 4.6x, a realistic structured
checkpoint 2.93x vs 1.67x (+76%), and a 5%-changed checkpoint delta ~30x smaller
than a from-scratch compress (1% changed → ~184x), for which ZipNN has no
equivalent. The trade is throughput: `z4ai` compresses several times slower than
ZipNN's compiled-C core (decompression is competitive), which suits the
write-once, read-many lifecycle of a stored checkpoint.

These wins do not make `z4ai` a replacement for repository-scale storage systems.
The closest such system, ZipLLM [@zipllm], uses content-defined chunking and
model-family clustering to deduplicate an entire corpus and reports substantially
higher whole-corpus savings than any per-file codec can reach — a regime `z4ai`
does not target. The two are complementary: ZipLLM optimizes a *managed
repository*, whereas `z4ai` is a self-contained *codec* — single file in,
byte-exact out, with per-tensor random-access reads, an explicit single-reference
delta, and the palette transform — for the many pipelines and object stores that
have no such repository service to build on.

`z4ai` is most useful in settings the Hugging Face Hub's Xet backend does not
cover — self-hosted registries, internal MLOps pipelines, and plain object storage —
and to researchers studying float-tensor and checkpoint-sequence compression who
need a reproducible, byte-exact baseline that captures structural redundancy.

# Design

A float tensor is decomposed into `[sign | exponent | mantissa]` planes; the
low-entropy exponent/sign planes are entropy-coded with an interleaved range-ANS
coder [@rans; @ryg_rans] (or `zstd`, whichever is smaller per stream), while the
noise-like mantissa is stored verbatim or `zstd`-compressed. A whole-tensor
long-distance-matching pass then deduplicates repeated and tied weights before a
best-of selection keeps the smallest encoding — guaranteeing the output is never
larger than a plain `zstd` pass. Pruned weights take a zero-aware path, and the
`safetensors`/container format adds a per-tensor index for random-access reads and
stores tied tensors once. An optional `effort="max"` tier adds a chunk-parallel
context-modeling backend [@brotli] that reaches *below* the order-0 floor on real
transformer weights (+3–12% over ZipNN), falling back to the fast default on
genuinely incompressible data. `compress_delta`/`model_delta` implement
name-aligned, per-tensor lossless deltas (copy / XOR-delta / full) robust to
reordered or added tensors. General-purpose float codecs such as ALP [@alp] and
Pcodec [@pcodec] were evaluated but target smooth or decimal-origin numeric data,
not the full-entropy mantissa of trained weights, and were not adopted.

# Acknowledgements

We thank the authors of ZipNN, NeuZip, DFloat11, and ZipLLM, whose published
analyses of float-field and checkpoint redundancy directly informed this work.

# References
