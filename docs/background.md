<!-- SPDX-License-Identifier: Apache-2.0 -->

# Background & references

z4ai's building blocks are well-studied; what is specific to z4ai is applying them
to the byte structure of model weights and matching across the whole tensor - and
across checkpoints. **Honest framing:** on *dense* weights every lossless codec is
bounded by the same ~1.51x (bf16) / ~1.2x (fp32) entropy ceiling, so z4ai cannot
meaningfully out-ratio ZipNN there. The production ratios that matter come from
structure byte-grouping codecs cannot see - reduced precision, sparsity,
within-checkpoint structure, and cross-checkpoint deltas.

## Core techniques

- **Float field decorrelation** - P. Lindstrom and M. Isenburg, "Fast and
  Efficient Compression of Floating-Point Data," *IEEE TVCG* 2006
  ([fpzip](https://computing.llnl.gov/projects/fpzip)). Applied to neural-network
  weights by [ZipNN](https://arxiv.org/abs/2411.05239).
- **Long-distance LZ matching** - J. Ziv and A. Lempel, "A Universal Algorithm for
  Sequential Data Compression," *IEEE Trans. Information Theory*, 1977
  ([LZ77](https://ieeexplore.ieee.org/document/1055714)); realized through
  Zstandard's long-distance matching
  ([RFC 8878](https://datatracker.ietf.org/doc/html/rfc8878)).
- **Entropy coding near the floor (rANS)** - J. Duda, "Asymmetric Numeral
  Systems," 2013 ([arXiv:1311.2540](https://arxiv.org/abs/1311.2540)). Unlike
  Huffman, ANS/FSE has no integer-bit-per-symbol penalty
  ([Finite State Entropy](https://github.com/Cyan4973/FiniteStateEntropy)).
  **Honest caveat:** order-0 rANS is *not* a win over Zstd on real exponents
  (Zstd's LZ stage exploits run-structure to go below the order-0 floor); only an
  order-1 context model beats it. Fast interleaved rANS: F. Giesen
  ([arXiv:1402.3392](https://arxiv.org/abs/1402.3392),
  [ryg_rans](https://github.com/rygorous/ryg_rans)).
- **Context-modeling backend (`effort="max"`)** - Brotli, Alakuijala et al., *ACM
  TOIS* 2018 ([RFC 7932](https://datatracker.ietf.org/doc/html/rfc7932),
  [google/brotli](https://github.com/google/brotli)), run chunk-parallel.

## Neural-network weight compression

- **ZipNN** - Hershcovitch et al. ([arXiv:2411.05239](https://arxiv.org/abs/2411.05239),
  IEEE 2024): exponent/byte split + Huffman, **no LZ**, 256 KiB chunks. The codec
  z4ai is benchmarked against; its chunked, LZ-free design is exactly why it misses
  the cross-tensor redundancy z4ai captures.
- **The ~1.51x BF16 ceiling** - exponent entropy ~2.57-2.74 bits; reported in
  *ZipServ* ([arXiv:2603.17435](https://arxiv.org/abs/2603.17435)).
- **DFloat11** - Zhang et al., NeurIPS 2025
  ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)): Huffman BF16 exponents
  with GPU decode. Confirms sign ~1 bit and mantissa ~7 bits are full-entropy.
- **ECF8** - Yang et al. ([arXiv:2510.02676](https://arxiv.org/abs/2510.02676),
  2025): derives the exponent's low entropy from the alpha-stable distribution of
  trained weights.
- **NeuZip** - Hao et al., NeurIPS 2024
  ([arXiv:2410.20650](https://arxiv.org/abs/2410.20650)): ANS on the exponent for
  memory-efficient training/inference.
- **DietGPU** - Johnson, Meta AI
  ([facebookresearch/dietgpu](https://github.com/facebookresearch/dietgpu)):
  GPU ANS float compression with a sign/exponent/mantissa split.
- **Cross-checkpoint delta** - [arXiv:2508.19263](https://arxiv.org/abs/2508.19263):
  XOR/delta between consecutive checkpoints zeros most bits - basis for
  `compress_delta`.
- **ZipLLM** - Wang et al., NSDI 2026
  ([arXiv:2505.06252](https://arxiv.org/abs/2505.06252)): dedups identical tensors
  across a corpus and stores fine-tunes as a lossless XOR delta against their base
  (~54% storage, ~20% better than ZipNN) - the largest lossless win, and the basis
  for `model_delta`. It dedups *before* compressing.
- **Palette / dictionary coding for quantized weights** - quantized models
  (INT4/INT8/FP8 via GPTQ / AWQ /
  [`compressed-tensors`](https://huggingface.co/docs/transformers/en/quantization/compressed_tensors))
  are commonly shipped *dequantised into a wide float container*, leaving only a
  small set of distinct values. z4ai relabels them to a dense index + codebook (a
  bijective, lossless transform) - basis for `z4ai.palette`. Low-precision NN-
  component compression is surveyed in
  [arXiv:2508.19263](https://arxiv.org/abs/2508.19263).

## General float codecs surveyed but not adopted

These target a different data shape than NN weights (whose mantissa is
full-entropy noise):

- **ALP** - Afroozeh, Kuffo, Boncz, SIGMOD 2024
  ([code](https://github.com/cwida/ALP)): great for doubles that originated as
  decimals; NN weights have no decimal structure.
- **Pcodec** - Loncaric ([arXiv:2502.06112](https://arxiv.org/abs/2502.06112)):
  converges to the entropy of *smooth* numeric sequences; weight matrices are not
  smooth along any axis.
