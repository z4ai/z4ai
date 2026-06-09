# Copyright 2026 The z4ai Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Honest, reproducible measurement of z4ai's ONE real ratio win over ZipNN:
long-range deduplication of a tied embedding.

Most decoder LLMs (GPT-2, LLaMA, Mistral, Gemma, Qwen, ...) tie the input
embedding and the output projection (``embed_tokens`` == ``lm_head``), so the
checkpoint contains the *same* large tensor twice, megabytes apart. ZipNN
compresses in independent 256 KiB chunks, so it cannot see a duplicate that far
away; z4ai's whole-buffer long-distance-matching (LDM) window dedups it for
free. On the *dense* part of the model both codecs sit at the order-0 entropy
floor (a tie) -- the win comes entirely from the structure ZipNN's chunking
misses.

This is the legitimate, generalizable counterpart to the (non-reproducible)
"2.24x on bert-tiny" claims that have circulated: it isolates the exact
mechanism and is deterministic. Run:

    python benchmarks/bench_tied_embedding.py
    python benchmarks/bench_tied_embedding.py --vocab 32000 --d 768 --layers 6
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import z4ai


def build_checkpoint(vocab: int, d: int, layers: int, seed: int = 0) -> bytes:
    """A realistic decoder-LLM weight layout with a tied embedding/lm_head."""
    rng = np.random.default_rng(seed)
    embed = rng.normal(0, 0.02, (vocab, d)).astype(np.float32)  # token embedding
    blocks = [
        rng.normal(0, 0.02, (d, 4 * d)).astype(np.float32)      # dense per-layer
        for _ in range(layers)
    ]
    # lm_head is the SAME tensor as embed (weight tying) -> exact duplicate.
    parts = [embed.ravel()] + [b.ravel() for b in blocks] + [embed.ravel()]
    return np.concatenate(parts).astype(np.float32).tobytes()


def zipnn_ratio(data: bytes):
    """Ratio for ZipNN, verifying lossless round-trip. ZipNN mutates its input."""
    try:
        import zipnn
    except Exception:
        return None, None
    for kw in ({"bytearray_dtype": "float32"}, {}):
        try:
            z = zipnn.ZipNN(**kw)
            comp = z.compress(bytearray(data))
            dec = z.decompress(comp)
            return len(data) / len(comp), bytes(dec) == bytes(data)
        except Exception:
            continue
    return None, None


def best_of(fn, k: int = 3) -> float:
    fn()
    best = float("inf")
    for _ in range(k):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()

    raw = build_checkpoint(args.vocab, args.d, args.layers)
    mb = len(raw) / 1e6
    print(f"checkpoint: {mb:.0f} MB fp32, tied embedding "
          f"({args.vocab}x{args.d} = {args.vocab * args.d * 4 / 1e6:.0f} MB duplicated)")

    blob = z4ai.compress(raw, dtype="fp32")
    lossless = bytes(z4ai.decompress(blob)) == raw
    z_ratio = len(raw) / len(blob)
    c = best_of(lambda: z4ai.compress(raw, dtype="fp32"))
    d = best_of(lambda: z4ai.decompress(blob))
    print(f"z4ai : ratio={z_ratio:.3f}  lossless={lossless}  "
          f"compress={mb / 1e3 / c:.2f} GB/s  decompress={mb / 1e3 / d:.2f} GB/s")

    zr, zok = zipnn_ratio(raw)
    if zr is not None:
        print(f"zipnn: ratio={zr:.3f}  lossless={zok}")
        print(f"\n-> z4ai is {z_ratio / zr - 1:+.0%} vs ZipNN on ratio "
              f"({len(blob) / 1e6:.0f} MB vs {len(raw) / zr / 1e6:.0f} MB). "
              f"The dense part ties; the win is the deduped tied embedding.")
    else:
        print("zipnn: not installed (pip install zipnn) -- showing z4ai only")


if __name__ == "__main__":
    main()
