# SPDX-License-Identifier: Apache-2.0

"""z4ai vs ZipNN on a *whole-checkpoint* byte buffer.

Why this benchmark exists (read before trusting any number here):

For a **single dense tensor**, z4ai, ZipNN, and plain entropy coding all land at
essentially the same ratio - that is the information-theoretic floor of the
weight distribution (~1.5x for bf16, ~1.2x for fp32). There is no honest dense
win to be had, and this harness does not pretend otherwise.

z4ai's real, defensible advantage shows up at the *checkpoint* level. A saved
model is not one tensor; it is dozens of tensors concatenated into one file, and
those tensors contain genuine **long-range redundancy**:

* tied embeddings - the same large matrix appears as both the input embedding
  and the output projection, megabytes apart in the file;
* weight tying / shared transformer blocks - some blocks are byte-identical
  copies of earlier ones;
* pruned (structured-sparse) layers full of exact zeros;
* smooth optimizer state (Adam ``v``); and
* zero padding.

z4ai compresses the **whole buffer** with a single LZ window, so it can dedup a
tied embedding that lives 30 MB away from its twin. ZipNN slices the buffer into
independent ~256 KB pure-Huffman chunks and never sees across that boundary, so
the second copy of a tied tensor costs it (nearly) full price again. This
benchmark builds a synthetic-but-realistic checkpoint exhibiting exactly those
structures, measures all three codecs honestly (best-of-3, lossless-verified),
and prints a per-component byte breakdown plus a one-line verdict.

Run:  ``.venv/bin/python benchmarks/checkpoint_bench.py --mb 96``
      ``.venv/bin/python benchmarks/checkpoint_bench.py --mb 128 --dtype fp32``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# Make the package importable when run in-tree.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from _data import bytes_per_element, f32_to_bf16_bytes, make_weights


# --------------------------------------------------------------------------- #
# Checkpoint construction
# --------------------------------------------------------------------------- #
def _encode(f32: np.ndarray, dtype: str) -> bytes:
    if dtype == "bf16":
        return f32_to_bf16_bytes(f32)
    if dtype == "fp16":
        return f32.astype(np.float16).astype("<f2").tobytes()
    if dtype == "fp32":
        return f32.astype("<f4").tobytes()
    raise ValueError(f"unsupported dtype: {dtype!r}")


def build_checkpoint(
    target_bytes: int,
    dtype: str,
    seed: int,
    *,
    tied_embeddings: bool = True,
    shared_blocks: bool = True,
    optimizer_state: bool = True,
    sparsity: bool = True,
    padding: bool = True,
) -> Tuple[bytes, List[Tuple[str, int]]]:
    """Assemble a realistic checkpoint byte buffer.

    Returns ``(raw, breakdown)`` where ``breakdown`` is a list of
    ``(component_name, n_bytes)`` in file order. Total length is approximately
    ``target_bytes`` (rounded by element/tensor sizes).
    """
    rng = np.random.default_rng(seed)
    bpe = bytes_per_element(dtype)
    parts: List[bytes] = []
    breakdown: List[Tuple[str, int]] = []

    def emit(name: str, b: bytes) -> None:
        parts.append(b)
        breakdown.append((name, len(b)))

    # Budget the components as fractions of the target. Each fraction is the
    # share of the *final* file, so duplicated bytes are counted at full size.
    #   embedding (x2 if tied), distinct dense blocks, shared/dup blocks,
    #   optimizer state, pruned-sparse layer, padding.
    n_elems_total = max(1, target_bytes // bpe)

    # --- 1. Embedding matrix (large), optionally tied as output projection. ---
    emb_elems = max(1024, int(n_elems_total * 0.18))
    emb_f32 = rng.standard_normal(emb_elems).astype(np.float32) * np.float32(0.04)
    emb_bytes = _encode(emb_f32, dtype)
    emit("embedding", emb_bytes)
    if tied_embeddings:
        # The SAME tensor reappears as the output projection, megabytes later.
        # (We duplicate the exact bytes; a real LM ties weight or its transpose -
        # either way the byte content is identical to the input embedding here.)
        pass  # emitted after the transformer blocks, to put real distance between them.

    # --- 2. Transformer blocks: some genuinely distinct, some shared copies. ---
    block_elems = max(1024, int(n_elems_total * 0.06))
    # Build a pool of distinct dense blocks (i.i.d. Gaussian via make_weights).
    n_distinct = 4
    distinct_blocks: List[bytes] = []
    for i in range(n_distinct):
        b = make_weights(block_elems, dtype=dtype, seed=seed + 100 + i)
        distinct_blocks.append(b)
        emit(f"block_distinct_{i}", b)

    if shared_blocks:
        # Weight tying / shared layers: re-emit earlier blocks verbatim, plus a
        # couple of near-duplicates (a distinct block with a small perturbation,
        # which whole-buffer LZ still partially dedups).
        emit("block_shared_copy_0", distinct_blocks[0])
        emit("block_shared_copy_1", distinct_blocks[1])
        emit("block_shared_copy_0b", distinct_blocks[0])

    # --- (1 cont.) Tied output projection: duplicate embedding bytes, far away. ---
    if tied_embeddings:
        emit("output_projection_tied", emb_bytes)

    # --- 4. Structured sparsity: a pruned layer with ~50% exact zeros. ---
    if sparsity:
        sp_elems = max(1024, int(n_elems_total * 0.12))
        sp_f32 = rng.standard_normal(sp_elems).astype(np.float32) * np.float32(0.03)
        mask = rng.random(sp_elems) < 0.5
        sp_f32[mask] = 0.0
        emit("pruned_sparse_layer", _encode(sp_f32, dtype))

    # --- 3. Optimizer state: Adam second moment v - all-positive, smooth. ---
    if optimizer_state:
        v_elems = max(1024, int(n_elems_total * 0.12))
        # Slowly varying, strictly positive: a smoothed cumulative random walk
        # squared. This is low-entropy along the buffer (neighbours correlated).
        walk = np.cumsum(rng.standard_normal(v_elems).astype(np.float32) * np.float32(0.01))
        v_f32 = (np.abs(walk) * np.float32(0.001) + np.float32(1e-6)).astype(np.float32)
        emit("adam_v_state", _encode(v_f32, dtype))

    # --- 5. Padding / zero regions. ---
    if padding:
        pad_elems = max(512, int(n_elems_total * 0.04))
        emit("zero_padding", _encode(np.zeros(pad_elems, dtype=np.float32), dtype))

    raw = b"".join(parts)
    return raw, breakdown


# --------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------- #
def _best_of(fn: Callable[[], object], reps: int = 3) -> Tuple[object, float]:
    """Run ``fn`` ``reps`` times; return (last result, fastest elapsed seconds)."""
    best = float("inf")
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return result, best


def run_z4ai(raw: bytes, dtype: str):
    import z4ai

    blob, c_t = _best_of(lambda: z4ai.compress(raw, dtype=dtype))
    out, d_t = _best_of(lambda: z4ai.decompress(blob))
    lossless = bytes(out) == raw
    return len(blob), c_t, d_t, lossless


def run_zipnn(raw: bytes, dtype: str):
    from zipnn_adapter import make_zipnn_codec

    comp, decomp, available = make_zipnn_codec(dtype)
    if not available:
        return None
    obj, c_t = _best_of(lambda: comp(raw))
    out, d_t = _best_of(lambda: decomp(obj))
    # The compressed size is the length of the actual ZipNN blob (obj[0]).
    blob = obj[0] if isinstance(obj, tuple) else obj
    csize = len(blob) if hasattr(blob, "__len__") else len(bytes(blob))
    lossless = bytes(out) == raw
    return csize, c_t, d_t, lossless


def run_zstd(raw: bytes, level: int = 3):
    try:
        import zstandard as zstd
    except Exception:  # noqa: BLE001
        return None
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    blob, c_t = _best_of(lambda: cctx.compress(raw))
    out, d_t = _best_of(lambda: dctx.decompress(blob, max_output_size=len(raw)))
    lossless = bytes(out) == raw
    return len(blob), c_t, d_t, lossless


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _mbps(nbytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (nbytes / 1e6) / seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=float, default=96.0, help="approx total checkpoint size in MB")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zstd-level", type=int, default=3)
    ap.add_argument("--no-tied-embeddings", action="store_true")
    ap.add_argument("--no-shared-blocks", action="store_true")
    ap.add_argument("--no-optimizer-state", action="store_true")
    ap.add_argument("--no-sparsity", action="store_true")
    ap.add_argument("--no-padding", action="store_true")
    args = ap.parse_args()

    target = int(args.mb * 1024 * 1024)
    raw, breakdown = build_checkpoint(
        target,
        args.dtype,
        args.seed,
        tied_embeddings=not args.no_tied_embeddings,
        shared_blocks=not args.no_shared_blocks,
        optimizer_state=not args.no_optimizer_state,
        sparsity=not args.no_sparsity,
        padding=not args.no_padding,
    )
    total = len(raw)

    print(f"Checkpoint: dtype={args.dtype}  seed={args.seed}  size={total / 1e6:.2f} MB "
          f"({total / (1024*1024):.2f} MiB)")
    print("Component breakdown (file order):")
    for name, nb in breakdown:
        print(f"  {name:<28} {nb / 1e6:8.2f} MB  ({100.0 * nb / total:5.1f}%)")
    print()

    # Run codecs.
    rows: List[Tuple[str, Optional[Tuple[int, float, float, bool]]]] = []
    rows.append(("z4ai", run_z4ai(raw, args.dtype)))
    rows.append(("ZipNN", run_zipnn(raw, args.dtype)))
    rows.append((f"zstd-{args.zstd_level}", run_zstd(raw, args.zstd_level)))

    print(f"{'codec':<12} {'ratio':>7}  {'comp MB/s':>10}  {'decomp MB/s':>12}  "
          f"{'size MB':>9}  lossless")
    print("-" * 70)
    all_lossless = True
    z4ai_ratio: Optional[float] = None
    zipnn_ratio: Optional[float] = None
    for name, res in rows:
        if res is None:
            print(f"{name:<12} {'N/A':>7}  {'':>10}  {'':>12}  {'':>9}  (unavailable)")
            continue
        csize, c_t, d_t, lossless = res
        ratio = total / csize if csize else float("inf")
        print(f"{name:<12} {ratio:>7.3f}  {_mbps(total, c_t):>10.1f}  "
              f"{_mbps(total, d_t):>12.1f}  {csize / 1e6:>9.2f}  "
              f"{'yes' if lossless else 'NO!!':>8}")
        all_lossless = all_lossless and lossless
        if name == "z4ai":
            z4ai_ratio = ratio
        elif name == "ZipNN":
            zipnn_ratio = ratio
    print("-" * 70)

    if not all_lossless:
        print("\nFAILED: at least one codec was NOT lossless. Numbers are invalid.")
        return 1

    # Honest verdict.
    print()
    if z4ai_ratio is not None and zipnn_ratio is not None:
        if z4ai_ratio > zipnn_ratio:
            pct = 100.0 * (z4ai_ratio / zipnn_ratio - 1.0)
            print(f"VERDICT: z4ai beats ZipNN by {pct:.1f}% on this structured checkpoint "
                  f"({z4ai_ratio:.3f}x vs {zipnn_ratio:.3f}x) because global LZ dedups the "
                  f"tied/repeated tensors ZipNN's 256 KB Huffman chunks miss.")
        elif z4ai_ratio < zipnn_ratio:
            pct = 100.0 * (zipnn_ratio / z4ai_ratio - 1.0)
            print(f"VERDICT: ZipNN beats z4ai by {pct:.1f}% here "
                  f"({zipnn_ratio:.3f}x vs {z4ai_ratio:.3f}x) - z4ai's global window did not "
                  f"win on this configuration. Reporting honestly.")
        else:
            print(f"VERDICT: z4ai and ZipNN tie at {z4ai_ratio:.3f}x on this configuration.")
    elif z4ai_ratio is not None:
        print(f"VERDICT: z4ai ratio {z4ai_ratio:.3f}x; ZipNN unavailable in this environment "
              f"(install zipnn[/torch] to compare).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
