# SPDX-License-Identifier: Apache-2.0

"""Benchmark z4ai vs ZipNN vs raw Zstd on a **real** model checkpoint.

Synthetic i.i.d. weights are the *worst* case for differentiating codecs - the
mantissa is pure noise, so everyone is pinned to the entropy floor and ties.
Real checkpoints are where the differences live: tied embeddings, repeated /
low-rank / near-zero layers, padding, and tightly-clustered per-tensor exponents.
This harness downloads an actual trained model, concatenates its weights, and -
because a real fp32 tensor truncates to realistic fp16/bf16 bit patterns - also
derives fp16 and bf16 views, so all three z4ai paths are exercised on *real*
weight distributions from a single download.

ZipNN is driven through the verified-lossless adapter in ``zipnn_adapter.py``
(its raw ``input_format="byte"`` path corrupts data on this build - see that file).

Usage:
    python benchmarks/real_model.py                         # default small model
    python benchmarks/real_model.py --model EleutherAI/pythia-70m
    python benchmarks/real_model.py --max-mb 64 --dtypes bf16 fp16 fp32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import zstandard as zstd  # noqa: E402

from zipnn_adapter import make_zipnn_codec  # noqa: E402

# A few small, real, public checkpoints with .safetensors weights.  We try them
# in order and use the first that downloads.  All are genuine trained models
# (not random-init), so their weight statistics are representative.
_DEFAULT_MODELS = [
    "prajjwal1/bert-tiny",        # ~17 MB fp32, real BERT
    "sshleifer/tiny-gpt2",        # ~ 2 MB, real-ish GPT2
    "EleutherAI/pythia-70m",      # ~166 MB fp32, real transformer
]


def _load_weights(model_id: str, max_bytes: int) -> np.ndarray:
    """Download a checkpoint and return its weights as one contiguous fp32 array.

    Tries safetensors first, then a torch ``.bin``.  Concatenates every
    floating tensor (cast to fp32) up to ``max_bytes`` of fp32 data.
    """
    from huggingface_hub import hf_hub_download

    chunks = []
    total = 0

    def _add(arr: np.ndarray) -> bool:
        nonlocal total
        flat = np.ascontiguousarray(arr).ravel().astype(np.float32)
        room = (max_bytes - total) // 4
        if room <= 0:
            return False
        if flat.size > room:
            flat = flat[:room]
        chunks.append(flat)
        total += flat.size * 4
        return total < max_bytes

    # Preferred: safetensors.
    try:
        path = hf_hub_download(model_id, "model.safetensors")
        from safetensors.numpy import load_file

        for _, t in load_file(path).items():
            if t.dtype.kind == "f" and not _add(t):
                break
        if chunks:
            return np.concatenate(chunks)
    except Exception as e:  # noqa: BLE001
        print(f"  (safetensors unavailable for {model_id}: {e})")

    # Fallback: torch .bin
    try:
        import torch

        path = hf_hub_download(model_id, "pytorch_model.bin")
        state = torch.load(path, map_location="cpu", weights_only=True)
        for _, t in state.items():
            if torch.is_floating_point(t) and not _add(t.detach().float().numpy()):
                break
        if chunks:
            return np.concatenate(chunks)
    except Exception as e:  # noqa: BLE001
        print(f"  (torch .bin unavailable for {model_id}: {e})")

    raise RuntimeError(f"could not load weights for {model_id}")


def _to_dtype_bytes(weights_f32: np.ndarray, dtype: str) -> bytes:
    """Render the fp32 weights into raw little-endian bytes of the target dtype.

    bf16 uses round-to-nearest-even truncation (matches PyTorch's ``.bfloat16()``).
    """
    if dtype == "fp32":
        return np.ascontiguousarray(weights_f32, dtype="<f4").tobytes()
    if dtype == "fp16":
        return weights_f32.astype("<f2").tobytes()
    if dtype == "bf16":
        u32 = np.ascontiguousarray(weights_f32, dtype=np.float32).view(np.uint32)
        bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)  # round-to-nearest-even
        return ((u32 + bias) >> 16).astype("<u2").tobytes()
    raise ValueError(dtype)


def _timed(fn, repeats=2):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _mbps(n, s):
    return (n / 1e6) / s if s > 0 else float("inf")


def _run_codec(name, comp, decomp, data: bytes):
    (c, c_s) = _timed(lambda: comp(bytearray(data)))
    (d, d_s) = _timed(lambda: decomp(c))
    csize = len(c) if isinstance(c, (bytes, bytearray)) else len(c[0])
    return {
        "name": name,
        "ratio": len(data) / csize,
        "comp_mbps": _mbps(len(data), c_s),
        "decomp_mbps": _mbps(len(data), d_s),
        "lossless": bytes(d) == data,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="HF model id (default: try a small list)")
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32"])
    p.add_argument("--max-mb", type=float, default=48.0, help="cap fp32 weight bytes")
    p.add_argument("--level", type=int, default=None, help="z4ai/zstd level (None=default)")
    args = p.parse_args(argv)

    import z4ai

    models = [args.model] if args.model else _DEFAULT_MODELS
    weights = None
    for m in models:
        print(f"Loading {m} ...")
        try:
            weights = _load_weights(m, int(args.max_mb * 1e6))
            used_model = m
            break
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
    if weights is None:
        print("Could not load any model. Check network / model ids.")
        return 1

    print(f"\nModel: {used_model}  ({weights.size:,} float params, "
          f"{weights.nbytes/1e6:.1f} MB as fp32)\n")

    zlevel = 3 if args.level is None else args.level
    for dtype in args.dtypes:
        data = _to_dtype_bytes(weights, dtype)
        print(f"== {dtype}  ({len(data)/1e6:.1f} MB) ==")
        hdr = f"{'codec':<14}{'ratio':>9}{'comp MB/s':>12}{'decomp MB/s':>14}{'lossless':>10}"
        print(hdr); print("-" * len(hdr))
        rows = []

        cc = zstd.ZstdCompressor(level=zlevel, threads=-1)
        dd = zstd.ZstdDecompressor()
        rows.append(_run_codec(f"zstd-{zlevel}", cc.compress, dd.decompress, data))

        def zc(b):
            return z4ai.compress(b, dtype=dtype, level=args.level)
        rows.append(_run_codec("z4ai", zc, z4ai.decompress, data))

        zcomp, zdecomp, ok = make_zipnn_codec(dtype)
        if ok:
            rows.append(_run_codec("zipnn", zcomp, zdecomp, data))
        else:
            print("  (zipnn: no lossless path for this dtype in this env)")

        for r in rows:
            print(f"{r['name']:<14}{r['ratio']:>9.3f}{r['comp_mbps']:>12.0f}"
                  f"{r['decomp_mbps']:>14.0f}{('yes' if r['lossless'] else 'NO!'):>10}")

        by = {r["name"]: r for r in rows}
        if "z4ai" in by and "zipnn" in by:
            z, b = by["z4ai"], by["zipnn"]
            dr = (z["ratio"] / b["ratio"] - 1) * 100
            verdict = "WIN" if dr > 0.5 else ("TIE" if dr > -0.5 else "loss")
            print(f"  -> z4ai vs zipnn ratio: {dr:+.1f}%  [{verdict}]   "
                  f"(comp {z['comp_mbps']/b['comp_mbps']:.2f}x, "
                  f"decomp {z['decomp_mbps']/b['decomp_mbps']:.2f}x)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
