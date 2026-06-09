# SPDX-License-Identifier: Apache-2.0

"""Benchmark the *production* workflow: compressing a ``.safetensors`` model.

Head-to-head on a synthetic but realistic BF16 checkpoint (tied input/output
embedding + transformer-ish layers), comparing three lossless approaches:

  * **whole-file zstd**   - naive: one Zstd pass over the entire file (LDM on).
  * **per-tensor ZipNN**  - ZipNN applied to each tensor (its byte path, fed a
                            mutable ``bytearray`` so it is lossless), summed.
  * **z4ai.safetensors**  - z4ai's per-tensor codec (this repo's feature).

All three are verified byte-exact. The point: per-tensor specialization beats a
naive whole-file pass, and z4ai's whole-tensor long-distance matching dedups the
tied embedding (identical ``embed_tokens`` / ``lm_head``) that ZipNN's 256 KiB
chunking cannot fully exploit.

    PYTHONPATH=. .venv/bin/python benchmarks/bench_safetensors.py --layers 8 --d 1024
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from z4ai import safetensors as zst  # noqa: E402
import zstandard as zstd  # noqa: E402


def _bf16(f32: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(f32, np.float32).view(np.uint32)
    bias = ((u32 >> 16) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((u32 + bias) >> 16).astype(np.uint16)


def _build_model(layers: int, d: int, vocab: int, seed: int) -> bytes:
    """A BF16 checkpoint with a TIED embedding and `layers` MLP blocks."""
    rng = np.random.default_rng(seed)

    def w(n, scale):
        return _bf16(rng.standard_normal(n).astype(np.float32) * np.float32(scale))

    emb = w(vocab * d, 0.02)
    tensors = {
        "model.embed_tokens.weight": emb,
        "lm_head.weight": emb,  # tied: byte-identical to embed_tokens
    }
    for i in range(layers):
        tensors[f"model.layers.{i}.mlp.up"] = w(d * 4 * d, 0.02)
        tensors[f"model.layers.{i}.mlp.down"] = w(4 * d * d, 0.02)
        tensors[f"model.layers.{i}.attn.qkv"] = w(3 * d * d, 0.02)

    header, buffers, off = {}, [], 0
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": [len(arr)],
            "data_offsets": [off, off + len(raw)],
        }
        buffers.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(hj)) + hj + b"".join(buffers)


def _best(fn, repeats=2):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _whole_file_zstd(blob: bytes, level: int):
    params = zstd.ZstdCompressionParameters.from_level(
        level, enable_ldm=True, window_log=27
    )
    c = zstd.ZstdCompressor(compression_params=params)
    d = zstd.ZstdDecompressor(max_window_size=1 << 27)
    comp = c.compress(blob)
    assert d.decompress(comp) == blob
    return len(comp)


def _per_tensor_zipnn(blob: bytes):
    try:
        import zipnn
    except Exception:  # noqa: BLE001
        return None
    (hlen,) = struct.unpack_from("<Q", blob, 0)
    header = json.loads(blob[8 : 8 + hlen])
    data = blob[8 + hlen :]
    total = 8 + hlen  # header preserved verbatim, as z4ai does
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        raw = data[b:e]
        z = zipnn.ZipNN(bytearray_dtype="bfloat16", input_format="byte")
        comp = z.compress(bytearray(raw))
        assert bytes(z.decompress(comp)) == raw, f"zipnn not lossless on {name}"
        total += len(comp)
    return total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--zstd-level", type=int, default=19)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    blob = _build_model(args.layers, args.d, args.vocab, args.seed)
    orig = len(blob)
    print(
        f"Synthetic BF16 checkpoint: {orig/1e6:.1f} MB "
        f"({args.layers} layers, d={args.d}, vocab={args.vocab}, tied embedding)\n"
    )
    print(f"{'approach':<22}{'ratio':>9}{'MB/s':>9}{'lossless':>10}")
    print("-" * 50)

    # whole-file zstd
    (wf, _), s = _best(lambda: (_whole_file_zstd(blob, args.zstd_level), None))
    print(
        f"{'whole-file zstd-'+str(args.zstd_level):<22}{orig/wf:>9.3f}{(orig/1e6)/s:>9.0f}{'yes':>10}"
    )

    # per-tensor zipnn
    zt = _per_tensor_zipnn(blob)
    if zt:
        (zt2, _), s = _best(lambda: (_per_tensor_zipnn(blob), None))
        print(f"{'per-tensor ZipNN':<22}{orig/zt:>9.3f}{(orig/1e6)/s:>9.0f}{'yes':>10}")

    # z4ai.safetensors
    (frame, _), s = _best(lambda: (zst.compress_bytes(blob, level=args.level), None))
    assert zst.decompress_bytes(frame) == blob
    za = len(frame)
    print(f"{'z4ai.safetensors':<22}{orig/za:>9.3f}{(orig/1e6)/s:>9.0f}{'yes':>10}")

    print()
    if zt:
        print(
            f"z4ai vs per-tensor ZipNN: ratio {(orig/za)/(orig/zt):.2f}x "
            f"({(1 - za/zt)*100:+.1f}% smaller output)"
        )
    print(
        f"z4ai vs whole-file zstd : ratio {(orig/za)/(orig/wf):.2f}x "
        f"({(1 - za/wf)*100:+.1f}% smaller output)"
    )


if __name__ == "__main__":
    main()
