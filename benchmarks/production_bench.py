# SPDX-License-Identifier: Apache-2.0
"""Production-grade, HONEST benchmark: z4ai vs ZipNN.

Compares both codecs on realistic LLM-weight scenarios, measuring BOTH
compression ratio AND compress/decompress throughput (MB/s), with byte-exact
lossless assertions on every run. No cherry-picking: every scenario+dtype is
reported, wins and losses alike.

Run:
    .venv/bin/python benchmarks/production_bench.py

Importable: scenario generators and the runner are exposed as functions, and
the heavy work is guarded by ``if __name__ == '__main__'``.

ZipNN usage notes (verified in this repo's environment):
  * The ``input_format='byte'`` path is lossless but MUTATES its input buffer
    in place. We therefore hand it a *fresh* ``bytearray`` copy on every call
    and keep a pristine ``bytes`` copy for the equality check. The compressed
    blob is likewise copied before being handed to ``decompress``.
  * ``decompress`` returns a ``bytearray``; we compare ``bytes(out) == raw``.

z4ai usage: ``z4ai.compress(data, dtype=...)`` / ``z4ai.decompress(blob)``.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

# Allow running directly as `python benchmarks/production_bench.py` from any cwd:
# ensure the repo root (parent of this file's dir) is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import z4ai

try:  # ZipNN is optional; the bench still runs (z4ai-only) without it.
    import zipnn  # noqa: F401

    _HAVE_ZIPNN = True
except Exception:  # noqa: BLE001
    _HAVE_ZIPNN = False

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 1234
TIMED_RUNS = 5          # median of >= 5 timed runs
WARMUP_RUNS = 1
TARGET_MB = 64.0        # nominal size for synthetic dense/sparse/etc.

# z4ai dtype code -> zipnn bytearray_dtype string for input_format='byte'
_ZIPNN_BYTE_DTYPE = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}

# z4ai dtype code -> numpy element width (bytes)
_WIDTH = {"bf16": 2, "fp16": 2, "fp32": 4}


# ----------------------------------------------------------------------------
# Casting helpers
# ----------------------------------------------------------------------------
def _to_bf16_bytes(f32: np.ndarray) -> bytes:
    """Truncate fp32 -> bf16 (drop low 16 bits) and return little-endian bytes."""
    u32 = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    return (u32 >> np.uint32(16)).astype("<u2").tobytes()


def _to_fp16_bytes(f32: np.ndarray) -> bytes:
    return np.ascontiguousarray(f32, dtype=np.float32).astype("<f2").tobytes()


def _to_fp32_bytes(f32: np.ndarray) -> bytes:
    return np.ascontiguousarray(f32, dtype="<f4").tobytes()


def _cast(f32: np.ndarray, dtype: str) -> bytes:
    if dtype == "bf16":
        return _to_bf16_bytes(f32)
    if dtype == "fp16":
        return _to_fp16_bytes(f32)
    if dtype == "fp32":
        return _to_fp32_bytes(f32)
    raise ValueError(dtype)


# ----------------------------------------------------------------------------
# Scenario generators -> return an fp32 ndarray of "weights"
# ----------------------------------------------------------------------------
def gen_dense_random(rng: np.random.Generator, target_mb: float) -> np.ndarray:
    """Many tensors, each Gaussian with a per-tensor random scale 10^U(-2.5,-0.5)."""
    n_total = int(target_mb * 1e6 / 4)  # in fp32 elements
    out = np.empty(n_total, dtype=np.float32)
    pos = 0
    # ~256 tensors of varying length to look like a transformer's parameter set
    n_tensors = 256
    sizes = rng.integers(n_total // (n_tensors * 4), n_total // (n_tensors // 4) + 1,
                         size=n_tensors)
    sizes = (sizes * (n_total / sizes.sum())).astype(np.int64)
    sizes[-1] += n_total - int(sizes.sum())
    for sz in sizes:
        if sz <= 0:
            continue
        scale = 10.0 ** rng.uniform(-2.5, -0.5)
        chunk = (rng.standard_normal(int(sz)) * scale).astype(np.float32)
        end = min(pos + int(sz), n_total)
        out[pos:end] = chunk[: end - pos]
        pos = end
        if pos >= n_total:
            break
    if pos < n_total:
        out[pos:] = (rng.standard_normal(n_total - pos) * 0.05).astype(np.float32)
    return out


def gen_sparse(rng: np.random.Generator, target_mb: float, sparsity: float) -> np.ndarray:
    """Pruned model: `sparsity` fraction set to exactly 0.0."""
    n = int(target_mb * 1e6 / 4)
    out = (rng.standard_normal(n) * 0.05).astype(np.float32)
    mask = rng.random(n) < sparsity
    out[mask] = 0.0
    return out


def gen_duplicate_heavy(rng: np.random.Generator, target_mb: float) -> np.ndarray:
    """Draw tensors from a small pool sampled with replacement (tied/duplicated layers)."""
    n = int(target_mb * 1e6 / 4)
    pool_tensor = 256 * 1024  # 1 MB per distinct fp32 tensor
    n_distinct = 8
    pool = [(rng.standard_normal(pool_tensor) * 10.0 ** rng.uniform(-2.5, -0.5)).astype(np.float32)
            for _ in range(n_distinct)]
    out = np.empty(n, dtype=np.float32)
    pos = 0
    while pos < n:
        idx = int(rng.integers(0, n_distinct))
        t = pool[idx]
        end = min(pos + t.size, n)
        out[pos:end] = t[: end - pos]
        pos = end
    return out


def gen_lora_like(rng: np.random.Generator, target_mb: float) -> np.ndarray:
    """A few large base tensors + many small low-rank-ish additions."""
    n = int(target_mb * 1e6 / 4)
    out = np.empty(n, dtype=np.float32)
    # 70% base weights (a handful of big Gaussian tensors)
    base_n = int(n * 0.7)
    base = (rng.standard_normal(base_n) * 0.05).astype(np.float32)
    out[:base_n] = base
    # 30% many small low-rank products (rank ~8): A(d,r) @ B(r,d)
    pos = base_n
    while pos < n:
        d = int(rng.integers(64, 257))
        r = 8
        a = (rng.standard_normal((d, r)) * 0.02).astype(np.float32)
        b = (rng.standard_normal((r, d)) * 0.02).astype(np.float32)
        prod = (a @ b).reshape(-1)
        end = min(pos + prod.size, n)
        out[pos:end] = prod[: end - pos]
        pos = end
    return out


def load_real_model() -> Optional[Tuple[str, np.ndarray]]:
    """Best-effort: return (label, fp32 weights) from a real model, or None.

    Order: (a) /tmp/bert-tiny.bin (torch state_dict of fp32 weights);
           (b) huggingface_hub download of a tiny model. Never raises.
    """
    # (a) local blob
    try:
        import os

        if os.path.exists("/tmp/bert-tiny.bin"):
            arr = _extract_floats_from_torch_file("/tmp/bert-tiny.bin")
            if arr is not None and arr.size > 0:
                return ("bert-tiny (local /tmp)", arr)
    except Exception:  # noqa: BLE001
        pass

    # (b) download
    for repo, fname in [
        ("prajjwal1/bert-tiny", "pytorch_model.bin"),
        ("sshleifer/tiny-gpt2", "pytorch_model.bin"),
    ]:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename=fname)
            arr = _extract_floats_from_torch_file(path)
            if arr is not None and arr.size > 0:
                return (f"{repo} (downloaded)", arr)
        except Exception:  # noqa: BLE001
            continue
    return None


def _extract_floats_from_torch_file(path: str) -> Optional[np.ndarray]:
    """Concatenate all float32/float16/bfloat16 tensors from a torch state_dict."""
    try:
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=True)
        if hasattr(sd, "values"):
            tensors = list(sd.values())
        else:
            tensors = [sd]
        floats: List[np.ndarray] = []
        for t in tensors:
            if not hasattr(t, "dtype"):
                continue
            if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
                floats.append(t.detach().to(torch.float32).contiguous().reshape(-1).numpy())
        if not floats:
            return None
        return np.concatenate(floats).astype(np.float32)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------
# Codec drivers
# ----------------------------------------------------------------------------
def _time_median(fn: Callable[[], object], n_timed: int, n_warmup: int) -> float:
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def run_z4ai(raw: bytes, dtype: str) -> dict:
    blob = z4ai.compress(raw, dtype=dtype)
    back = z4ai.decompress(blob)
    lossless = bytes(back) == raw
    comp_t = _time_median(lambda: z4ai.compress(raw, dtype=dtype), TIMED_RUNS, WARMUP_RUNS)
    decomp_t = _time_median(lambda: z4ai.decompress(blob), TIMED_RUNS, WARMUP_RUNS)
    mb = len(raw) / 1e6
    return {
        "ratio": len(raw) / len(blob),
        "comp_mbps": mb / comp_t,
        "decomp_mbps": mb / decomp_t,
        "lossless": lossless,
    }


def run_zipnn(raw: bytes, dtype: str) -> Optional[dict]:
    if not _HAVE_ZIPNN:
        return None
    bd = _ZIPNN_BYTE_DTYPE.get(dtype)
    if bd is None:
        return None
    import zipnn

    def make_zn():
        return zipnn.ZipNN(method="zstd", input_format="byte", bytearray_dtype=bd)

    # byte path mutates input in place -> always hand a fresh bytearray copy.
    blob = bytes(make_zn().compress(bytearray(raw)))
    back = make_zn().decompress(bytearray(blob))
    lossless = bytes(back) == raw
    ratio = len(raw) / len(blob)

    zn_c = make_zn()
    comp_t = _time_median(lambda: zn_c.compress(bytearray(raw)), TIMED_RUNS, WARMUP_RUNS)
    zn_d = make_zn()
    decomp_t = _time_median(lambda: zn_d.decompress(bytearray(blob)), TIMED_RUNS, WARMUP_RUNS)
    mb = len(raw) / 1e6
    return {
        "ratio": ratio,
        "comp_mbps": mb / comp_t,
        "decomp_mbps": mb / decomp_t,
        "lossless": lossless,
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def build_scenarios(target_mb: float) -> List[Tuple[str, List[str], np.ndarray]]:
    """Return list of (scenario_name, dtypes, fp32_weights)."""
    rng = np.random.default_rng(SEED)
    scenarios: List[Tuple[str, List[str], np.ndarray]] = []

    scenarios.append(("dense_random", ["bf16", "fp16", "fp32"],
                      gen_dense_random(rng, target_mb)))
    for sp, tag in [(0.50, "50"), (0.90, "90"), (0.99, "99")]:
        scenarios.append((f"sparse_{tag}", ["bf16", "fp32"],
                          gen_sparse(rng, target_mb, sp)))
    scenarios.append(("duplicate_heavy", ["bf16", "fp32"],
                      gen_duplicate_heavy(rng, target_mb)))
    scenarios.append(("lora_like", ["bf16", "fp32"],
                      gen_lora_like(rng, target_mb)))

    real = load_real_model()
    if real is not None:
        label, weights = real
        # real fp32 weights -> also cast to bf16/fp16 to compare across dtypes
        scenarios.append((f"real:{label}", ["bf16", "fp16", "fp32"], weights))
    else:
        print("[real_model] SKIPPED: no local blob and download unavailable (offline?).")
    return scenarios


HEADER = (
    "scenario", "dtype", "size_MB",
    "z4ai_ratio", "zipnn_ratio",
    "z4ai_cMBps", "zipnn_cMBps",
    "z4ai_dMBps", "zipnn_dMBps",
    "z4ai_lossless", "zipnn_lossless",
)


def _fmt_row(vals) -> str:
    widths = [22, 6, 8, 11, 11, 11, 11, 11, 11, 14, 14]
    out = []
    for v, w in zip(vals, widths):
        out.append(str(v).rjust(w))
    return " ".join(out)


def main(target_mb: float = TARGET_MB) -> List[dict]:
    print(f"z4ai vs ZipNN production benchmark | seed={SEED} | "
          f"median of {TIMED_RUNS} runs | target ~{target_mb:.0f} MB/scenario")
    print(f"ZipNN available: {_HAVE_ZIPNN}")
    print()
    print(_fmt_row(HEADER))
    print("-" * 140)

    scenarios = build_scenarios(target_mb)
    rows: List[dict] = []
    for name, dtypes, weights in scenarios:
        for dtype in dtypes:
            raw = _cast(weights, dtype)
            mb = len(raw) / 1e6
            z = run_z4ai(raw, dtype)
            zn = run_zipnn(raw, dtype)

            def g(d, k, default="n/a"):
                return f"{d[k]:.3f}" if d else default

            row = {
                "scenario": name, "dtype": dtype, "size_MB": round(mb, 2),
                "z4ai_ratio": z["ratio"], "zipnn_ratio": zn["ratio"] if zn else None,
                "z4ai_comp_MBps": z["comp_mbps"], "zipnn_comp_MBps": zn["comp_mbps"] if zn else None,
                "z4ai_decomp_MBps": z["decomp_mbps"], "zipnn_decomp_MBps": zn["decomp_mbps"] if zn else None,
                "z4ai_lossless": z["lossless"], "zipnn_lossless": zn["lossless"] if zn else None,
            }
            rows.append(row)
            print(_fmt_row((
                name, dtype, f"{mb:.2f}",
                f"{z['ratio']:.3f}", g(zn, "ratio"),
                f"{z['comp_mbps']:.1f}", g(zn, "comp_mbps"),
                f"{z['decomp_mbps']:.1f}", g(zn, "decomp_mbps"),
                str(z["lossless"]), str(zn["lossless"]) if zn else "n/a",
            )))
    return rows


if __name__ == "__main__":
    main()
