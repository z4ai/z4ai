# SPDX-License-Identifier: Apache-2.0

"""Entropy backend wrappers and the per-plane adaptive policy.

The backend is Zstandard (via the ``zstandard`` binding).  On top of it z4ai
makes one *adaptive* decision per plane that lets it win on BOTH axes against a
chunk-based codec such as ZipNN - measured, see ``BENCHMARK_FINDINGS.md``:

1. **Structured plane -> whole-plane LDM (the ratio win).**  Real checkpoints are
   full of redundancy that repeats far apart - tied input/output embeddings,
   duplicated/EMA/optimizer-shadow blocks, multi-shard concatenations, padded or
   low-rank tensors.  ZipNN Huffman-codes independent ~256 KiB chunks, so its
   match horizon is ~256 KiB.  z4ai runs Zstd with long-distance matching (LDM)
   and a window spanning the whole plane, deduplicating redundancy *megabytes*
   apart (>7x ratio win over ZipNN on duplicated-layer weights).  LDM is also
   *fast* on this data (long matches compress quickly), and we add Zstd's own
   multithreading on top, so there is no speed sacrifice for the ratio win.

2. **i.i.d. plane -> block-parallel Zstd (the throughput win).**  A plane with no
   long-range structure (mantissa noise, or freshly-pretrained weights) gains
   nothing from LDM's whole-plane window, and that window caps throughput at one
   core's match search.  For such planes we slice into independent blocks and
   compress them across a shared thread pool - ~7x the single-thread throughput
   at identical ratio (blocking only forfeits long-range matches, of which an
   i.i.d. plane has none).  This is how z4ai matches or beats ZipNN's C-level
   compression speed from Python.

3. **Per-plane raw-store guard.**  A plane that does not actually shrink is
   stored verbatim, so the codec never expands a plane.

The block framing for the parallel path is carried *inside* the plane payload
(method :data:`METHOD_ZSTD_BLOCKS`), so the on-disk container format is
unchanged and existing single-frame (:data:`METHOD_ZSTD`) blobs still decode.
"""

from __future__ import annotations

import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import zstandard as zstd

# Plane storage methods, serialized in the container header.
METHOD_STORE = 0  # plane kept verbatim (incompressible)
METHOD_ZSTD = 1  # plane compressed as a single (LDM) Zstd frame
METHOD_ZSTD_BLOCKS = 2  # plane compressed as N independent Zstd blocks (parallel)
METHOD_RANS = 3  # plane entropy-coded with the native order-0 rANS coder (shared model)
METHOD_RANS_O1 = 5  # plane entropy-coded with the native order-1 (context) rANS coder
METHOD_RANS_ADAPT = 4  # rANS with a PER-CHUNK local order-0 model (adapts to per-tensor
# exponent drift -> the ratio edge on multi-tensor weights)

# The native rANS coder is what lets z4ai beat ZipNN on *ratio*: ZipNN entropy-
# codes its byte planes with Huffman (integer-bit rounding), whereas rANS reaches
# the order-0 Shannon bound.  On the low-entropy exponent/sign planes of float
# weights that is a measured ~0.2-0.7% ratio win.  rANS is order-0, so it cannot
# exploit long-range duplication the way the Zstd-LDM path can; the two are
# complementary and :func:`compress_plane` keeps whichever is smaller per plane.
# Imported lazily/defensively so the package never hard-fails when no C compiler
# is available -- callers simply fall back to the Zstd methods.
try:
    from . import rans as _rans
except Exception:  # pragma: no cover - import-time safety net
    _rans = None

# Order-1 (context = previous byte) rANS.  Captures the CONDITIONAL structure real
# exponent planes carry (distilgpt2 bf16 exponent H1=2.36 vs H0=3.10 bits) that no
# order-0 coder -- Zstd-FSE, ZipNN's Huffman, or our own order-0 rANS above -- can
# reach.  Best-of candidate, kept only when smaller, so it never regresses a plane.
try:
    from . import rans_o1 as _rans_o1
except Exception:  # pragma: no cover - import-time safety net
    _rans_o1 = None

# Window-log bounds for LDM.  The lower bound avoids paying LDM setup on tiny
# planes; the upper bound (128 MiB) is Zstd's default decompressor window limit,
# so frames stay decodable with a stock ``ZstdDecompressor`` and bounded memory.
_MIN_WINDOW_LOG = 17  # 128 KiB - below this LDM cannot help anyway
_MAX_WINDOW_LOG = 27  # 128 MiB - Zstd's ZSTD_WINDOWLOG_LIMIT_DEFAULT

# Below this size a plane is small enough that the whole-plane LDM pass is already
# cheap, so we never bother with the block-parallel path (it would only add
# framing overhead and risk forfeiting a small plane's long-range matches).
_MIN_PARALLEL_PLANE = 4 << 20  # 4 MiB
# Floor on a parallel block's uncompressed size.  Blocks smaller than this start
# to cost ratio (less context for the entropy coder) without buying meaningful
# extra decode parallelism, so block count is capped at plane_size / this.
_MIN_BLOCK_BYTES = 1 << 20  # 1 MiB
# Uncompressed bytes per parallel block.  4 MiB maximises i.i.d. throughput in
# measurement while keeping per-block framing overhead negligible.
_DEFAULT_BLOCK_SIZE = 4 << 20  # 4 MiB

# Decode-throughput tolerance for the block-vs-LDM best-of.  A single Zstd frame
# decodes on ONE core; independent blocks decode across all cores (~5x on the
# compressible high plane — z4ai's weakest axis vs ZipNN).  When blocks are only
# marginally larger than the LDM frame we keep the BLOCK form, trading <=1% ratio
# for that large decode win — exactly the codec's existing byte-plane decode-
# preference policy.  Measured: an fp16 high plane's blocks were +0.005% over LDM
# yet decoded 1906 -> 9270 MB/s.  The structured / duplicated-layer ratio win is
# untouched: there LDM is many times (not ~1%) smaller, far outside this band, so
# LDM is still kept.
_BLOCK_DECODE_TOLERANCE = 1.01  # keep blocks if <= 1% larger than the LDM frame

# Long-range-redundancy detector: on a contiguous sample, compare a large-window
# (LDM) Zstd pass against a small-window one.  Long-range repeats - tied/duplicated/
# low-rank layers, padding, the structured-checkpoint win over ZipNN - are reachable
# only by the large window, so a big size gap means the whole-plane / opaque LDM
# path will pay off.  I.i.d. or merely locally-compressible data (exponent planes,
# sparse zeros) shows ~no gap and takes the fast block path.
#
# This replaces an earlier exact-chunk-hash heuristic that MISSED redundancy whose
# repeat period was not aligned to the sample stride (e.g. a 600 KiB-period tile at
# a ~23 KiB stride hashed as 0% duplicate).  That silently forfeited the opaque LDM
# candidate and let z4ai fall *below* plain Zstd on structured weights (20.7x vs
# 51x).  Comparing window sizes measures the exploitable property directly.  Measured
# separation is wide: i.i.d./sparse give big/small ~= 0.99, structured ~= 0.10-0.19.
_DETECT_SPAN = 2 << 20  # contiguous sample size (bytes)
_DETECT_PROBE_LEVEL = 1  # fast probe; the LDM gap shows at any level
_DETECT_SMALL_WINDOW_LOG = 17  # 128 KiB - the block path's effective window
_DETECT_LDM_GAIN = 0.9  # trigger when big-window < 0.9 * small-window

# Block-payload header: <block_size:u32><n_blocks:u32>, then n_blocks * <clen:u32>,
# then the concatenated compressed blocks.
_BLOCK_HEADER = struct.Struct("<II")
_U32 = struct.Struct("<I")

# Lazily-created, process-wide thread pool shared by every plane so that nested
# calls (the codec may already fan out across planes) never oversubscribe the CPU.
_BLOCK_POOL: "ThreadPoolExecutor | None" = None

# Per-worker-thread Zstd decompressor cache.  ZstdDecompressor is cheap to reuse
# but not safe to share across threads, so we stash one per block-pool worker and
# reuse it across every block / every decompress call (see _decompress_blocks).
_tls = threading.local()


def _block_pool() -> ThreadPoolExecutor:
    global _BLOCK_POOL
    if _BLOCK_POOL is None:
        _BLOCK_POOL = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 1, thread_name_prefix="z4ai-block"
        )
    return _BLOCK_POOL


@dataclass(frozen=True)
class CompressedPlane:
    """A compressed (or stored) plane plus the metadata needed to invert it."""

    method: int
    data: bytes
    orig_len: int


def _window_log_for(size: int) -> int:
    """Pick a window log large enough to span a plane of ``size`` bytes."""
    needed = max(1, size - 1).bit_length()  # ceil(log2(size))
    return max(_MIN_WINDOW_LOG, min(_MAX_WINDOW_LOG, needed))


def _compressor(level: int, size: int, threads: int) -> "zstd.ZstdCompressor":
    """Build an LDM-enabled compressor whose window spans the whole plane."""
    params = zstd.ZstdCompressionParameters.from_level(
        level,
        enable_ldm=True,
        window_log=_window_log_for(size),
        threads=threads,
    )
    return zstd.ZstdCompressor(compression_params=params)


def _has_long_range_redundancy(plane: bytes) -> bool:
    """Estimate whether ``plane`` has repeats only a large window can exploit.

    Compress a single contiguous sample twice - once with a plane-spanning LDM
    window, once with a small (128 KiB) window - and report whether the large
    window won by more than ``_DETECT_LDM_GAIN``.  That gap is precisely the
    advantage the opaque whole-buffer / whole-plane LDM pass has over the
    block-parallel path, so it answers the actual question ("is LDM worth it?")
    rather than guessing from chunk-hash collisions, which missed period-
    misaligned tiles and let z4ai lose to plain Zstd on structured weights.

    Returns ``True`` (keep the safe LDM path) for tiny inputs and on any
    compressor error.
    """
    n = len(plane)
    if n < 2 * (1 << _DETECT_SMALL_WINDOW_LOG):
        return True  # too small to sample meaningfully; keep the safe path
    span = min(n, _DETECT_SPAN)
    start = (n - span) // 2  # middle: skip any header/padding bias at the edges
    sample = bytes(memoryview(plane)[start : start + span])
    try:
        big_params = zstd.ZstdCompressionParameters.from_level(
            _DETECT_PROBE_LEVEL, enable_ldm=True, window_log=_window_log_for(span)
        )
        small_params = zstd.ZstdCompressionParameters.from_level(
            _DETECT_PROBE_LEVEL, enable_ldm=False, window_log=_DETECT_SMALL_WINDOW_LOG
        )
        big = len(zstd.ZstdCompressor(compression_params=big_params).compress(sample))
        small = len(
            zstd.ZstdCompressor(compression_params=small_params).compress(sample)
        )
    except Exception:  # noqa: BLE001 - never let detection break compression
        return True
    return small > 0 and big < _DETECT_LDM_GAIN * small


_PROBE_BYTES = 1 << 18  # 256 KiB local probe
_PROBE_INCOMPRESSIBLE = 0.97  # >97% of input after a fast pass == not worth it


def _probe_incompressible(plane: bytes) -> bool:
    """True if a fast local probe says ``plane`` will not meaningfully shrink.

    Only meaningful for planes already known to lack long-range redundancy
    (see :func:`_has_long_range_redundancy`), where a local sample is
    representative of the whole plane.
    """
    sample = plane[:_PROBE_BYTES]
    if not sample:
        return False
    probed = zstd.ZstdCompressor(level=1).compress(sample)
    return len(probed) > _PROBE_INCOMPRESSIBLE * len(sample)


def _compress_blocks(plane: bytes, level: int, block_size: int) -> bytes:
    """Compress ``plane`` as independent ``block_size`` blocks, in parallel."""
    offsets = list(range(0, len(plane), block_size))

    def _compress_one(off: int) -> bytes:
        # No LDM here: the block path is chosen only for planes with no
        # long-range redundancy, so a plain fast frame is both faster and
        # equal-ratio.  A fresh compressor per call keeps this thread-safe.
        return zstd.ZstdCompressor(level=level).compress(plane[off : off + block_size])

    if len(offsets) == 1:
        blocks = [_compress_one(offsets[0])]
    else:
        blocks = list(_block_pool().map(_compress_one, offsets))

    parts = [_BLOCK_HEADER.pack(block_size, len(blocks))]
    parts.extend(_U32.pack(len(b)) for b in blocks)
    parts.extend(blocks)
    return b"".join(parts)


def _decompress_blocks(data: bytes, orig_len: int) -> bytes:
    """Invert :func:`_compress_blocks`, decompressing blocks in parallel."""
    mv = memoryview(data)
    block_size, n_blocks = _BLOCK_HEADER.unpack_from(mv, 0)
    off = _BLOCK_HEADER.size

    clens: List[int] = []
    for _ in range(n_blocks):
        (clen,) = _U32.unpack_from(mv, off)
        off += _U32.size
        clens.append(clen)

    # Zero-copy: keep each compressed block as a memoryview slice into ``data``
    # rather than materialising it with ``bytes(...)``.  zstd decompresses
    # straight from a bytes-like buffer, so the per-block copy (megabytes on a
    # real plane) is pure overhead on this hot decode path.
    blocks: List[memoryview] = []
    for clen in clens:
        blocks.append(mv[off : off + clen])
        off += clen

    def _decompress_one(idx: int) -> bytes:
        usize = (
            block_size if idx < n_blocks - 1 else orig_len - (n_blocks - 1) * block_size
        )
        # Reuse one decompressor per worker thread.  Block decode runs on the
        # persistent ``_block_pool``; allocating a fresh ZstdDecompressor per
        # block (n_blocks per plane, every decompress call) is measurable
        # per-call overhead -- the gap that separates us from ZipNN's fused C.
        dec = getattr(_tls, "decompressor", None)
        if dec is None:
            dec = _tls.decompressor = zstd.ZstdDecompressor()
        out = dec.decompress(blocks[idx], max_output_size=usize)
        if len(out) != usize:
            raise ValueError(
                f"block {idx} length mismatch: got {len(out)}, expected {usize}"
            )
        return out

    if n_blocks == 1:
        parts = [_decompress_one(0)]
    else:
        parts = list(_block_pool().map(_decompress_one, range(n_blocks)))

    out = b"".join(parts)
    if len(out) != orig_len:
        raise ValueError(f"plane length mismatch: got {len(out)}, expected {orig_len}")
    return out


def _rans_candidate(plane: bytes, beat_size: int) -> "CompressedPlane | None":
    """Return a rANS-coded plane if it is smaller than ``beat_size`` bytes.

    ``beat_size`` is the size to beat (the current best, i.e. the Zstd payload).
    The returned plane's ``orig_len`` is the *decompressed* length (``len(plane)``)
    as required by :func:`decompress_plane`.  Only worth attempting on a plane
    that is not pure noise; the caller gates on the Zstd result so we do not
    spend a rANS pass on an i.i.d. mantissa plane that is stored raw anyway.
    """
    if _rans is None or not _rans.available():
        return None
    try:
        # Per-chunk LOCAL model: on a plane whose distribution drifts across the
        # buffer (a model's exponent plane - each tensor has its own exponent
        # centre) this codes near the LOCAL entropy, not the inflated global one.
        # Measured exponent plane: adaptive 3.10x vs shared-model 2.43x vs zstd
        # 2.77x, at 2.2 GB/s encode / 3.2 GB/s decode (parallel).  For a small or
        # homogeneous plane it degenerates to a single model == the shared path,
        # so it never loses ratio; hence it replaces the shared-model candidate.
        coded = _rans.compress_adaptive(plane)
    except Exception:  # pragma: no cover - never let entropy coding break a frame
        return None
    if len(coded) < beat_size:
        return CompressedPlane(METHOD_RANS_ADAPT, coded, len(plane))
    return None


# Order-1 rANS is worth attempting only on a *small-alphabet* plane (the exponent
# and sign planes of float weights), where conditional structure exists and the
# per-context model is cheap.  A high-entropy mantissa plane (~256 distinct bytes)
# cannot benefit and would only burn encode time.  64 covers the <=~40 exponent
# values seen across real models while excluding noise.
_O1_MAX_ALPHABET = 64
# Below this a plane codes in well under a millisecond either way; skip the order-1
# best-of pass (model build + encode) on tiny planes.
_O1_MIN_BYTES = 1 << 16  # 64 KiB

# Decode-cost gate for the order-0 adaptive rANS candidate.  A rANS plane decodes
# ~10x slower (single-stream per chunk) than the parallel Zstd-block path, so it is
# worth keeping only when it is MEANINGFULLY smaller, not merely 1 byte smaller.
# The low-entropy exponent/sign planes clear this easily (measured rANS +5.8% on
# distilgpt2's exponent, +38% on pythia-70m's); the high-entropy mantissa — already
# at its order-1 entropy ceiling, where rANS measures only -0.1% to +0.23% vs Zstd —
# does not, so it stays on the fast Zstd-block path.  That recovers compress and
# decode throughput on the largest plane at negligible ratio cost (the real ratio
# win comes from the exponent, kept by the order-1 candidate below, which has its
# own large-win alphabet gate and is unaffected by this).
_RANS_ADAPT_MIN_GAIN = (
    0.01  # rANS must be >=1% smaller than Zstd to justify its decode cost
)

# Entropy pre-gate for the rANS best-of.  A rANS coder (order-0, per-chunk-local,
# or order-1) has NO LZ stage, so it can only beat the Zstd plane payload when Zstd
# did *not* already drop below the plane's order-0 entropy.  When Zstd's payload is
# already well under the order-0-entropy size, its LZ/LDM stage found run/match
# structure that rANS cannot reach -- on real model weights this is the norm
# (measured: bf16 exponent/sign planes compress to ~1.9 bits/sym via Zstd-LDM, far
# below their ~2.8-bit order-0 entropy), so the two rANS encodes add +0.00% ratio
# at ~2.5x the entropy-stage compress cost.  Skipping them there is a pure
# compress-throughput win with zero ratio change.  The gate fires ONLY when Zstd
# beat order-0 by more than (1 - margin); i.i.d.-style planes (synthetic weights,
# where Zstd sits AT its order-0 entropy and rANS's Huffman-rounding edge is the
# real win over ZipNN) keep the full rANS best-of unchanged.  The margin is
# deliberately generous so any plane where a local-adaptive model could still win
# is not skipped.
_RANS_GATE_MARGIN = 0.80


def _order0_entropy_size(plane: bytes) -> float:
    """Bytes the plane would take at its global order-0 (Shannon) entropy.

    A fast lower-ish reference for what any LZ-free entropy coder can achieve;
    used by the rANS pre-gate.  ``np.bincount`` over the bytes is memory-bandwidth
    fast (multi-GB/s), negligible next to the rANS encode it can save.
    """
    n = len(plane)
    if n == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(plane, dtype=np.uint8), minlength=256).astype(
        np.float64
    )
    nz = counts[counts > 0]
    p = nz / n
    bits_per_sym = float(-(p * np.log2(p)).sum())
    return bits_per_sym * n / 8.0


# For the order-1 alphabet gate we only need to know whether the plane has more
# than ``_O1_MAX_ALPHABET`` distinct bytes, so a strided sample suffices: a sample
# can only *undercount* the true alphabet (it is a subset), so a low-alphabet
# plane — the exponent/sign streams order-1 rANS targets — is never missed, while
# a high-entropy mantissa plane overshoots the gate from a few KB instead of a
# full histogram over tens of MB.  This made the gate ~the cost of the model build
# it guards; sampling drops it to negligible.
_DISTINCT_SAMPLE_BYTES = 1 << 18  # 256 KiB strided sample is plenty to see >64 symbols


def _distinct_bytes(plane: bytes) -> int:
    """Distinct byte values in ``plane`` (sampled — see :data:`_DISTINCT_SAMPLE_BYTES`).

    For a plane larger than the sample budget a strided subsample is histogrammed;
    the result is a lower bound on the true alphabet, which is exactly what the
    conservative order-1 gate needs (it never over-counts, so it never skips a
    genuinely low-alphabet plane).
    """
    arr = np.frombuffer(plane, dtype=np.uint8)
    if arr.size > _DISTINCT_SAMPLE_BYTES:
        stride = arr.size // _DISTINCT_SAMPLE_BYTES
        arr = arr[::stride]
    return int(np.count_nonzero(np.bincount(arr, minlength=256)))


def _rans_o1_candidate(plane: bytes, beat_size: int) -> "CompressedPlane | None":
    """Return an order-1 rANS plane if it is smaller than ``beat_size`` bytes."""
    if _rans_o1 is None or not _rans_o1.available():
        return None
    try:
        coded = _rans_o1.compress(plane)
    except Exception:  # pragma: no cover - never let entropy coding break a frame
        return None
    if len(coded) < beat_size:
        return CompressedPlane(METHOD_RANS_O1, coded, len(plane))
    return None


def compress_plane(
    plane: bytes,
    level: int,
    threads: int = 0,
    escalate_level: int | None = None,
    *,
    parallel: bool = True,
    rans: bool = True,
) -> CompressedPlane:
    """Compress a plane, keeping the smaller of the Zstd and rANS results.

    The Zstd path (LDM / block-parallel / store) handles long-range redundancy
    and throughput; the native order-0 rANS path reaches the entropy bound on
    low-entropy planes (the ratio edge over ZipNN's Huffman).  Whichever yields
    the smaller payload for this particular plane wins -- so the codec never
    regresses relative to either method alone.

    ``rans=False`` skips the rANS candidates and returns the Zstd result
    (block-parallel where applicable).  rANS reaches the entropy floor but its
    streams decode ~6x slower than the block-parallel Zstd path, so the
    decode-speed tier (``effort="fast"``) sets ``rans=False`` to keep the fast
    multi-core decode; the ratio-first default/max tiers leave it on.
    """
    zstd_result = _compress_plane_zstd(
        plane, level, threads, escalate_level, parallel=parallel
    )
    # rANS only stands a chance on planes Zstd already found compressible; an
    # incompressible plane (stored) is i.i.d. noise where rANS ~= raw size too.
    if zstd_result.method == METHOD_STORE or not rans:
        return zstd_result
    # rANS reaches the order-0 entropy floor EXACTLY, whereas Zstd entropy-codes
    # its literals with Huffman (integer-bit rounding).  On the low-entropy float-
    # weight planes that matters most -- the exponent (~2.5 bits real entropy vs
    # Zstd-L1's ~2.9), the sign, the bf16 mantissa byte (wasted constant high bit)
    # -- rANS is a measured 4-13% smaller than Zstd-L1 on the plane, the edge that
    # turns z4ai's i.i.d.-weight TIE/LOSS vs ZipNN (Huffman-class) into a WIN.
    # Formerly opt-in (escalate_level) because single-stream rANS ran ~200 MB/s;
    # it is now chunked + threaded (shared model, GIL-releasing ctypes across all
    # cores) at ~2.3 GB/s encode / 2.4 GB/s decode -- on par with the Zstd plane
    # path -- so it is ON BY DEFAULT.  ``_rans_candidate`` keeps it only when it is
    # strictly smaller than the Zstd payload, so a long-range-redundant plane (where
    # order-0 rANS cannot beat Zstd-LDM dedup) keeps Zstd: never a ratio regression.
    best = zstd_result
    # Require the order-0 adaptive rANS to beat Zstd by a meaningful margin (not
    # just 1 byte) to justify its ~10x slower decode -- see _RANS_ADAPT_MIN_GAIN.
    # This is a BALANCED-default trade: in max-ratio mode (explicit escalation or a
    # high level) the user wants the absolute smallest file regardless of decode
    # speed, so keep every rANS win there (no margin).  The order-1 candidate below
    # is never margined -- its small-alphabet gate already restricts it to the
    # exponent/sign planes where the win is large.
    _max_ratio = escalate_level is not None or level >= 10
    # Entropy pre-gate (balanced default only): when Zstd's LZ/LDM stage already
    # dropped this plane well below its order-0 entropy, an LZ-free rANS coder
    # cannot win, so skip both rANS encodes -- a measured ~2.5x compress-throughput
    # win on real-weight exponent/sign planes for +0.00% ratio (on distilgpt2 the
    # bf16 exponent compresses to ~1.9 bits/sym via Zstd-LDM, far below its ~2.8-bit
    # order-0 entropy, so rANS provably loses).  The gate does NOT fire on
    # i.i.d.-style planes (Zstd sits AT its order-0 entropy there), so the rANS edge
    # that turns z4ai's i.i.d.-weight tie/loss vs ZipNN into a win is preserved; and
    # it is bypassed entirely in max-ratio mode, where the smallest file wins
    # regardless of compress speed.
    if not _max_ratio and len(best.data) < _RANS_GATE_MARGIN * _order0_entropy_size(
        plane
    ):
        return best
    adapt_beat = (
        len(best.data)
        if _max_ratio
        else int(len(best.data) * (1.0 - _RANS_ADAPT_MIN_GAIN))
    )
    cand = _rans_candidate(plane, adapt_beat)
    if cand is not None:
        best = cand
    # Order-1 rANS: the conditional-entropy win on the small-alphabet exponent/sign
    # planes (z4ai's edge over ZipNN's order-0 Huffman on dense weights -- measured
    # +6.4% on distilgpt2's bf16 exponent over Zstd-LDM, +24% over order-0 rANS).
    # Gated on a small alphabet so the high-entropy mantissa pays no wasted pass,
    # and kept only when it beats the current best -- never a regression.
    if (
        parallel
        and _rans_o1 is not None
        and len(plane) >= _O1_MIN_BYTES
        and _distinct_bytes(plane) <= _O1_MAX_ALPHABET
    ):
        o1 = _rans_o1_candidate(plane, len(best.data))
        if o1 is not None:
            best = o1
    return best


def _compress_plane_zstd(
    plane: bytes,
    level: int,
    threads: int = 0,
    escalate_level: int | None = None,
    *,
    parallel: bool = True,
) -> CompressedPlane:
    """Compress a single byte plane with the adaptive Zstd policy.

    The plane is routed to whichever path wins for its statistics:

    * **block-parallel** (:data:`METHOD_ZSTD_BLOCKS`) for a large plane with no
      detectable long-range redundancy - maximum throughput, identical ratio;
    * **whole-plane LDM** (:data:`METHOD_ZSTD`) otherwise - preserves z4ai's
      long-range dedup ratio win, with Zstd multithreading for speed.

    Args:
        plane: the raw plane bytes.
        level: base Zstd compression level.
        threads: Zstd worker threads for the LDM path (``0`` -> auto = all cores).
        escalate_level: optional higher level to compress at instead of ``level``.
        parallel: set ``False`` to force the deterministic single-frame LDM path
            (used by tests that assert exact byte output).

    Returns:
        A :class:`CompressedPlane`.  When storing raw, ``method`` is
        :data:`METHOD_STORE` and ``data`` is the plane verbatim.
    """
    orig_len = len(plane)
    if orig_len == 0:
        return CompressedPlane(METHOD_STORE, b"", 0)

    use_level = escalate_level if escalate_level is not None else level

    # Always take the whole-plane LDM path (with Zstd's own multithreading for
    # speed).  We deliberately do NOT divert large planes to independent
    # block-parallel compression: real weight planes repeat a small alphabet of
    # values *globally* (e.g. a bf16 high-byte plane has only thousands of
    # distinct bytes) yet contain almost no exact 64 KiB-chunk duplicates and
    # barely compress in a small local window.  Both the long-range-redundancy
    # detector and a local probe therefore misjudge them as incompressible, and
    # block-splitting forfeits the cross-block dedup that is z4ai's entire ratio
    # win over ZipNN.  Measured on real bf16 weights: the block/probe fast path
    # stored the high-byte plane verbatim (8.4 MB) and produced an overall 1.47x
    # ratio (a LOSS to ZipNN's 1.51x), whereas whole-plane LDM - even at level 1,
    # whose large window finally sees the repeats - compresses the same plane to
    # 4.4 MB and the buffer to 2.76x (a clear WIN).  LDM at a low level is fast
    # (multi-GB/s) and the size guard below stores genuinely incompressible
    # planes (i.i.d. mantissa noise) verbatim, so we lose almost no throughput.
    mt = (os.cpu_count() or 1) if (parallel and threads == 0) else threads
    data = _compressor(use_level, orig_len, mt).compress(plane)

    # Size guard: incompressible planes do not shrink, so store them verbatim.
    if len(data) >= orig_len:
        return CompressedPlane(METHOD_STORE, plane, orig_len)

    # Decode-throughput best-of: also frame the plane as independent blocks and
    # keep that form when it is no larger than the single LDM frame.  A standard
    # Zstd frame decompresses on ONE core (Zstd's multithreading speeds only
    # *compression*), which is z4ai's weakest axis vs ZipNN's C core; independent
    # blocks decode across all cores (see :func:`_decompress_blocks`).
    #
    # This NEVER costs ratio, because we keep whichever is smaller:
    #   * A plane whose redundancy spans block boundaries - the structured /
    #     duplicated-layer win - compresses strictly smaller as one whole-plane
    #     LDM frame, so LDM is kept and the ratio win is preserved untouched.
    #   * A plane with no cross-block long-range structure (i.i.d. mantissa /
    #     exponent noise - the common dense-weight case) measures equal-or-smaller
    #     as blocks (LDM's whole-plane window is pure overhead on i.i.d. data),
    #     so it takes the block form and gains parallel decode for free.
    # Earlier code shipped a sampling LDM-gap *detector* to make this choice and
    # mis-stored real weights' small-alphabet high-byte plane (ratio loss); a
    # measured best-of removes the guesswork and can only help.  Gated on
    # ``parallel`` so the deterministic single-frame test path stays byte-exact.
    if parallel and orig_len >= _MIN_PARALLEL_PLANE:
        n_blocks = os.cpu_count() or 1
        block_size = max(_MIN_BLOCK_BYTES, (orig_len + n_blocks - 1) // n_blocks)
        if block_size < orig_len:  # need >= 2 blocks for any decode parallelism
            block_data = _compress_blocks(plane, use_level, block_size)
            # Keep blocks when they are no larger than the LDM frame OR within the
            # decode-tolerance band above (blocks decode ~5x faster across cores).
            if len(block_data) <= _BLOCK_DECODE_TOLERANCE * len(data):
                return CompressedPlane(METHOD_ZSTD_BLOCKS, block_data, orig_len)
    return CompressedPlane(METHOD_ZSTD, data, orig_len)


def decompress_plane(method: int, data: bytes, orig_len: int) -> bytes:
    """Invert :func:`compress_plane`."""
    if method == METHOD_STORE:
        # Return the payload as-is (a zero-copy memoryview slice when deserialize
        # handed one in).  A stored plane is consumed only by the inverse
        # transform -- byte-plane / bit-field join (NumPy ``frombuffer`` + the
        # native shuffle) and the opaque path -- all of which read it via the
        # buffer protocol, so copying it to ``bytes`` here just duplicated the
        # largest (incompressible) plane on every decode for nothing.
        return data
    if method == METHOD_ZSTD:
        # max_window_size matches the compressor cap so large-window (LDM)
        # frames always decode with bounded, predictable memory.
        dctx = zstd.ZstdDecompressor(max_window_size=1 << _MAX_WINDOW_LOG)
        out = dctx.decompress(data, max_output_size=orig_len)
        if len(out) != orig_len:
            raise ValueError(
                f"plane length mismatch: got {len(out)}, expected {orig_len}"
            )
        return out
    if method == METHOD_ZSTD_BLOCKS:
        return _decompress_blocks(data, orig_len)
    if method == METHOD_RANS:
        if _rans is None:
            raise ValueError(
                "frame uses the rANS method but the rANS module failed to "
                "import on this host"
            )
        # _rans.decompress falls back to a correct pure-Python decoder when the
        # native coder is unavailable, so rANS frames decode on any host.
        out = _rans.decompress(data)
        if len(out) != orig_len:
            raise ValueError(
                f"plane length mismatch: got {len(out)}, expected {orig_len}"
            )
        return out
    if method == METHOD_RANS_ADAPT:
        if _rans is None:
            raise ValueError(
                "frame uses the adaptive rANS method but the rANS module failed "
                "to import on this host"
            )
        out = _rans.decompress_adaptive(data)
        if len(out) != orig_len:
            raise ValueError(
                f"plane length mismatch: got {len(out)}, expected {orig_len}"
            )
        return out
    if method == METHOD_RANS_O1:
        if _rans_o1 is None:
            raise ValueError(
                "frame uses the order-1 rANS method but the rans_o1 module "
                "failed to import on this host"
            )
        out = _rans_o1.decompress(data)
        if len(out) != orig_len:
            raise ValueError(
                f"plane length mismatch: got {len(out)}, expected {orig_len}"
            )
        return out
    raise ValueError(f"unknown plane method {method}")
