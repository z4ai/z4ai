# SPDX-License-Identifier: Apache-2.0

"""Public z4ai codec API.

The codec follows ``API_CONTRACT.md``.  The two primary entry points are:

* :func:`compress` / :func:`decompress` - operate on raw ``bytes`` (or a NumPy
  array, or any buffer) and return / accept a single self-describing frame.
* :func:`compress_ndarray` / :func:`decompress_ndarray` - convenience wrappers
  that round-trip a NumPy array's dtype and shape exactly.

How z4ai beats byte-grouping codecs such as ZipNN
--------------------------------------------------
ZipNN splits a float tensor into *byte* planes.  That already exposes the
low-entropy exponent, but a byte boundary does not line up with a float's
field boundaries, so the exponent is split across planes and contaminated by
mantissa noise (see :mod:`z4ai.bitfield`).  For ``bf16``/``fp16``/``fp32`` z4ai
instead splits at the *bit-field* level - a pure exponent stream, a bit-packed
sign stream, and the mantissa - then compresses each stream with an adaptive
per-stream Zstd policy (high level on the tiny structured streams, raw-store on
the incompressible mantissa).  That yields a better ratio on the part that
matters and spends no CPU on the part that does not.

Both paths are lossless and bit-exact for every input length, including empty
buffers and lengths that are not a whole multiple of the element size.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np

import zstandard as zstd

from . import backend, bitfield, format, palette, sparse, transforms

# Default Zstd base/probe level.
#
# !!! DO NOT bump back to 3+ without re-running benchmarks/ablation.py !!!
# This was reverted to 3 once; here is the measured reason it must be 1:
# zstd ratio is NON-MONOTONIC on byte-transposed float weights - level 1 gives
# a BETTER ratio than 2-5 *and* is the fastest (the transpose already exposes
# the runs).  2000x2000 fp32 gaussian, per-plane, no escalation:
#     lvl1 -> 1.188 @ 26ms     lvl3 -> 1.171 @ 42ms     lvl5 -> 1.177 @ 74ms
# Full table + repro: benchmarks/TUNING.md.
DEFAULT_LEVEL = 1
# Escalation (recompressing "compressible" streams at a high level) cost ~700x
# wall-clock for <3% ratio and made the codec ~700-900x SLOWER THAN ZIPNN - the
# very thing we are trying to beat.  Disabled by default; pass escalate_level=
# explicitly for archival ("max ratio, time irrelevant") use.
DEFAULT_ESCALATE_LEVEL = None

# Level for the whole-buffer "opaque" best-of candidate.  This path exists to
# catch LONG-RANGE redundancy (tied embeddings, repeated / low-rank / zero
# layers, padding) that the byte/bit transpose scatters.  zstd ratio is again
# non-monotonic here but the OPPOSITE way to the transposed planes: level 1's
# small window MISSES long-period repeats (structured fp16: L1 -> 1.09x), while
# level 3 with long-distance matching catches them (-> 42x).  So this candidate
# always uses >= level 3 + LDM regardless of the (low) plane level above.
_OPAQUE_LEVEL = 3

Buffer = Union[bytes, bytearray, memoryview]

# Public dtype string code -> byte-plane width (used for the byte-plane path and
# by the CLI's ``--width`` resolution).  ``None`` / opaque buffers use width 1.
_DTYPE_WIDTH = {
    None: 1,
    "uint8": 1,
    "u8": 1,
    "int8": 1,
    "i8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int16": 2,
    "uint16": 2,
    "fp32": 4,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "fp64": 8,
    "float64": 8,
    "int64": 8,
    "uint64": 8,
}

# dtype codes that take the bit-field path (handled by bitfield.py).
#
# ONLY bf16 belongs here.  bf16's mantissa is exactly 7 bits, so the field
# split yields one clean byte per field with no waste.  fp16 (10-bit mantissa)
# and fp32 (23-bit mantissa) do NOT pack into whole bytes - splitting them
# rounds the mantissa up to 2/3-4 bytes per element and EXPANDS the data
# (measured: fp16 bit-field ratio 0.817, i.e. 22% larger than the input!).
# fp16/fp32 fall through to the byte-plane path, which is both correct and
# higher-ratio for them.  See benchmarks/TUNING.md.
_BITFIELD_DTYPES = {
    "bf16", "bfloat16",
}


def width_for_dtype(dtype: Optional[str]) -> int:
    """Resolve a public dtype string code to its byte-plane width.

    Raises:
        ValueError: if ``dtype`` is not a recognised code.
    """
    key = dtype.lower() if isinstance(dtype, str) else dtype
    try:
        return _DTYPE_WIDTH[key]
    except KeyError:
        raise ValueError(
            f"unknown dtype code {dtype!r}; expected one of "
            f"{sorted(c for c in _DTYPE_WIDTH if c)}"
        ) from None


# --------------------------------------------------------------------------- #
# Plane compression helpers
# --------------------------------------------------------------------------- #
# Process-wide pool for compressing a frame's planes concurrently.  A single
# plane's compression is a GIL-releasing native zstd call that does NOT saturate
# every core (the LDM/level-1 path on a few-MB plane uses only 1-3 effective
# threads), so overlapping the independent planes turns the compress wall-clock
# from the SUM of the per-plane times into ~their MAX.  Measured bf16 16 MiB: the
# three field planes compress 29.6 ms serially vs 17.7 ms overlapped (~1.7x), at
# byte-identical output (each plane compresses independently; result order is
# preserved).  Persisting the pool avoids per-call create/teardown, which
# dominates on the many small planes of a real multi-tensor checkpoint.  Distinct
# from backend._block_pool so a plane that itself block-parallelises never submits
# to the pool it is running on.
_COMPRESS_POOL: "ThreadPoolExecutor | None" = None

# Total plane bytes below which a frame compresses faster serially than across a
# pool: the per-plane work is smaller than the thread hand-off, so fanning out
# (let alone churning a fresh pool) is a net loss.  See :func:`_compress_planes`.
_MIN_PARALLEL_FRAME_BYTES = 1 << 20  # 1 MiB


def _compress_pool() -> ThreadPoolExecutor:
    global _COMPRESS_POOL
    if _COMPRESS_POOL is None:
        _COMPRESS_POOL = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 1, thread_name_prefix="z4ai-compress"
        )
    return _COMPRESS_POOL


def _compress_planes(
    planes: List[bytes],
    level: int,
    escalate_level: Optional[int],
    threads: int,
    plane_workers: int,
    rans: bool = True,
) -> List[backend.CompressedPlane]:
    """Compress every plane, fanning the independent planes out across threads.

    ``plane_workers`` lets a caller pin the fan-out; at the auto default (all
    cores) we overlap the planes on the shared :data:`_COMPRESS_POOL`, because no
    single plane saturates the cores and serial plane compression leaves most of
    them idle.  Output is byte-identical to the serial path.

    ``rans=False`` (the ``effort="fast"`` decode-speed tier) skips the rANS
    entropy candidates so every plane keeps the block-parallel Zstd form, which
    decodes ~6x faster across cores than serial rANS streams.
    """
    if len(planes) <= 1:
        return [
            backend.compress_plane(p, level, threads, escalate_level, rans=rans)
            for p in planes
        ]
    # A real checkpoint is mostly *small* tensors, whose planes each compress in
    # well under the ~50 us thread hand-off; there the pool - and especially
    # creating a fresh one per call - costs more than it saves.  Measured on 4x
    # 80 KiB fp32 planes: serial 130 us vs fresh-pool-per-call 295 us (2.3x).  So
    # compress tiny frames serially; only fan out once a frame is big enough that
    # the per-plane work dominates the hand-off.
    total = sum(len(p) for p in planes)
    if total < _MIN_PARALLEL_FRAME_BYTES:
        return [
            backend.compress_plane(p, level, threads, escalate_level, rans=rans)
            for p in planes
        ]
    # Honor an explicit *below-auto* fan-out with a scoped pool; otherwise (the
    # ``threads=0`` auto default, plane_workers = all cores) reuse the persistent
    # pool rather than churning a fresh ThreadPoolExecutor on every compress.
    cpu = os.cpu_count() or 1
    if 1 < plane_workers < cpu:
        with ThreadPoolExecutor(max_workers=plane_workers) as pool:
            return list(
                pool.map(
                    lambda p: backend.compress_plane(
                        p, level, threads, escalate_level, rans=rans
                    ),
                    planes,
                )
            )
    pool = _compress_pool()
    futures = [
        pool.submit(
            backend.compress_plane, p, level, threads, escalate_level, rans=rans
        )
        for p in planes
    ]
    return [f.result() for f in futures]


def _plane_entries(
    compressed: List[backend.CompressedPlane],
) -> List[format.PlaneEntry]:
    return [
        format.PlaneEntry(cp.method, cp.orig_len, cp.data) for cp in compressed
    ]


# Process-wide pool for the auto-mode best-of *side* work - the opaque LDM probe,
# the byte-plane candidate build, and the opaque candidate build - run
# concurrently with (and each other, behind) the primary plane compression (see
# :func:`compress`).  All are independent, GIL-releasing native zstd calls, so
# overlapping them hides the side passes behind the primary instead of paying
# them serially.  Three persistent workers cover the at-most-three side tasks
# (probe + byte-plane + opaque); persisting avoids per-compress create/teardown.
_BESTOF_POOL: "ThreadPoolExecutor | None" = None


def _bestof_pool() -> ThreadPoolExecutor:
    global _BESTOF_POOL
    if _BESTOF_POOL is None:
        _BESTOF_POOL = ThreadPoolExecutor(
            max_workers=max(3, (os.cpu_count() or 1) // 2), thread_name_prefix="z4ai-bestof"
        )
    return _BESTOF_POOL


# Plane decode stays a SERIAL walk - do NOT fan it across a thread pool.  It is
# tempting (each plane is an independent, GIL-releasing zstd decode) but measured
# as a regression on real float weights: only ONE plane does substantial decode
# work - the compressed exponent / high-byte plane - while the mantissa planes are
# stored verbatim (instant memcpy), and that one real plane is itself often
# METHOD_ZSTD_BLOCKS (already internally parallel on backend's block pool).  With
# no second heavy plane to overlap, an outer pool only adds thread hand-off and
# core contention: an interleaved A/B measured 0.86-0.93x (slower) on fp16/fp32/
# bf16.  The one plane that matters is parallelized inside the backend instead.
def _decompress_planes(container: format.Container) -> List[bytes]:
    return [
        backend.decompress_plane(p.method, p.data, p.orig_len)
        for p in container.planes
    ]


# --------------------------------------------------------------------------- #
# Transform planning
# --------------------------------------------------------------------------- #
def _plan(raw: bytes, dtype: Optional[str], width: Optional[int]):
    """Decide the transform for a raw buffer.

    Returns ``(transform, width, dtype_code, planes, tail)``.
    """
    key = dtype.lower() if isinstance(dtype, str) else dtype

    # 1) Bit-field path for the float formats that benefit most.
    if key in _BITFIELD_DTYPES:
        spec = bitfield.resolve_spec(key)
        itemsize = spec.itemsize
        tlen = len(raw) % itemsize
        body = raw[: len(raw) - tlen] if tlen else raw
        tail = raw[len(raw) - tlen :] if tlen else b""
        split = bitfield.split_fields(body, dtype=spec.name)
        planes = [split.sign, split.exponent, split.mantissa]
        return format.TRANSFORM_BITFIELD, 0, spec.name, planes, tail

    # 2) Byte-plane path (fp64, multi-byte ints, or an explicit width).
    if key is not None:
        w = width_for_dtype(key)
    elif width is not None:
        w = width
    else:
        w = 1
    if w < 1:
        raise ValueError("width must be >= 1")

    if w == 1:
        return format.TRANSFORM_OPAQUE, 1, "", [raw], b""

    tlen = transforms.tail_len(len(raw), w)
    body = raw[: len(raw) - tlen] if tlen else raw
    tail = raw[len(raw) - tlen :] if tlen else b""
    planes = transforms.split_planes(body, w)
    return format.TRANSFORM_BYTEPLANE, w, "", planes, tail


def _normalise_input(data) -> Tuple[bytes, bool, str, Tuple[int, ...], Optional[str]]:
    """Coerce ``data`` to raw bytes plus optional NumPy metadata.

    Returns ``(raw, is_numpy, np_dtype_str, shape, inferred_dtype_code)``.
    """
    if isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
        raw = arr.tobytes()
        return raw, True, arr.dtype.str, tuple(int(d) for d in arr.shape), _np_to_code(arr.dtype)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data), False, "", (), None
    # Best-effort: anything else exposing the buffer protocol (e.g. a torch
    # tensor's .numpy()).  Fall back to numpy coercion.
    arr = np.ascontiguousarray(np.asarray(data))
    return arr.tobytes(), True, arr.dtype.str, tuple(int(d) for d in arr.shape), _np_to_code(arr.dtype)


def _np_to_code(dt: np.dtype) -> Optional[str]:
    """Map a NumPy dtype to a public dtype code, or ``None`` for opaque."""
    mapping = {
        np.dtype("float16"): "fp16",
        np.dtype("float32"): "fp32",
        np.dtype("float64"): "fp64",
    }
    if dt in mapping:
        return mapping[dt]
    # Multi-byte integers benefit from a byte-plane transpose.
    if dt.kind in ("i", "u") and dt.itemsize in (2, 4, 8):
        return {2: "uint16", 4: "uint32", 8: "uint64"}[dt.itemsize]
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compress(
    data,
    *,
    dtype: Optional[str] = None,
    width: Optional[int] = None,
    level: Optional[int] = None,
    escalate_level: Optional[int] = DEFAULT_ESCALATE_LEVEL,
    threads: int = 0,
    plane_workers: int = 1,
    auto: bool = True,
    rans: bool = True,
) -> bytes:
    """Compress ``data`` into a self-describing z4ai frame.

    Args:
        data: ``bytes``/``bytearray``/``memoryview`` (raw buffer), a NumPy
            ``ndarray`` (dtype and shape are recorded and restored by
            :func:`decompress_ndarray`), or any buffer-protocol object.
        dtype: element type for a raw buffer, e.g. ``"bf16"``, ``"fp16"``,
            ``"fp32"``, ``"fp64"``.  ``None`` treats the buffer as opaque bytes.
            For ``bf16``/``fp16``/``fp32`` the high-ratio bit-field split is used.
        width: explicit byte-plane width (alternative to ``dtype`` for raw
            buffers / used by the CLI).  Ignored when ``dtype`` is given.
        level: base Zstd level (1..22).  ``None`` -> :data:`DEFAULT_LEVEL`.
        escalate_level: higher Zstd level applied to streams that probe as
            compressible; ``None`` disables escalation.
        threads: Zstd worker threads per stream (0 = single threaded).
        plane_workers: Python threads used to compress streams concurrently.
        auto: when True (default), also try a plain whole-buffer Zstd pass and
            keep whichever frame is smaller.  Guarantees z4ai never loses to
            plain Zstd and captures long-range redundancy the transpose would
            otherwise destroy, at the cost of a second compression pass.  Set
            ``False`` for maximum compression speed on i.i.d. weights.

    Returns:
        A ``bytes`` frame beginning with the magic ``b"Z4AI"``.
    """
    raw, is_numpy, np_dtype, shape, inferred = _normalise_input(data)
    if dtype is None and is_numpy:
        dtype = inferred

    resolved_level = DEFAULT_LEVEL if level is None else level

    transform, w, dtype_code, planes, tail = _plan(raw, dtype, width)

    # Opaque (non-float / width-1 / ``dtype=None``) inputs — token-id buffers,
    # attention masks, int/bool tensors, arbitrary bytes — are not byte-plane
    # transposable, so :func:`_plan` routes them to a single whole-buffer plane.
    # Compressed at the base level (1) for speed, that fell BELOW plain zstd-3 on
    # compressible buffers (measured: a uint8 model tensor 762x vs zstd-3 1415x;
    # such tensors are common in real checkpoints), and z4ai must NEVER lose to
    # plain Zstd.  Build them with the high-level (>= ``_OPAQUE_LEVEL``) + LDM
    # opaque builder instead — the same builder the float best-of uses as its
    # opaque candidate — which spans a whole-buffer window (catches long-range
    # redundancy) and stores verbatim when compression would not help
    # (never-expand).  In max-ratio mode (explicit ``escalate_level``) the opaque
    # build uses that higher level too.  This path is the primary for opaque
    # input, so the best-of / sparse branches below (which target float widths)
    # are correctly skipped.
    if transform == format.TRANSFORM_OPAQUE and raw:
        return _opaque_primary_frame(
            raw, resolved_level, escalate_level, is_numpy, np_dtype, shape
        )

    # Launch the auto-mode best-of side work CONCURRENTLY with the primary plane
    # compression.  The primary plane compress, the opaque LDM probe, and the
    # byte-plane candidate build are three independent, GIL-releasing native zstd
    # operations; run serially they dominated bf16 auto-mode compress (three
    # ~equal ~20 ms passes - the headline bf16 throughput gap).  Overlapping them
    # on a 2-worker pool hides the two side passes behind the primary, with
    # byte-identical output: the futures produce the exact same probe size /
    # candidate frame whether computed here or below, so the candidate set and the
    # min()/decode-tolerance choice are unchanged - only wall-clock moves.  Gated
    # to launch only work the best-of branch will actually consume (auto; the
    # byte-plane candidate exists only for a bit-field primary) and only for
    # buffers large enough that the thread hand-off pays for itself.  Measured
    # bf16 auto compress ~230 -> ~680 MB/s, ratio unchanged.
    bp_width = _itemsize_for(dtype, width)
    _large = bool(auto and raw and len(raw) >= _PROBE_MIN_BYTES)
    ldm_future = None
    byteplane_future = None
    if _large and transform != format.TRANSFORM_OPAQUE:
        ldm_future = _bestof_pool().submit(_ldm_probe_size, raw)
        if bp_width and bp_width > 1 and transform != format.TRANSFORM_BYTEPLANE:
            byteplane_future = _bestof_pool().submit(
                _byteplane_frame, raw, bp_width, resolved_level, escalate_level,
                threads, plane_workers, is_numpy, np_dtype, shape, rans,
            )

    compressed = _compress_planes(
        planes, resolved_level, escalate_level, threads, plane_workers, rans=rans
    )
    container = format.Container(
        planes=_plane_entries(compressed),
        transform=transform,
        width=w,
        dtype_code=dtype_code,
        tail=tail,
        is_numpy=is_numpy,
        dtype=np_dtype,
        shape=shape,
    )
    frame = format.serialize(container)

    # AUTO best-of (default): the float transform wins on i.i.d. weights, but a
    # byte/bit transpose DESTROYS the long-range redundancy (tied embeddings,
    # repeated and low-rank layers, LoRA adapters, padding) that a single
    # whole-buffer Zstd pass captures - there, plain Zstd can beat the transform
    # by 10x or more.  So unless ``auto`` is disabled, also build an opaque
    # whole-buffer frame and keep whichever is smaller.  This (a) makes z4ai
    # never worse than plain Zstd, (b) turns structured checkpoints into a large
    # win over ZipNN's fixed-size-chunk compression, and (c) doubles as the
    # never-expand safety net for inputs the float path mis-fits.  Skip the
    # extra pass when the transform was already an opaque Zstd pass.
    needs_alternatives = raw and transform != format.TRANSFORM_OPAQUE and (
        auto or len(frame) >= 0.98 * len(raw)
    )
    # Alternative-candidate policy.  Two alternatives can beat the primary frame,
    # and they compress *different byte orders*:
    #   * opaque    - compresses the raw *interleaved* buffer; wins on long-range
    #                 redundancy (tied embeddings, duplicated / low-rank layers).
    #   * byte-plane - compresses the *transposed* planes; wins when de-interleaving
    #                 exposes the exponent (real bf16) and ALSO decodes ~5x faster
    #                 than the bit-field frame.
    # The opaque candidate is the only *expensive* one (a high-level whole-buffer
    # LDM pass), so it is gated behind a cheap level-1 whole-buffer probe.  The
    # byte-plane candidate is NOT gated by a probe: a byte-plane *probe* re-runs a
    # whole-buffer level-1 pass over the same transposed planes the candidate
    # itself compresses, i.e. it costs as much as just building the candidate
    # (measured: byte-plane probe size == built frame size, ~equal wall-clock).
    # So we build the byte-plane frame directly and let the size comparison below
    # decide - strictly same-or-better ratio than the probe (it can keep a win the
    # probe would have mis-vetoed) and ~1.3-1.5x faster overall on bf16.  Probing
    # opaque (not a sampling hash detector) still matters because sampling misses
    # repeats whose period is unaligned to its stride.
    if needs_alternatives:
        has_byteplane_alt = bool(
            bp_width and bp_width > 1 and transform != format.TRANSFORM_BYTEPLANE
        )
        is_bitfield = transform == format.TRANSFORM_BITFIELD

        # The cheap opaque probe is trustworthy only for a large buffer the
        # transform already shrank; otherwise (small buffer, or the never-expand
        # safety net where frame >= 0.98*raw) just build it - cheap or required.
        cheap_probe = (
            auto and len(raw) >= _PROBE_MIN_BYTES and len(frame) < 0.95 * len(raw)
        )

        # Reuse the LDM probe started concurrently above when it was launched;
        # otherwise (small buffer / non-auto never-expand path) probe inline.
        if not cheap_probe:
            build_opaque = True
        else:
            ldm_size = (
                ldm_future.result() if ldm_future is not None
                else _ldm_probe_size(raw)
            )
            build_opaque = ldm_size < len(frame)

        # Launch the opaque candidate the moment the probe clears it, BEFORE
        # awaiting the byte-plane candidate, so the two run concurrently on the
        # best-of pool.  The opaque pass is the only *expensive* alternative (a
        # high-level whole-buffer LDM pass) and on structured / duplicated weights
        # - the headline win over ZipNN - it is usually the chosen frame, so it was
        # the last unoverlapped serial cost on that path.  The fast LDM probe has
        # already freed its pool slot, so opaque overlaps the byte-plane build
        # instead of running serially after it.  On i.i.d. data the probe leaves
        # build_opaque False, so nothing extra is built.  Output is byte-identical:
        # _opaque_frame is a pure function of its args, so the candidate set and the
        # min()/decode-tolerance choice are unchanged - only wall-clock moves.
        opaque_future = (
            _bestof_pool().submit(
                _opaque_frame, raw, resolved_level, is_numpy, np_dtype, shape
            )
            if build_opaque else None
        )

        # Build the byte-plane candidate directly (no probe - see policy note),
        # reusing the build started concurrently above when present.  The
        # byte-plane alternative only applies to the bit-field (bf16) primary; for
        # a byte-plane primary it would just re-compress the same planes.
        byteplane_cand = None
        if has_byteplane_alt:
            byteplane_cand = (
                byteplane_future.result() if byteplane_future is not None
                else _byteplane_frame(
                    raw, bp_width, resolved_level, escalate_level, threads,
                    plane_workers, is_numpy, np_dtype, shape, rans,
                )
            )

        candidates = [frame]
        if opaque_future is not None:
            candidates.append(opaque_future.result())
        if byteplane_cand is not None:
            candidates.append(byteplane_cand)
        best = min(candidates, key=len)

        # Decode-aware preference (bf16 only): the byte-plane frame decodes ~5x
        # faster than the bit-field frame (byte de-interleave vs per-element bit
        # unpacking) at near-identical size.  When the bit-field split is the
        # primary transform and the byte-plane candidate is within tolerance of
        # the smallest candidate, prefer byte-plane - trading <=1% ratio for a
        # large win on decompression speed, our weakest axis vs ZipNN.  Measured
        # bf16 mixture weights: +0.18% size, decode 767 -> 3790 MB/s (4.9x).
        if (
            byteplane_cand is not None
            and is_bitfield
            and len(byteplane_cand) <= _BYTEPLANE_DECODE_TOLERANCE * len(best)
        ):
            best = byteplane_cand
        frame = best

    # Sparse zero-aware candidate.  Pruned / structured-sparse checkpoints carry
    # many exact-zero weights; an explicit nonzero bitmap + densified values
    # (see :mod:`z4ai.sparse`) beats byte-grouping codecs such as ZipNN by a
    # widening margin as sparsity rises (bf16 50% 2.47x vs 2.20x; 95% 17.3x vs
    # 6.3x).  Gated on a cheap sparsity probe so dense weights pay nothing, and
    # kept only when it actually shrinks the frame, so it never hurts ratio.
    # Considered independently of the long-range best-of above because sparsity
    # is orthogonal to the redundancy detector.
    if auto and raw:
        sw = _itemsize_for(dtype, width)
        if sw and sparse.supported_width(sw) and sparse.should_use(raw, sw):
            sparse_frame = sparse.compress(
                raw, sw, level=resolved_level,
                is_numpy=is_numpy, np_dtype=np_dtype, shape=shape,
            )
            if len(sparse_frame) < len(frame):
                frame = sparse_frame

    # Palette (dictionary) candidate.  Quantised weights (INT4/INT8/FP8 via
    # GPTQ/AWQ/compressed-tensors) are increasingly shipped *dequantised into a
    # wide float container*, so a tensor holds only a small set of distinct values
    # spread thinly across the value space - which the byte/bit transpose does not
    # exploit.  Relabelling to a dense ``0..k-1`` index + tiny codebook (see
    # :mod:`z4ai.palette`) is a bijective, lossless transform that both shrinks the
    # stream (uint8 indices for k<=256) and gives Zstd a dense alphabet to crush
    # (measured: dequantised INT8 +11.7%, INT4 +27.5% vs the byte-plane path).
    # Gated on a cheap distinct-count probe so genuine full-entropy weights pay
    # nothing, and kept only when it actually shrinks the frame, so it never hurts
    # ratio.  Orthogonal to sparsity and to the long-range best-of above.
    if auto and raw:
        pw = _itemsize_for(dtype, width)
        if pw and palette.supported_width(pw) and palette.should_use(raw, pw):
            palette_frame = palette.compress(
                raw, pw, level=resolved_level,
                is_numpy=is_numpy, np_dtype=np_dtype, shape=shape,
            )
            if len(palette_frame) < len(frame):
                frame = palette_frame
    return frame


# Buffers below this size compress in well under a millisecond per pass, so the
# 3-pass best-of is already cheap and not worth a probe - keep the exact prior
# behaviour (always try the alternatives) there.
_PROBE_MIN_BYTES = 1 << 20  # 1 MiB

# bf16 decode-aware selection: prefer the byte-plane frame over the bit-field
# frame when within this size ratio, because byte-plane decodes ~5x faster.  1%
# keeps the ratio give-up negligible while capturing the decode-speed win on the
# common near-tie case.  See the selection logic in compress() for measurements.
_BYTEPLANE_DECODE_TOLERANCE = 1.01
_LDM_PROBE_LEVEL = -3


def _ldm_probe_size(raw: bytes) -> int:
    """Estimated compressed size of a fast *whole-buffer-window* pass over ``raw``.

    A *direct* measure of how much long-range redundancy a whole-buffer pass can
    capture - the only thing the opaque best-of candidate adds over the transposed
    frames.  It is robust where a sampling hash detector is not: an exact repeat
    whose period is unaligned to the sample stride (e.g. a 2 MiB-periodic tiled
    tensor) is invisible to sampling but caught here, because the match finder
    actually scans the buffer.  A window spanning the whole buffer lets the match
    finder see every repeat without paying for long-distance matching.  The probe
    runs at :data:`_LDM_PROBE_LEVEL` (a fast negative level) — see that constant
    for why weakening it is monotonically safe for ratio.
    """
    return _probe_window_size(raw)


def _probe_window_size(buf: bytes) -> int:
    """Compressed size of ``buf`` under a fast pass whose window spans it.

    The whole-buffer window lets the match finder see every repeat (no LDM
    overhead) - see :func:`_ldm_probe_size`.

    Implemented as a SINGLE ``compress()`` call rather than a streaming
    ``chunker()`` loop on purpose: this probe is started on a worker thread to
    run *concurrently* with the primary plane compression (see :func:`compress`),
    and zstd releases the GIL only for the duration of a native call.  A chunker
    loop returns to Python after every emitted chunk to run ``total += len(...)``,
    reacquiring the GIL each time and serialising against the plane-compression
    thread; one ``compress()`` is a single uninterrupted GIL-released call that
    genuinely overlaps it (measured: best-case fp32 auto compress 79 -> 59 ms).

    The returned size differs from the old streamed count by at most a few bytes
    of frame framing (a one-shot frame records the content size; the chunker's
    did not), which is immaterial: this value only *gates* whether an alternative
    candidate is built, a decision whose margins are megabytes wide, and the
    chosen frame is always the real ``min(len(...))`` of fully-built candidates -
    so the probe never affects the output bytes, only which passes we bother to
    run.  The transient compressed buffer (~0.9x the input) is discarded
    immediately and is far smaller than the buffers the codec already holds
    (raw + planes + frame), so it is not a meaningful addition to peak memory.
    """
    if not buf:
        return 0
    window_log = max(17, min(27, max(1, len(buf) - 1).bit_length()))
    params = zstd.ZstdCompressionParameters.from_level(
        _LDM_PROBE_LEVEL, window_log=window_log
    )
    return len(zstd.ZstdCompressor(compression_params=params).compress(buf))


def _itemsize_for(dtype: Optional[str], width: Optional[int]) -> int:
    """Element width (bytes) implied by ``dtype``/``width``; 0 if unknown."""
    key = dtype.lower() if isinstance(dtype, str) else dtype
    if key in _BITFIELD_DTYPES:
        return bitfield.resolve_spec(key).itemsize
    if key is not None:
        return width_for_dtype(key)
    if width:
        return width
    return 0


def _byteplane_frame(
    raw: bytes,
    w: int,
    level: int,
    escalate_level: Optional[int],
    threads: int,
    plane_workers: int,
    is_numpy: bool,
    np_dtype: str,
    shape: Tuple[int, ...],
    rans: bool = True,
) -> bytes:
    """Build a byte-plane (byte-grouping) frame for an explicit element width."""
    tlen = transforms.tail_len(len(raw), w)
    body = raw[: len(raw) - tlen] if tlen else raw
    tail = raw[len(raw) - tlen :] if tlen else b""
    planes = transforms.split_planes(body, w)
    compressed = _compress_planes(
        planes, level, escalate_level, threads, plane_workers, rans=rans
    )
    container = format.Container(
        planes=_plane_entries(compressed),
        transform=format.TRANSFORM_BYTEPLANE,
        width=w,
        dtype_code="",
        tail=tail,
        is_numpy=is_numpy,
        dtype=np_dtype,
        shape=shape,
    )
    return format.serialize(container)


def _opaque_frame(
    raw: bytes,
    level: int,
    is_numpy: bool,
    np_dtype: str,
    shape: Tuple[int, ...],
) -> bytes:
    """Build the smaller of an opaque whole-buffer Zstd frame and a verbatim
    store frame.

    Uses ``>= _OPAQUE_LEVEL`` with long-distance matching enabled so far-apart
    repeats (the structured-checkpoint win over ZipNN) are actually found; a low
    plane level would miss them.  LDM is transparent to the decoder (a standard
    Zstd frame), so :func:`decompress` needs no change.  Falls back to a
    verbatim store when compression would not help, so a frame is never larger
    than the input plus a small constant header.

    Two parameters matter for this candidate to deliver the structured-weights
    ratio win the DECISIONS.md invariant promises:

    * **a whole-buffer window** (``window_log`` spanning ``raw``).  zstd's default
      window for level 3 is only ~8 MiB, so without this the opaque pass cannot
      see repeats farther apart than that - the exact long-range redundancy this
      candidate exists to catch.  It also made the build inconsistent with the
      ``_ldm_probe_size`` probe (which already uses a whole-buffer window) that
      *decides* to build it: the probe promised a small frame the build could not
      deliver.  Measured on an 8x-duplicated bf16 buffer (4 MiB repeat period):
      ratio 1.29x -> 10.28x and ~3x faster (long matches compress quickly).
    * **multithreading** (``threads``).  zstd keeps LDM serial across job
      boundaries, so MT preserves the ratio while parallelising the match search
      - ~2x more on top for the structured case.  The output is still a single
      standard zstd frame, so :func:`decompress` is unchanged (its decoder window
      cap already matches ``_MAX_WINDOW_LOG``).
    """
    use_level = max(level, _OPAQUE_LEVEL)
    params = zstd.ZstdCompressionParameters.from_level(
        use_level,
        enable_ldm=True,
        window_log=backend._window_log_for(len(raw)),
        threads=os.cpu_count() or 0,
    )
    comp = zstd.ZstdCompressor(compression_params=params).compress(raw)
    if len(comp) < len(raw):
        method, payload, orig_len = backend.METHOD_ZSTD, comp, len(raw)
    else:
        method, payload, orig_len = backend.METHOD_STORE, raw, len(raw)
    container = format.Container(
        planes=[format.PlaneEntry(method, orig_len, payload)],
        transform=format.TRANSFORM_OPAQUE,
        width=1,
        dtype_code="",
        tail=b"",
        is_numpy=is_numpy,
        dtype=np_dtype,
        shape=shape,
    )
    return format.serialize(container)


def _opaque_primary_frame(
    raw: bytes,
    level: int,
    escalate_level: Optional[int],
    is_numpy: bool,
    np_dtype: str,
    shape: Tuple[int, ...],
) -> bytes:
    """Primary frame for opaque (non-float / width-1) input: best-of three.

    Non-float / arbitrary-byte tensors (token-id buffers, attention masks,
    int/bool buffers) have no float-field structure to exploit, so the only
    question is which Zstd configuration is smallest.  Two configs win on
    different inputs; we keep whichever is smaller, plus the verbatim-store guard:

    * **plain high-level** (default window, no LDM) — best on locally /
      short-period compressible data (e.g. a tiled attention mask).  This is the
      candidate that GUARANTEES z4ai never loses to plain Zstd at the same level
      (whole-window LDM + MT can trade a few % of ratio there — measured: a
      regular-period uint8 mask 963x with LDM+MT vs 1416x plain).
    * **whole-window + LDM + MT** — best on long-range redundancy (duplicated,
      padded, or tied buffers) that Zstd's small default window misses.

    The two passes are independent GIL-releasing zstd calls and run concurrently.
    Output is a standard ``TRANSFORM_OPAQUE`` frame the existing decoder reads.
    """
    use_level = max(level, _OPAQUE_LEVEL)
    if escalate_level is not None:
        use_level = max(use_level, escalate_level)

    def _ldm() -> bytes:
        params = zstd.ZstdCompressionParameters.from_level(
            use_level,
            enable_ldm=True,
            window_log=backend._window_log_for(len(raw)),
            threads=os.cpu_count() or 0,
        )
        return zstd.ZstdCompressor(compression_params=params).compress(raw)

    ldm_future = _bestof_pool().submit(_ldm)
    plain = zstd.ZstdCompressor(level=use_level).compress(raw)
    try:
        ldm = ldm_future.result()
    except Exception:  # noqa: BLE001 — never let the LDM pass break compression
        ldm = plain
    comp = plain if len(plain) <= len(ldm) else ldm

    if len(comp) < len(raw):
        method, payload, orig_len = backend.METHOD_ZSTD, comp, len(raw)
    else:
        method, payload, orig_len = backend.METHOD_STORE, raw, len(raw)
    container = format.Container(
        planes=[format.PlaneEntry(method, orig_len, payload)],
        transform=format.TRANSFORM_OPAQUE,
        width=1,
        dtype_code="",
        tail=b"",
        is_numpy=is_numpy,
        dtype=np_dtype,
        shape=shape,
    )
    return format.serialize(container)


def _reconstruct(container: format.Container) -> bytes:
    """Apply the inverse transform to a decoded container -> original bytes."""
    planes = _decompress_planes(container)

    if container.transform == format.TRANSFORM_OPAQUE:
        # A stored opaque plane is now a zero-copy memoryview (see
        # backend.decompress_plane); materialise it to bytes only here, where the
        # concatenation with the tail forces a single allocation anyway.
        body = bytes(planes[0]) if planes else b""
        return body + container.tail

    if container.transform == format.TRANSFORM_BYTEPLANE:
        return transforms.join_planes(planes, container.width, container.tail)

    if container.transform == format.TRANSFORM_BITFIELD:
        sign, exponent, mantissa = planes
        count = len(exponent)  # exponent stream is exactly one byte per element
        result = bitfield.SplitResult(
            spec_name=container.dtype_code,
            count=count,
            sign=sign,
            exponent=exponent,
            mantissa=mantissa,
        )
        return bitfield.join_fields(result) + container.tail

    raise ValueError(f"unknown transform {container.transform}")


def decompress(blob: bytes, *, out: Optional[bytearray] = None) -> bytes:
    """Decompress a z4ai frame back into the original buffer (byte-identical).

    Args:
        blob: a frame produced by :func:`compress`.
        out: optional writable buffer (``bytearray``/``memoryview``) to decode
            into; must be at least as large as the original.  When given, the
            decoded bytes are copied into it and ``out`` is returned.
    """
    if sparse.peek_is_sparse(blob):
        result = sparse.decompress(blob)
    elif palette.peek_is_palette(blob):
        result = palette.decompress(blob)
    else:
        container = format.deserialize(blob)
        result = _reconstruct(container)
    if out is not None:
        n = len(result)
        if len(out) < n:
            raise ValueError(
                f"out buffer too small: need {n}, have {len(out)}"
            )
        out[:n] = result
        return out
    return result


def compress_ndarray(arr: np.ndarray, **kwargs) -> bytes:
    """Compress a NumPy array, recording dtype and shape for exact restore."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("compress_ndarray expects a numpy.ndarray")
    return compress(arr, **kwargs)


def decompress_ndarray(blob: bytes) -> np.ndarray:
    """Decompress a frame created by :func:`compress_ndarray` into an array."""
    if sparse.peek_is_sparse(blob):
        meta = sparse.read_numpy_metadata(blob)
        if meta is None:
            raise ValueError("blob was not produced from a numpy array")
        dtype_str, shape = meta
        raw = sparse.decompress(blob)
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
    if palette.peek_is_palette(blob):
        meta = palette.read_numpy_metadata(blob)
        if meta is None:
            raise ValueError("blob was not produced from a numpy array")
        dtype_str, shape = meta
        raw = palette.decompress(blob)
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
    container = format.deserialize(blob)
    if not container.is_numpy:
        raise ValueError("blob was not produced from a numpy array")
    raw = _reconstruct(container)
    arr = np.frombuffer(raw, dtype=np.dtype(container.dtype))
    return arr.reshape(container.shape)


# Backwards-compatible aliases (earlier internal name).
compress_array = compress_ndarray
decompress_array = decompress_ndarray
