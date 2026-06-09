# SPDX-License-Identifier: Apache-2.0

"""Native static order-0 rANS entropy coder (the ratio edge over ZipNN's FSE).

This wraps the C kernel in ``_native/rans.c`` via :mod:`ctypes`.  The shared
library is built lazily with the system C compiler on first use; if no compiler
is available, :func:`available` returns ``False`` and callers fall back to the
Zstd backend (the package never hard-fails on import).

Public surface:

* :func:`available` - is the native coder usable?
* :func:`compress` / :func:`decompress` - self-describing rANS frames over bytes.

A frame is ``[u64 n][freq: 256 x u16 LE][payload]``.  The 256-entry frequency
table (512 bytes) is the order-0 model; it is negligible next to a multi-MB
exponent plane and makes the frame standalone.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import sysconfig
from struct import Struct
from threading import Lock
from typing import Optional
import warnings

import numpy as np

PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS  # 16384 - must match PROB_BITS in _native/rans.c
RANS_L = 1 << 23             # normalised-interval lower bound; must match _native/rans.c

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.join(_HERE, "_native")
_SRC = os.path.join(_NATIVE_DIR, "rans.c")
_LIB = os.path.join(_NATIVE_DIR, "libz4ai_rans" + (".dylib" if sys.platform == "darwin" else ".so"))

_HEADER = Struct("<Q")  # element count
_lib = None
_lib_lock = Lock()


def _compiler() -> Optional[str]:
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC")
    if cc:
        # CC may be e.g. "gcc -pthread"; take the program name.
        cc = cc.split()[0]
        if _which(cc):
            return cc
    for cand in ("clang", "cc", "gcc"):
        if _which(cand):
            return cand
    return None


def _which(prog: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, prog)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


# Compiler flag sets, tried in order from most- to least-optimized.  The kernel
# is built on the host that will run it (lazily, into the package dir), so it is
# safe to tune for the local CPU - exactly as the ``_native_shuffle`` extension
# does (see build_native.py / setup.py).  ``-march=native`` lets the compiler use
# the host's widest integer units for the interleaved rANS state updates (a real
# win on x86 AVX2 / older ARM where the baseline ISA is conservative); a host
# whose compiler rejects the flag simply falls through to the plain ``-O3`` build,
# and a host with no working compiler at all falls back to the pure-Python
# decoder.  The arch flag is gated to platforms where it exists (not MSVC/Windows).
_OPT_FLAGS = ["-O3", "-funroll-loops"]
if sys.platform.startswith(("darwin", "linux")):
    _BUILD_FLAG_SETS = [_OPT_FLAGS + ["-march=native"], _OPT_FLAGS, ["-O3"]]
else:  # pragma: no cover - non-POSIX toolchains: keep the conservative baseline
    _BUILD_FLAG_SETS = [_OPT_FLAGS, ["-O3"]]


def _build() -> bool:
    """Compile the native library if it is missing or stale.  Returns success.

    Tries progressively more conservative flag sets so an arch-tuned build is
    used where the toolchain supports it, while a compiler that rejects
    ``-march=native`` (or ``-funroll-loops``) still produces a working library
    rather than silently dropping to the slow pure-Python decoder.
    """
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return True
    cc = _compiler()
    if cc is None:
        return False
    for flags in _BUILD_FLAG_SETS:
        cmd = [cc, *flags, "-shared", "-fPIC", "-o", _LIB, _SRC]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError):
            continue
        if os.path.exists(_LIB):
            return True
    return False


def _load():
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        if not _build():
            return None
        lib = ctypes.CDLL(_LIB)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        lib.z4ai_rans_encode.restype = ctypes.c_size_t
        lib.z4ai_rans_encode.argtypes = [u8p, ctypes.c_size_t, u16p, u16p, u8p, ctypes.c_size_t]
        lib.z4ai_rans_decode.restype = None
        lib.z4ai_rans_decode.argtypes = [u8p, ctypes.c_size_t, u16p, u16p, u8p, u8p, ctypes.c_size_t]
        # Block-wise decoder that builds cum/slot2sym in C from freq.  Lets the
        # per-chunk LOCAL-model path (compress_adaptive) decode each chunk without
        # rebuilding the 16384-entry slot table in NumPy per chunk (that overhead,
        # not the kernel, capped small-block decode at ~375 MB/s vs ~2.4 GB/s raw).
        try:
            lib.z4ai_rans_decode_f.restype = None
            lib.z4ai_rans_decode_f.argtypes = [u8p, ctypes.c_size_t, u16p, u8p, ctypes.c_size_t]
        except AttributeError:  # pragma: no cover - only with a stale prebuilt lib
            pass
        # One-call local-model chunk encoder (histogram+normalize+encode in C, GIL
        # released).  Collapses the per-chunk Python/ctypes overhead of the adaptive
        # path; absent in a stale prebuilt lib (guarded — falls back to Python).
        try:
            lib.z4ai_rans_encode_local.restype = ctypes.c_size_t
            lib.z4ai_rans_encode_local.argtypes = [u8p, ctypes.c_size_t, u8p, ctypes.c_size_t, u16p]
        except AttributeError:  # pragma: no cover - only with a stale prebuilt lib
            pass
        # Optional fast native histogram (present in current rans.c; absent in an
        # older prebuilt lib — guarded so a stale .so still loads and we fall back
        # to NumPy for the model build).
        u32p = ctypes.POINTER(ctypes.c_uint32)
        try:
            lib.z4ai_rans_hist.restype = None
            lib.z4ai_rans_hist.argtypes = [u8p, ctypes.c_size_t, u32p]
        except AttributeError:  # pragma: no cover - only with a stale prebuilt lib
            pass
        _lib = lib
        return _lib


def _histogram(lib, src: np.ndarray) -> np.ndarray:
    """256-bin byte histogram of ``src`` as int64 counts.

    Uses the native ``z4ai_rans_hist`` (GIL-released, folded across threads for a
    large buffer) when available; otherwise NumPy ``bincount``.  The model-build
    histogram dominated rANS encode wall-clock (~80%: NumPy bincount over a few-
    distinct-value plane is pathologically slow), so this is the throughput lever
    that makes default-on rANS practical.
    """
    n = src.size
    if not hasattr(lib, "z4ai_rans_hist"):
        return np.bincount(src, minlength=256).astype(np.int64)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u32 = ctypes.POINTER(ctypes.c_uint32)

    def _one(a: np.ndarray) -> np.ndarray:
        h = np.zeros(256, dtype=np.uint32)
        lib.z4ai_rans_hist(a.ctypes.data_as(u8), ctypes.c_size_t(a.size),
                           h.ctypes.data_as(u32))
        return h

    if n < 2 * _MIN_CHUNK_BYTES:
        return _one(src).astype(np.int64)
    k = _chunk_count(n)
    bounds = _bounds(n, k)
    parts = list(_thread_pool().map(lambda b: _one(src[b[0]:b[1]]), bounds))
    total = parts[0].astype(np.int64)
    for p in parts[1:]:
        total += p
    return total


def available() -> bool:
    """Return True if the native rANS kernel is built and loadable."""
    try:
        return _load() is not None
    except OSError:
        return False


def _normalize(hist: np.ndarray) -> np.ndarray:
    """Scale a 256-bin histogram to integer frequencies summing to PROB_SCALE.

    Every symbol that occurs gets at least frequency 1 (rANS cannot code a
    zero-probability symbol).  Rounding error is absorbed by the most frequent
    symbol, which keeps the model within ~1e-4 bits/symbol of the true entropy.
    """
    total = int(hist.sum())
    freq = np.zeros(256, dtype=np.int64)
    if total == 0:
        return freq.astype(np.uint16)
    nz = hist > 0
    scaled = hist.astype(np.float64) * (PROB_SCALE / total)
    f = np.floor(scaled).astype(np.int64)
    f[nz & (f == 0)] = 1
    diff = PROB_SCALE - int(f.sum())
    if diff > 0:
        f[int(np.argmax(f))] += diff
    elif diff < 0:
        need = -diff
        for i in np.argsort(-f):
            if need == 0:
                break
            room = f[i] - 1
            if room > 0:
                take = min(room, need)
                f[i] -= take
                need -= take
    return f.astype(np.uint16)


def _tables(freq: np.ndarray):
    cum = np.zeros(256, dtype=np.uint16)
    cum[1:] = np.cumsum(freq[:-1].astype(np.uint32)).astype(np.uint16)
    slot2sym = np.repeat(np.arange(256, dtype=np.uint8), freq.astype(np.int64))
    return cum, slot2sym


# --------------------------------------------------------------------------- #
# Chunked / multi-threaded rANS
# --------------------------------------------------------------------------- #
# A single rANS state chain is inherently sequential, so one stream tops out at
# the ~200-350 MB/s of the scalar kernel - too slow to be the *default* entropy
# coder.  But a buffer can be split into independent chunks that all share ONE
# order-0 model and are coded as separate streams: the streams have no data
# dependency, so they encode/decode fully in PARALLEL across a thread pool, and
# because ``ctypes`` releases the GIL around each native call the speedup is real
# (measured 219/338 -> ~2025/1898 MB/s on a 16-core host, ~9x/5.6x).  The shared
# model means the ratio is IDENTICAL to the single-stream frame (chunking adds
# only ~20 bytes/chunk of flushed states + a length, negligible next to a multi-MB
# plane).  This is what lets z4ai turn rANS on BY DEFAULT for the low-entropy
# exponent/sign/mantissa planes - reaching the order-0 entropy floor (the ratio
# edge over ZipNN's integer-bit Huffman) without the old throughput tar pit.
#
# Frame compatibility: a chunked frame sets the high bit of the u64 count header
# (``_CHUNK_FLAG``); a legacy single-stream frame never does (a real element count
# never reaches 2^63), so :func:`decompress` dispatches on that bit and still
# decodes every previously-written frame byte-for-byte.
_CHUNK_FLAG = 1 << 63
_CHUNK_HDR = Struct("<QH")        # (count | flag, n_chunks)
_LEN32 = Struct("<I")             # per-chunk compressed length
# Below this a single stream already codes in well under a millisecond, so the
# thread-pool fan-out would cost more than it saves: code it as one chunk.
_MIN_CHUNK_BYTES = 1 << 19        # 512 KiB
_MAX_CHUNKS = 64

_pool_lock = Lock()
_pool: "object" = None


def _thread_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from concurrent.futures import ThreadPoolExecutor
                _pool = ThreadPoolExecutor(
                    max_workers=os.cpu_count() or 1, thread_name_prefix="z4ai-rans"
                )
    return _pool


def _chunk_count(n: int) -> int:
    """Number of parallel chunks for an ``n``-byte buffer (1 when small)."""
    if n < 2 * _MIN_CHUNK_BYTES:
        return 1
    workers = os.cpu_count() or 1
    return max(1, min(_MAX_CHUNKS, workers, n // _MIN_CHUNK_BYTES))


def _bounds(n: int, k: int):
    """Deterministic chunk boundaries shared by encoder and decoder."""
    return [(i * n // k, (i + 1) * n // k) for i in range(k)]


def _encode_one(lib, src: np.ndarray, freq: np.ndarray, cum: np.ndarray) -> bytes:
    n = src.size
    cap = n * 2 + 1024
    out = np.empty(cap, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    length = lib.z4ai_rans_encode(
        src.ctypes.data_as(u8), ctypes.c_size_t(n),
        freq.ctypes.data_as(u16), cum.ctypes.data_as(u16),
        out.ctypes.data_as(u8), ctypes.c_size_t(cap),
    )
    if length == 0:
        raise RuntimeError("rANS encode overflow")
    return out[:length].tobytes()


def compress(data: bytes) -> bytes:
    """rANS-compress a byte buffer into a self-describing frame.

    Large buffers are split into independent chunks coded in parallel under one
    shared order-0 model (see the module note above): byte-for-byte the same
    ratio as a single stream, but ~9x faster to encode on a multi-core host.

    Raises :class:`RuntimeError` if the native coder is unavailable.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("native rANS unavailable (no C compiler?)")
    src = np.frombuffer(data, dtype=np.uint8)
    n = src.size
    if n == 0:
        return _HEADER.pack(0)

    hist = _histogram(lib, src)
    freq = _normalize(hist)
    cum, _ = _tables(freq)

    k = _chunk_count(n)
    if k == 1:
        payload = _encode_one(lib, src, freq, cum)
        return _HEADER.pack(n) + freq.tobytes() + payload

    bounds = _bounds(n, k)
    pool = _thread_pool()
    payloads = list(
        pool.map(lambda b: _encode_one(lib, src[b[0]:b[1]], freq, cum), bounds)
    )
    header = _CHUNK_HDR.pack(n | _CHUNK_FLAG, k) + freq.tobytes()
    lens = b"".join(_LEN32.pack(len(p)) for p in payloads)
    return header + lens + b"".join(payloads)


_warned_fallback = False


def _warn_fallback_once() -> None:
    global _warned_fallback
    if not _warned_fallback:
        _warned_fallback = True
        warnings.warn(
            "z4ai: decoding a rANS frame without the native coder (no C "
            "compiler on this host) - using the pure-Python fallback, which is "
            "correct but much slower. Install a C compiler for native speed.",
            RuntimeWarning,
            stacklevel=2,
        )


_N_LANES = 4  # interleaved rANS states; must match N_LANES in _native/rans.c


def _decode_python(payload: bytes, freq: np.ndarray, n: int) -> bytes:
    """Pure-Python rANS decode - the portability fallback when no native lib.

    Mirrors ``z4ai_rans_decode`` in ``_native/rans.c`` byte-for-byte so frames
    produced on a host *with* a C compiler stay decodable on a host *without*
    one (preserving z4ai's "decodes anywhere" guarantee).  Like the C decoder it
    runs four interleaved states, but the lanes are inherently sequential here
    (one Python loop), so this stays a slow correctness fallback only.
    """
    cum, slot2sym = _tables(freq)
    # Plain Python ints/lists are markedly faster than NumPy scalar indexing in
    # this tight, inherently-sequential loop.
    fr = freq.astype(np.int64).tolist()
    cu = cum.astype(np.int64).tolist()
    s2s = slot2sym.tobytes()           # indexing a bytes object yields an int
    out = bytearray(n)
    data = payload
    dlen = len(data)
    mask = PROB_SCALE - 1
    L = RANS_L

    # Seed the four interleaved states from the front (state 0 first), matching
    # the C encoder's flush order.
    x = [0] * _N_LANES
    for l in range(_N_LANES):
        o = l * 4
        x[l] = data[o] | (data[o + 1] << 8) | (data[o + 2] << 16) | (data[o + 3] << 24)
    pos = 4 * _N_LANES

    for i in range(n):
        lane = i & (_N_LANES - 1)
        xi = x[lane]
        slot = xi & mask
        s = s2s[slot]
        out[i] = s
        xi = fr[s] * (xi >> PROB_BITS) + slot - cu[s]
        while xi < L:
            nb = data[pos] if pos < dlen else 0
            pos += 1
            xi = (xi << 8) | nb
        x[lane] = xi
    return bytes(out)


def _decode_one(lib, payload: bytes, freq, cum, slot2sym, n: int) -> bytes:
    """Native decode of one rANS stream into an ``n``-byte buffer."""
    payload_arr = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    lib.z4ai_rans_decode(
        payload_arr.ctypes.data_as(u8), ctypes.c_size_t(payload_arr.size),
        freq.ctypes.data_as(u16), cum.ctypes.data_as(u16),
        slot2sym.ctypes.data_as(u8), out.ctypes.data_as(u8), ctypes.c_size_t(n),
    )
    return out.tobytes()


def decompress(blob: bytes) -> bytes:
    """Inverse of :func:`compress` (handles both single-stream and chunked frames).

    Uses the native coder when available; otherwise falls back to a correct
    (but slow) pure-Python decoder so rANS frames remain decodable on any host.
    """
    (raw_n,) = _HEADER.unpack_from(blob, 0)
    if raw_n == 0:
        return b""

    lib = _load()

    # Chunked frame: high bit of the count is the chunk flag.
    if raw_n & _CHUNK_FLAG:
        n = raw_n & ~_CHUNK_FLAG
        _, k = _CHUNK_HDR.unpack_from(blob, 0)
        off = _CHUNK_HDR.size
        freq = np.frombuffer(blob, dtype="<u2", count=256, offset=off).astype(np.uint16)
        off += 512
        clens = []
        for _ in range(k):
            (cl,) = _LEN32.unpack_from(blob, off)
            off += _LEN32.size
            clens.append(cl)
        payloads = []
        for cl in clens:
            payloads.append(bytes(blob[off:off + cl]))
            off += cl
        bounds = _bounds(n, k)
        sizes = [hi - lo for lo, hi in bounds]
        cum, slot2sym = _tables(freq)
        if lib is None:
            _warn_fallback_once()
            parts = [
                _decode_python(payloads[i], freq, sizes[i]) for i in range(k)
            ]
        elif k == 1:
            parts = [_decode_one(lib, payloads[0], freq, cum, slot2sym, sizes[0])]
        else:
            pool = _thread_pool()
            parts = list(
                pool.map(
                    lambda i: _decode_one(
                        lib, payloads[i], freq, cum, slot2sym, sizes[i]
                    ),
                    range(k),
                )
            )
        return b"".join(parts)

    # Legacy single-stream frame.
    n = raw_n
    off = _HEADER.size
    freq = np.frombuffer(blob, dtype="<u2", count=256, offset=off).astype(np.uint16)
    off += 512
    payload = bytes(blob[off:])
    if lib is None:
        _warn_fallback_once()
        return _decode_python(payload, freq, n)
    cum, slot2sym = _tables(freq)
    return _decode_one(lib, payload, freq, cum, slot2sym, n)


# --------------------------------------------------------------------------- #
# Per-chunk LOCAL-model adaptive rANS
# --------------------------------------------------------------------------- #
# :func:`compress` above shares ONE order-0 model across all chunks, so it codes
# at the *global* entropy of the buffer.  That is optimal for a homogeneous plane
# but LOSES on a concatenation of regions with different symbol distributions -
# e.g. a model's exponent plane, where each tensor has its own exponent centre.
# Measured on a 64 MB multi-tensor bf16 blob: the GLOBAL exponent entropy is 3.29
# bits, but the average LOCAL (per-256 KiB-block) entropy is only 2.55 bits.  A
# shared-model coder reaches ~2.43x on that plane; a per-chunk LOCAL model reaches
# ~3.0x - the difference between losing to and beating ZipNN's chunked entropy
# coder on dense real weights.
#
# So :func:`compress_adaptive` gives every chunk its OWN frequency table (built
# from that chunk's bytes), capturing the local distribution.  Chunks are still
# independent, so encode/decode run in PARALLEL on the shared pool; decode uses
# the native ``z4ai_rans_decode_f`` (builds cum/slot2sym in C) so the per-chunk
# table build does not move back into Python.  The cost is one 512-byte freq table
# per chunk (~0.2% of a 256 KiB chunk's worth of exponents) - dwarfed by the
# 0.74-bit/symbol local-adaptation win.
#
# Frame: ``[u64 n | _ADAPT_FLAG][u32 k]`` then ``k`` x ``[u32 clen]`` then ``k`` x
# 512-byte freq tables then the concatenated payloads.  Self-describing; decoded
# only via :func:`decompress_adaptive` (the codec dispatches on its plane method),
# so it never collides with :func:`decompress`'s single-stream/shared-chunk frames.
_ADAPT_FLAG = 1 << 62
_ADAPT_HDR = Struct("<QII")         # (count | _ADAPT_FLAG, n_chunks, freq_blob_len)
# Target chunk size.  64 KiB is small enough that a chunk rarely straddles two
# tensors (so it tracks each tensor's local exponent distribution even for the
# many small tensors of a real checkpoint - measured exp ratio 2.75x at 256 KiB
# vs 2.95x at 64 KiB on a 256-tensor blob, approaching the 3.11x per-tensor
# ideal) while the chunk count stays modest.  The per-chunk 512-byte freq tables
# would otherwise cost ~2.3% at this size, so they are zstd-packed together
# (they are sparse and similar -> ~6% of raw) before framing.
_ADAPT_CHUNK_BYTES = 1 << 16


def _pack_freqs(freq_bytes: bytes) -> bytes:
    """zstd-pack the concatenated per-chunk freq tables (sparse + similar -> ~6%)."""
    import zstandard as _zstd
    return _zstd.ZstdCompressor(level=3).compress(freq_bytes)


def _unpack_freqs(blob: bytes, n_tables: int):
    import zstandard as _zstd
    raw = _zstd.ZstdDecompressor().decompress(blob, max_output_size=n_tables * 512)
    return [
        np.frombuffer(raw, dtype="<u2", count=256, offset=i * 512).astype(np.uint16)
        for i in range(n_tables)
    ]


def _adapt_bounds(n: int):
    k = max(1, (n + _ADAPT_CHUNK_BYTES - 1) // _ADAPT_CHUNK_BYTES)
    return _bounds(n, k), k


def _encode_chunk_local(lib, chunk: np.ndarray):
    """Encode one chunk under its OWN local model; return (freq_bytes, payload).

    Builds only ``cum`` (a 256-entry cumsum), NOT the 16384-entry ``slot2sym``
    table that :func:`_tables` also returns: the encoder never uses slot2sym, and
    rebuilding it per chunk (np.repeat over 16384 slots) was the bulk of the
    small-chunk encode cost.
    """
    n = chunk.size
    # Fast path: one GIL-releasing C call does histogram + normalise + encode, so
    # the thread pool actually parallelises (no per-chunk Python/ctypes overhead).
    if n and hasattr(lib, "z4ai_rans_encode_local"):
        freq = np.empty(256, dtype=np.uint16)
        cap = n * 2 + 1024
        out = np.empty(cap, dtype=np.uint8)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        u16 = ctypes.POINTER(ctypes.c_uint16)
        length = lib.z4ai_rans_encode_local(
            chunk.ctypes.data_as(u8), ctypes.c_size_t(n),
            out.ctypes.data_as(u8), ctypes.c_size_t(cap),
            freq.ctypes.data_as(u16),
        )
        if length == 0:
            raise RuntimeError("local-model rANS encode overflow")
        return freq.tobytes(), out[:length].tobytes()
    # Fallback (stale lib without the one-call encoder): build the model in Python.
    freq = _normalize(_histogram(lib, chunk))
    cum = np.zeros(256, dtype=np.uint16)
    cum[1:] = np.cumsum(freq[:-1].astype(np.uint32)).astype(np.uint16)
    return freq.tobytes(), _encode_one(lib, chunk, freq, cum)


def compress_adaptive(data: bytes) -> bytes:
    """rANS-compress with a PER-CHUNK local order-0 model (see module note).

    Byte-for-byte lossless; decoded by :func:`decompress_adaptive`.  Falls back to
    a single shared-model stream is NOT done here - callers that want the global
    model use :func:`compress`.  Raises if the native coder is unavailable.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("native rANS unavailable (no C compiler?)")
    src = np.frombuffer(data, dtype=np.uint8)
    n = src.size
    if n == 0:
        return _HEADER.pack(0)
    bounds, k = _adapt_bounds(n)
    if k == 1:
        results = [_encode_chunk_local(lib, src)]
    else:
        results = list(
            _thread_pool().map(lambda b: _encode_chunk_local(lib, src[b[0]:b[1]]), bounds)
        )
    freq_blob = _pack_freqs(b"".join(f for f, _ in results))
    header = _ADAPT_HDR.pack(n | _ADAPT_FLAG, k, len(freq_blob))
    lens = b"".join(_LEN32.pack(len(p)) for _, p in results)
    payloads = b"".join(p for _, p in results)
    return header + lens + freq_blob + payloads


def is_adaptive(blob: bytes) -> bool:
    """True if ``blob`` is a per-chunk adaptive frame (see :func:`compress_adaptive`)."""
    if len(blob) < _HEADER.size:
        return False
    (raw_n,) = _HEADER.unpack_from(blob, 0)
    return bool(raw_n & _ADAPT_FLAG)


def _decode_chunk_f(lib, payload: bytes, freq: np.ndarray, n: int) -> bytes:
    """Native decode of one local-model chunk (cum/slot2sym built in C)."""
    if lib is None or not hasattr(lib, "z4ai_rans_decode_f"):
        return _decode_python(payload, freq, n)
    payload_arr = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    lib.z4ai_rans_decode_f(
        payload_arr.ctypes.data_as(u8), ctypes.c_size_t(payload_arr.size),
        freq.ctypes.data_as(u16), out.ctypes.data_as(u8), ctypes.c_size_t(n),
    )
    return out.tobytes()


def decompress_adaptive(blob: bytes) -> bytes:
    """Inverse of :func:`compress_adaptive`."""
    (raw_n,) = _HEADER.unpack_from(blob, 0)
    if raw_n == 0:
        return b""
    if not (raw_n & _ADAPT_FLAG):
        raise ValueError("not a z4ai adaptive rANS frame")
    n = raw_n & ~_ADAPT_FLAG
    _, k, freq_blob_len = _ADAPT_HDR.unpack_from(blob, 0)
    off = _ADAPT_HDR.size
    clens = []
    for _ in range(k):
        (cl,) = _LEN32.unpack_from(blob, off)
        off += _LEN32.size
        clens.append(cl)
    freqs = _unpack_freqs(blob[off:off + freq_blob_len], k)
    off += freq_blob_len
    payloads = []
    for cl in clens:
        payloads.append(bytes(blob[off:off + cl]))
        off += cl
    bounds = _bounds(n, k)
    lib = _load()
    if lib is None or not hasattr(lib, "z4ai_rans_decode_f"):
        # Portability fallback: decode each chunk in pure Python.
        if lib is None:
            _warn_fallback_once()
        parts = [
            _decode_python(payloads[i], freqs[i], hi - lo)
            for i, (lo, hi) in enumerate(bounds)
        ]
        return b"".join(parts)

    # Decode every chunk straight into its slice of ONE output buffer: no
    # per-chunk tobytes / final join (which dominated decode for the many small
    # chunks the local model uses).  Threads write disjoint slices, so the
    # parallel decode is race-free.
    out = np.empty(n, dtype=np.uint8)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    u16 = ctypes.POINTER(ctypes.c_uint16)

    def _dec(i):
        lo, hi = bounds[i]
        pa = np.frombuffer(payloads[i], dtype=np.uint8)
        sub = out[lo:hi]
        lib.z4ai_rans_decode_f(
            pa.ctypes.data_as(u8), ctypes.c_size_t(pa.size),
            freqs[i].ctypes.data_as(u16), sub.ctypes.data_as(u8),
            ctypes.c_size_t(hi - lo),
        )

    if k == 1:
        _dec(0)
    else:
        list(_thread_pool().map(_dec, range(k)))
    return out.tobytes()


def decode_is_portable() -> bool:
    """rANS frames are always decodable (native fast path or Python fallback)."""
    return True
