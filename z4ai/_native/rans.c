/* Copyright 2026 The z4ai Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Static order-0 rANS (range Asymmetric Numeral System) over a byte alphabet.
 *
 * Classic byte-wise rANS (32-bit state, 8-bit renorm, 14-bit probability
 * precision) reaching the order-0 entropy floor exactly - better than Zstd's
 * literal Huffman, and the point of dropping into C: a real ratio edge on the
 * exponent/sign planes of float weights (where ZipNN's order-0 FSE also lives)
 * without Zstd-level-19's ~1 MB/s tar pit.
 *
 * Two implementation choices make this kernel several times faster than a
 * textbook single-stream rANS, so it is no longer a throughput tar pit either:
 *
 *   1. **4-way interleaving.** Four independent rANS states code symbols
 *      i, i+1, i+2, i+3 in lockstep over one shared byte stream.  The four
 *      state-update chains are data-independent, so the CPU keeps four of them
 *      in flight at once instead of stalling on the latency of a single serial
 *      chain - the standard ILP win for rANS.
 *   2. **Reciprocal-multiply encoding.** The encoder's `x / freq` (a ~20-40
 *      cycle integer division that does not pipeline) is replaced by a
 *      multiply-and-shift using a per-symbol magic reciprocal (the classic
 *      "division by an invariant integer" transform, after Fabian Giesen's
 *      public-domain ryg_rans).  Multiplies pipeline fully, so combined with
 *      interleaving the divider is no longer the bottleneck.
 *
 * The frequency model (`freq`, `cum`, `slot2sym`) is built in Python and passed
 * in; the encoder derives its reciprocal table from `freq`/`cum` internally so
 * the ctypes ABI is unchanged.  Encoding processes symbols in reverse and emits
 * renorm bytes downward; the four 4-byte final states land at the front of the
 * stream (state 0 first), so the decoder seeds its four states and then consumes
 * renorm bytes forward.  Fully reversible and byte-exact for any input length.
 *
 * Wire format of the payload (after Python's [u64 n][freq 512] header):
 *   [state0:4 LE][state1:4 LE][state2:4 LE][state3:4 LE][renorm bytes ...]
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#define PROB_BITS 14u
#define PROB_SCALE (1u << PROB_BITS) /* 16384 - 14-bit model resolution tracks
                                      * the true symbol distribution closely
                                      * enough to recover ~0.2-0.4% ratio on
                                      * low-entropy exponent planes (enough to
                                      * turn the ZipNN tie into a win).  Safe:
                                      * RANS_L stays a multiple of PROB_SCALE and
                                      * x_max peaks at 2^31 < 2^32. */
#define RANS_L (1u << 23)            /* lower bound of the normalised interval */
#define PROB_MASK (PROB_SCALE - 1u)
#define N_LANES 4                    /* interleaved rANS states.  4 is the measured
                                      * sweet spot: it fills the state-update
                                      * latency without the register pressure that
                                      * made 8 lanes ~10% SLOWER on the test core
                                      * (and 8 also costs a few more flush bytes). */

/* Per-symbol encoder constants: a reciprocal-multiply replacement for the
 * coding step x -> (x / freq) * PROB_SCALE + (x % freq) + cum.  Identity:
 *   q  = floor(x / freq)            (computed as a multiply-shift)
 *   x' = x + bias + q * cmpl_freq   (== (x%freq) + q*PROB_SCALE + cum)         */
typedef struct {
    uint32_t x_max;     /* renorm threshold: emit a byte while x >= x_max */
    uint32_t rcp_freq;  /* magic reciprocal of freq */
    uint32_t bias;      /* additive bias (== cum, or cum+PROB_SCALE-1 if freq<2) */
    uint16_t cmpl_freq; /* PROB_SCALE - freq */
    uint16_t rcp_shift; /* shift applied after the reciprocal multiply */
} EncSym;

/* Build the reciprocal-multiply constants for one symbol (after ryg_rans's
 * RansEncSymbolInit).  `freq` may be 0 for a symbol that never occurs; such a
 * symbol is never encoded, so its (degenerate) table entry is inert. */
static void enc_sym_init(EncSym *s, uint32_t start, uint32_t freq) {
    s->x_max = ((RANS_L >> PROB_BITS) << 8) * freq;
    s->cmpl_freq = (uint16_t)(PROB_SCALE - freq);
    if (freq < 2) {
        /* freq <= 1: a multiply by ~0u yields q = x - 1, which the bias below
         * corrects.  Avoids a divide-by-zero / divide-by-one special path. */
        s->rcp_freq = ~0u;
        s->rcp_shift = 0;
        s->bias = start + PROB_SCALE - 1;
    } else {
        uint32_t shift = 0;
        while (freq > (1u << shift)) shift++;          /* shift = ceil(log2 freq) */
        s->rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
        s->rcp_shift = (uint16_t)(shift - 1);
        s->bias = start;
    }
}

/* Renorm + coding step for one symbol on one lane.  Emits low bytes of x
 * downward through *pptr until x < x_max, then advances the state. */
static inline void enc_put(uint32_t *state, uint8_t **pptr, const EncSym *sym) {
    uint32_t x = *state;
    uint32_t x_max = sym->x_max;
    if (x >= x_max) {
        uint8_t *ptr = *pptr;
        do {
            *(--ptr) = (uint8_t)(x & 0xffu);
            x >>= 8;
        } while (x >= x_max);
        *pptr = ptr;
    }
    uint32_t q = (uint32_t)(((uint64_t)x * sym->rcp_freq) >> 32);
    *state = x + sym->bias + (q >> sym->rcp_shift) * sym->cmpl_freq;
}

/* Order-0 byte histogram of `in[0..n)` into `hist[256]` (caller zeroes it or not;
 * this function overwrites).  Used to build the rANS model far faster than a
 * Python/NumPy histogram: it runs in C (so ctypes releases the GIL, letting the
 * caller fold several chunk histograms in parallel) and uses FOUR independent
 * accumulator tables to dodge the store-to-load-forwarding stall a single-table
 * histogram hits on repeated bytes (the exponent plane is ~25 distinct values,
 * the pathological case).  Memory-bound: ~one pass over the buffer. */
void z4ai_rans_hist(const uint8_t *in, size_t n, uint32_t *hist) {
    uint32_t h0[256] = {0}, h1[256] = {0}, h2[256] = {0}, h3[256] = {0};
    size_t i = 0;
    size_t aligned = n & ~(size_t)3;
    for (; i < aligned; i += 4) {
        h0[in[i]]++; h1[in[i + 1]]++; h2[in[i + 2]]++; h3[in[i + 3]]++;
    }
    for (; i < n; i++) h0[in[i]]++;
    for (int j = 0; j < 256; j++) hist[j] = h0[j] + h1[j] + h2[j] + h3[j];
}

/* Encode `n` bytes from `in` using the supplied model into `out` (capacity
 * `out_cap`).  Returns the compressed length, or 0 if `out_cap` was too small.
 */
size_t z4ai_rans_encode(const uint8_t *in, size_t n,
                        const uint16_t *freq, const uint16_t *cum,
                        uint8_t *out, size_t out_cap) {
    if (n == 0) return 0;

    /* Derive the reciprocal-multiply table once (256 symbols, negligible). */
    EncSym sym[256];
    for (int s = 0; s < 256; s++) {
        enc_sym_init(&sym[s], cum[s], freq[s]);
    }

    uint8_t *end = out + out_cap;
    uint8_t *ptr = end;
    uint32_t x0 = RANS_L, x1 = RANS_L, x2 = RANS_L, x3 = RANS_L;

    /* Process symbols in reverse.  Lane(i) = i & (N_LANES-1).  The tail (the
     * highest n % N_LANES symbols, indices >= aligned) is coded first, one lane
     * at a time, so the bulk below can run fully unrolled in groups of N_LANES. */
    size_t aligned = n & ~(size_t)(N_LANES - 1);
    uint32_t *lane[N_LANES] = {&x0, &x1, &x2, &x3};
    for (size_t i = n; i-- > aligned;) {
        enc_put(lane[i & (N_LANES - 1)], &ptr, &sym[in[i]]);
        if (ptr <= out) return 0; /* overflow guard (rare; checked per tail sym) */
    }
    /* Unrolled bulk: group g covers indices base..base+N_LANES-1 on lanes
     * 0..N_LANES-1.  Coded in reverse (highest lane first) so the shared stream
     * stays a clean LIFO that the forward decoder consumes in step. */
    for (size_t g = aligned / N_LANES; g-- > 0;) {
        size_t base = g * N_LANES;
        enc_put(&x3, &ptr, &sym[in[base + 3]]);
        enc_put(&x2, &ptr, &sym[in[base + 2]]);
        enc_put(&x1, &ptr, &sym[in[base + 1]]);
        enc_put(&x0, &ptr, &sym[in[base + 0]]);
        if (ptr <= out + N_LANES) return 0; /* guard once per group */
    }

    /* Flush all states at the front, state 0 lowest, each little-endian, so the
     * decoder reads state0, state1, ... in order. */
    if (ptr < out + 4 * N_LANES) return 0;
    ptr -= 4 * N_LANES;
    uint32_t states[N_LANES] = {x0, x1, x2, x3};
    for (int l = 0; l < N_LANES; l++) {
        ptr[l * 4 + 0] = (uint8_t)(states[l] & 0xffu);
        ptr[l * 4 + 1] = (uint8_t)((states[l] >> 8) & 0xffu);
        ptr[l * 4 + 2] = (uint8_t)((states[l] >> 16) & 0xffu);
        ptr[l * 4 + 3] = (uint8_t)((states[l] >> 24) & 0xffu);
    }

    size_t len = (size_t)(end - ptr);
    /* Move the compressed block to the start of `out` for a simple ctypes API. */
    memmove(out, ptr, len);
    return len;
}

/* Normalise a 256-bin histogram to frequencies summing to PROB_SCALE, every
 * occurring symbol getting >= 1.  Mirrors the Python ``_normalize`` (floor of the
 * scaled count, then absorb the rounding remainder into the largest frequencies).
 * The exact tie-break is immaterial to correctness because the per-chunk adaptive
 * frame STORES this table for the decoder — it only needs to be a valid, near-
 * optimal model.  Used by :func:`z4ai_rans_encode_local`. */
static void rans_normalize_c(const uint32_t *hist, uint16_t *freq) {
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) total += hist[i];
    for (int i = 0; i < 256; i++) freq[i] = 0;
    if (total == 0) return;

    int64_t f[256];
    int64_t fsum = 0;
    for (int i = 0; i < 256; i++) {
        int64_t v = (int64_t)(((unsigned long long)hist[i] * PROB_SCALE) / total);
        if (hist[i] > 0 && v == 0) v = 1;  /* an occurring symbol needs freq >= 1 */
        f[i] = v;
        fsum += v;
    }
    int64_t diff = (int64_t)PROB_SCALE - fsum;
    if (diff > 0) {
        int am = 0;
        for (int i = 1; i < 256; i++) if (f[i] > f[am]) am = i;
        f[am] += diff;                      /* give the surplus to the peak */
    } else {
        int64_t need = -diff;               /* reclaim the overflow from the peaks */
        while (need > 0) {
            int am = -1; int64_t best = 1;
            for (int i = 0; i < 256; i++) if (f[i] > best) { best = f[i]; am = i; }
            if (am < 0) break;              /* unreachable: total > 0 => some f > 1 */
            int64_t take = f[am] - 1;
            if (take > need) take = need;
            f[am] -= take;
            need -= take;
        }
    }
    for (int i = 0; i < 256; i++) freq[i] = (uint16_t)f[i];
}

/* One-call LOCAL-model chunk encode: histogram -> normalise -> cum -> rANS encode,
 * all in C with the GIL released (ctypes), so the Python per-chunk adaptive path
 * (compress_adaptive) parallelises across the thread pool WITHOUT paying ~6 Python/
 * NumPy ops and ctypes casts per 64 KiB chunk (the dominant cost once the model
 * build moved to C).  Writes the chunk's 256-entry (512-byte) freq table to
 * ``freq_out`` and the payload to ``out``; returns the payload length, or 0 on
 * overflow / empty input.  The frame format is unchanged (same per-chunk freq +
 * payload), so decode is unaffected. */
size_t z4ai_rans_encode_local(const uint8_t *in, size_t n,
                              uint8_t *out, size_t out_cap, uint16_t *freq_out) {
    if (n == 0) return 0;
    uint32_t hist[256];
    z4ai_rans_hist(in, n, hist);
    rans_normalize_c(hist, freq_out);
    uint16_t cum[256];
    cum[0] = 0;
    for (int i = 1; i < 256; i++) cum[i] = (uint16_t)(cum[i - 1] + freq_out[i - 1]);
    return z4ai_rans_encode(in, n, freq_out, cum, out, out_cap);
}

/* Decode + renorm step for one symbol on one lane. */
static inline uint8_t dec_get(uint32_t *state, const uint8_t **pptr,
                              const uint8_t *in_end, const uint16_t *freq,
                              const uint16_t *cum, const uint8_t *slot2sym) {
    uint32_t x = *state;
    uint32_t slot = x & PROB_MASK;
    uint8_t s = slot2sym[slot];
    x = (uint32_t)freq[s] * (x >> PROB_BITS) + slot - cum[s];
    if (x < RANS_L) {
        const uint8_t *ptr = *pptr;
        do {
            uint32_t next = (ptr < in_end) ? *ptr : 0u;
            ptr++;
            x = (x << 8) | next;
        } while (x < RANS_L);
        *pptr = ptr;
    }
    *state = x;
    return s;
}

/* Decode `n` symbols into `out` from the `in_len`-byte stream `in`. */
void z4ai_rans_decode(const uint8_t *in, size_t in_len,
                      const uint16_t *freq, const uint16_t *cum,
                      const uint8_t *slot2sym, uint8_t *out, size_t n) {
    if (n == 0) return;
    const uint8_t *ptr = in;
    const uint8_t *in_end = in + in_len;

    /* Seed all states (state 0 first), mirroring the encoder's flush. */
#define SEED_LANE(k) ((uint32_t)ptr[(k) * 4] | ((uint32_t)ptr[(k) * 4 + 1] << 8) | \
                      ((uint32_t)ptr[(k) * 4 + 2] << 16) | ((uint32_t)ptr[(k) * 4 + 3] << 24))
    uint32_t x0 = SEED_LANE(0), x1 = SEED_LANE(1), x2 = SEED_LANE(2), x3 = SEED_LANE(3);
#undef SEED_LANE
    ptr += 4 * N_LANES;

    /* Unrolled bulk: forward over groups of N_LANES (lane 0 first), the exact
     * reverse of the encoder's order, so each lane consumes the bytes its
     * counterpart produced. */
    size_t aligned = n & ~(size_t)(N_LANES - 1);
    for (size_t base = 0; base < aligned; base += N_LANES) {
        out[base + 0] = dec_get(&x0, &ptr, in_end, freq, cum, slot2sym);
        out[base + 1] = dec_get(&x1, &ptr, in_end, freq, cum, slot2sym);
        out[base + 2] = dec_get(&x2, &ptr, in_end, freq, cum, slot2sym);
        out[base + 3] = dec_get(&x3, &ptr, in_end, freq, cum, slot2sym);
    }
    /* Tail: highest n % N_LANES symbols, lane(i) = i & (N_LANES-1). */
    uint32_t *lane[N_LANES] = {&x0, &x1, &x2, &x3};
    for (size_t i = aligned; i < n; i++) {
        out[i] = dec_get(lane[i & (N_LANES - 1)], &ptr, in_end, freq, cum, slot2sym);
    }
}

/* Block-wise convenience decoder: derive `cum` and `slot2sym` from `freq`
 * INTERNALLY and decode into the caller's `out`.
 *
 * Why this exists: the per-block Python wrapper rebuilt the 16384-entry
 * `slot2sym` table (np.repeat) and copied the output (tobytes) for EVERY block,
 * which dominated block-wise decode time — small blocks (best ratio, via local
 * exponent adaptation) measured ~375 MB/s vs the ~2.4 GB/s raw kernel.  Building
 * the tables here (a cheap 16384-entry fill in C) lets a block-wise caller pass
 * only the 512-byte `freq` per block and decode straight into a slice of one
 * preallocated buffer, so small-block decode runs at near kernel speed.  Output
 * is byte-identical to z4ai_rans_decode. */
void z4ai_rans_decode_f(const uint8_t *in, size_t in_len, const uint16_t *freq,
                        uint8_t *out, size_t n) {
    if (n == 0) return;
    uint16_t cum[256];
    uint8_t slot2sym[PROB_SCALE];  /* 16 KiB scratch — derived per block */
    uint32_t c = 0;
    for (int s = 0; s < 256; s++) {
        cum[s] = (uint16_t)c;
        uint32_t f = freq[s];
        for (uint32_t j = 0; j < f; j++) slot2sym[c + j] = (uint8_t)s;
        c += f;
    }
    z4ai_rans_decode(in, in_len, freq, cum, slot2sym, out, n);
}

/* ======================================================================== *
 * Order-1 (context = previous byte) rANS.
 *
 * Why a separate path: the order-0 coder above reaches the order-0 entropy
 * floor, but real model exponent planes carry CONDITIONAL structure (measured:
 * distilgpt2 bf16 exponent H0=3.10 bits but H1=2.36 bits -> a 6.4% smaller
 * stream than any order-0 / Zstd-FSE coder, and below Zstd's whole-plane LDM
 * pass too).  An order-1 model is "an array of order-0 tables" (one per previous
 * byte), as in Giesen's rANS notes and jkbonfield's rANS_static.
 *
 * Interleaving for order-1 cannot use the order-0 stride trick (symbol i's
 * context is symbol i-1, which would sit on a different lane and serialise the
 * lanes).  Instead the buffer is split into NSEG CONTIGUOUS segments, each coded
 * by its own single rANS state with its own running context chain; the segments
 * are mutually independent, so their decode steps pipeline (the ILP win) while
 * each keeps an exact order-1 context.  Each segment's first byte uses context 0.
 *
 * Model: `freq` is a 256*256 row-major table (row = context = previous byte),
 * each non-empty row summing to O1_SCALE.  Probability resolution is 12-bit
 * (O1_SCALE=4096): ample for the <=~40-symbol exponent alphabet and keeps each
 * per-context slot table at 4 KiB so the handful of live contexts stay in cache.
 *
 * Payload layout written by the encoder:  [NSEG * u32 sublen][sub0]..[subN-1],
 * each sub = [state:4 LE][renorm bytes].  Self-delimiting given NSEG (the caller
 * stores n, NSEG and the freq table in the Python-level header).
 * ======================================================================== */
#define O1_PROB_BITS 12u
#define O1_SCALE (1u << O1_PROB_BITS)     /* 4096 */
#define O1_MASK  (O1_SCALE - 1u)
/* RANS_L (1<<23) is a multiple of O1_SCALE and x_max peaks at
 * (RANS_L>>O1_PROB_BITS)<<8 * freq_max = 2^19 * 2^12 = 2^31 < 2^32 - safe. */

static void o1_enc_sym_init(EncSym *s, uint32_t start, uint32_t freq) {
    s->x_max = ((RANS_L >> O1_PROB_BITS) << 8) * freq;
    s->cmpl_freq = (uint16_t)(O1_SCALE - freq);
    if (freq < 2) {
        s->rcp_freq = ~0u;
        s->rcp_shift = 0;
        s->bias = start + O1_SCALE - 1;
    } else {
        uint32_t shift = 0;
        while (freq > (1u << shift)) shift++;
        s->rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
        s->rcp_shift = (uint16_t)(shift - 1);
        s->bias = start;
    }
}

/* Segment boundaries shared by encoder and decoder: b[l] = l*n/nseg. */
static inline size_t o1_bound(size_t n, int nseg, int l) {
    return (size_t)((unsigned long long)n * (unsigned)l / (unsigned)nseg);
}

/* Build the order-1 JOINT histogram hist[ctx*256 + cur] of `in[0..n)`, where the
 * context of byte i is in[i-1] EXCEPT at the `n_resets` positions in `resets`
 * (sorted, unique, must include 0), where the context is forced to 0 — exactly
 * mirroring the segment/chunk context resets the encoder applies, so the model
 * the decoder is handed matches what the encoder used.
 *
 * Replaces a NumPy `bincount(ctx*256+cur, minlength=65536)` that first widened
 * the whole plane to int64 (≈3x the plane in scratch) and then scattered 88M
 * elements into 65536 bins — the dominant cost (~85%) of order-1 compress.  Here
 * the scatter runs in C (GIL released) with FOUR independent accumulator tables
 * to dodge the store-to-load-forwarding stall a single table hits on the heavily
 * repeated (ctx,cur) pairs of a ~40-symbol exponent plane.  `hist` must hold
 * 256*256 uint32 (caller need not pre-zero it). */
void z4ai_rans_o1_hist(const uint8_t *in, size_t n,
                       const uint32_t *resets, size_t n_resets,
                       uint32_t *hist) {
    const size_t TS = 256u * 256u;
    memset(hist, 0, TS * sizeof(uint32_t));
    if (n == 0) return;

    uint32_t *h = (uint32_t *)calloc(4 * TS, sizeof(uint32_t));
    if (h == NULL) {
        /* Low-memory fallback: single-table sequential scatter. */
        hist[0u * 256u + in[0]]++;
        for (size_t i = 1; i < n; i++) hist[(size_t)in[i - 1] * 256u + in[i]]++;
    } else {
        uint32_t *h0 = h, *h1 = h + TS, *h2 = h + 2 * TS, *h3 = h + 3 * TS;
        /* Position 0 always has context 0 (it is always a reset). */
        h0[(size_t)0u * 256u + in[0]]++;
        /* Bulk assumes ctx[i] = in[i-1]; reset positions are corrected below. */
        size_t i = 1;
        for (; i + 3 < n; i += 4) {
            h0[(size_t)in[i - 1] * 256u + in[i]]++;
            h1[(size_t)in[i]     * 256u + in[i + 1]]++;
            h2[(size_t)in[i + 1] * 256u + in[i + 2]]++;
            h3[(size_t)in[i + 2] * 256u + in[i + 3]]++;
        }
        for (; i < n; i++) h0[(size_t)in[i - 1] * 256u + in[i]]++;
        for (size_t j = 0; j < TS; j++) hist[j] = h0[j] + h1[j] + h2[j] + h3[j];
        free(h);
    }

    /* Reset fix-up: the bulk counted ctx=in[r-1] at each reset r>0; move that
     * count to ctx=0.  resets is tiny (<= chunks*nseg), so this is negligible. */
    for (size_t k = 0; k < n_resets; k++) {
        size_t r = (size_t)resets[k];
        if (r == 0 || r >= n) continue;  /* r==0 already has context 0 */
        uint8_t cur = in[r];
        hist[(size_t)in[r - 1] * 256u + cur]--;  /* remove the wrong (ctx=in[r-1]) */
        hist[(size_t)0u * 256u + cur]++;          /* add the correct  (ctx=0)      */
    }
}

/* Encode `n` bytes from `in` (order-1) into `out`; returns length or 0 on
 * overflow.  `freq` is 256*256 row-major (row = context). */
size_t z4ai_rans_o1_encode(const uint8_t *in, size_t n, int nseg,
                           const uint16_t *freq, uint8_t *out, size_t out_cap) {
    if (n == 0 || nseg <= 0) return 0;

    /* Per-(context,symbol) encoder constants, built only for contexts that
     * occur (row sum > 0).  ~1 MiB worst case; calloc so untouched rows inert. */
    EncSym *sym = (EncSym *)calloc((size_t)256 * 256, sizeof(EncSym));
    if (!sym) return 0;
    for (int c = 0; c < 256; c++) {
        const uint16_t *fr = freq + (size_t)c * 256;
        uint32_t acc = 0, rowsum = 0;
        for (int s = 0; s < 256; s++) rowsum += fr[s];
        if (rowsum == 0) continue;
        EncSym *srow = sym + (size_t)c * 256;
        for (int s = 0; s < 256; s++) {
            o1_enc_sym_init(&srow[s], acc, fr[s]);
            acc += fr[s];
        }
    }

    size_t hdr = (size_t)nseg * 4;                 /* sublen index */
    if (out_cap < hdr) { free(sym); return 0; }
    uint8_t *cursor = out + hdr;                   /* substream area */
    uint8_t *cap_end = out + out_cap;

    for (int l = 0; l < nseg; l++) {
        size_t lo = o1_bound(n, nseg, l);
        size_t hi = o1_bound(n, nseg, l + 1);
        uint8_t *ptr = cap_end;                    /* fill downward */
        uint32_t x = RANS_L;
        for (size_t i = hi; i-- > lo;) {
            uint32_t ctx = (i == lo) ? 0u : in[i - 1];
            enc_put(&x, &ptr, &sym[ctx * 256 + in[i]]);
            if (ptr <= cursor + 4) { free(sym); return 0; }
        }
        ptr -= 4;                                  /* flush state */
        if (ptr < cursor) { free(sym); return 0; }
        ptr[0] = (uint8_t)(x & 0xffu);
        ptr[1] = (uint8_t)((x >> 8) & 0xffu);
        ptr[2] = (uint8_t)((x >> 16) & 0xffu);
        ptr[3] = (uint8_t)((x >> 24) & 0xffu);
        size_t sublen = (size_t)(cap_end - ptr);
        memmove(cursor, ptr, sublen);              /* pack tight after index */
        out[(size_t)l * 4 + 0] = (uint8_t)(sublen & 0xffu);
        out[(size_t)l * 4 + 1] = (uint8_t)((sublen >> 8) & 0xffu);
        out[(size_t)l * 4 + 2] = (uint8_t)((sublen >> 16) & 0xffu);
        out[(size_t)l * 4 + 3] = (uint8_t)((sublen >> 24) & 0xffu);
        cursor += sublen;
    }
    free(sym);
    return (size_t)(cursor - out);
}

/* Build cum[256] and slot2sym[O1_SCALE] for one context row. */
static void o1_build_row(const uint16_t *fr, uint16_t *cum, uint8_t *slot2sym) {
    uint32_t acc = 0;
    for (int s = 0; s < 256; s++) {
        cum[s] = (uint16_t)acc;
        for (uint32_t k = 0; k < fr[s]; k++) slot2sym[acc + k] = (uint8_t)s;
        acc += fr[s];
    }
}

static inline uint8_t o1_dec_get(uint32_t *state, const uint8_t **pptr,
                                 const uint8_t *in_end, const uint16_t *fr,
                                 const uint16_t *cum, const uint8_t *slot2sym) {
    uint32_t x = *state;
    uint32_t slot = x & O1_MASK;
    uint8_t s = slot2sym[slot];
    x = (uint32_t)fr[s] * (x >> O1_PROB_BITS) + slot - cum[s];
    if (x < RANS_L) {
        const uint8_t *ptr = *pptr;
        do {
            uint32_t next = (ptr < in_end) ? *ptr : 0u;
            ptr++;
            x = (x << 8) | next;
        } while (x < RANS_L);
        *pptr = ptr;
    }
    *state = x;
    return s;
}

/* Decode `n` order-1 symbols into `out`.  `freq` is 256*256 row-major. */
void z4ai_rans_o1_decode(const uint8_t *in, size_t in_len, size_t n, int nseg,
                         const uint16_t *freq, uint8_t *out) {
    if (n == 0 || nseg <= 0) return;

    uint16_t *cum = (uint16_t *)calloc((size_t)256 * 256, sizeof(uint16_t));
    uint8_t *s2s = (uint8_t *)calloc((size_t)256 * O1_SCALE, 1);
    if (!cum || !s2s) { free(cum); free(s2s); return; }
    for (int c = 0; c < 256; c++) {
        const uint16_t *fr = freq + (size_t)c * 256;
        uint32_t rowsum = 0;
        for (int s = 0; s < 256; s++) rowsum += fr[s];
        if (rowsum == 0) continue;
        o1_build_row(fr, cum + (size_t)c * 256, s2s + (size_t)c * O1_SCALE);
    }

    int ns = nseg > 64 ? 64 : nseg;
    const uint8_t *sub[64];
    const uint8_t *sub_end[64];
    uint32_t st[64];
    size_t pos[64], end[64];
    const uint8_t *p = in + (size_t)ns * 4;        /* substreams follow index */
    const uint8_t *in_end = in + in_len;
    for (int l = 0; l < ns; l++) {
        uint32_t sublen = (uint32_t)in[(size_t)l * 4 + 0]
                        | ((uint32_t)in[(size_t)l * 4 + 1] << 8)
                        | ((uint32_t)in[(size_t)l * 4 + 2] << 16)
                        | ((uint32_t)in[(size_t)l * 4 + 3] << 24);
        st[l] = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
              | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        sub[l] = p + 4;
        sub_end[l] = p + sublen;
        if (sub_end[l] > in_end) sub_end[l] = in_end;
        pos[l] = o1_bound(n, nseg, l);
        end[l] = o1_bound(n, nseg, l + 1);
        p += sublen;
    }

    /* Lockstep over segments for ILP: one symbol per live lane per round. */
    int any = 1;
    while (any) {
        any = 0;
        for (int l = 0; l < ns; l++) {
            if (pos[l] >= end[l]) continue;
            size_t i = pos[l];
            uint32_t ctx = (i == o1_bound(n, nseg, l)) ? 0u : out[i - 1];
            out[i] = o1_dec_get(&st[l], &sub[l], sub_end[l],
                                freq + (size_t)ctx * 256,
                                cum + (size_t)ctx * 256,
                                s2s + (size_t)ctx * O1_SCALE);
            pos[l]++;
            any = 1;
        }
    }
    free(cum); free(s2s);
}
