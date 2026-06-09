// SPDX-License-Identifier: Apache-2.0
//
// Fused, multithreaded native codec for z4ai's chunked throughput path.
//
// This is a drop-in C implementation of the pure-Python pipeline in
// ``z4ai/chunked.py`` and produces / consumes the EXACT SAME ``Z4AIMF01`` frame
// format, so the two are byte-interoperable and ``chunked.py`` stays the
// reference + fallback.  The whole point is speed: the Python path is capped by
// (a) per-chunk interpreter overhead and (b) the GIL serializing the Python
// glue between GIL-releasing zstd calls.  Here the entire chunk loop — byte
// de-interleave, libzstd, re-interleave — runs in C with the GIL released and
// across a pool of pthreads, so it scales with cores the way zipnn's C does.
//
// Frame layout (little-endian), identical to chunked.py:
//   magic    8 bytes  "Z4AIMF01"
//   flags    u8 (0)
//   width    u8
//   n_chunks u32
//   per chunk: u32 n_elem, u8 tail_len, tail bytes, then
//              width * (u8 method, u32 comp_len, comp_bytes)
//   method 0 = stored raw, 1 = zstd.  Plane j = byte j of every element.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <zstd.h>

static const char MAGIC[8] = {'Z', '4', 'A', 'I', 'M', 'F', '0', '1'};
#define METHOD_STORE 0
#define METHOD_ZSTD 1
#define HEADER_LEN 14  // magic(8) + flags(1) + width(1) + n_chunks(4)

// ---------------------------------------------------------------------------
// Little-endian helpers (frames are LE on every supported platform)
// ---------------------------------------------------------------------------
static inline void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static inline uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

// ---------------------------------------------------------------------------
// Byte-plane transpose.  -O3 -march=native vectorizes these tight loops.
// split: plane[j][i] = src[i*width + j]   join: dst[i*width + j] = plane[j][i]
// ---------------------------------------------------------------------------
// De-interleave src into width plane-major buffers.  Reads src contiguously
// (cache-friendly) and writes each plane contiguously.  Width 2/4/8 are
// specialized so the compiler auto-vectorizes the inner loop.
static void split_planes(const uint8_t *src, size_t n_elem, int width,
                         uint8_t *planes /* width * n_elem, plane-major */) {
    if (width == 2) {
        uint8_t *p0 = planes, *p1 = planes + n_elem;
        for (size_t i = 0; i < n_elem; i++) {
            p0[i] = src[2 * i];
            p1[i] = src[2 * i + 1];
        }
    } else if (width == 4) {
        uint8_t *p0 = planes, *p1 = planes + n_elem;
        uint8_t *p2 = planes + 2 * n_elem, *p3 = planes + 3 * n_elem;
        for (size_t i = 0; i < n_elem; i++) {
            p0[i] = src[4 * i];     p1[i] = src[4 * i + 1];
            p2[i] = src[4 * i + 2]; p3[i] = src[4 * i + 3];
        }
    } else if (width == 8) {
        uint8_t *p[8];
        for (int j = 0; j < 8; j++) p[j] = planes + (size_t)j * n_elem;
        for (size_t i = 0; i < n_elem; i++) {
            const uint8_t *s = src + 8 * i;
            p[0][i] = s[0]; p[1][i] = s[1]; p[2][i] = s[2]; p[3][i] = s[3];
            p[4][i] = s[4]; p[5][i] = s[5]; p[6][i] = s[6]; p[7][i] = s[7];
        }
    } else {
        for (int j = 0; j < width; j++) {
            uint8_t *dst = planes + (size_t)j * n_elem;
            const uint8_t *s = src + j;
            for (size_t i = 0; i < n_elem; i++) dst[i] = s[(size_t)i * width];
        }
    }
}

// Re-interleave width plane-major buffers into dst.  Writes dst contiguously
// (the cache-critical side) and reads from the planes.  Width 2/4/8 specialized.
static void join_planes(const uint8_t *const *planes, size_t n_elem, int width,
                        uint8_t *dst) {
    if (width == 2) {
        const uint8_t *p0 = planes[0], *p1 = planes[1];
        for (size_t i = 0; i < n_elem; i++) {
            dst[2 * i] = p0[i];
            dst[2 * i + 1] = p1[i];
        }
    } else if (width == 4) {
        const uint8_t *p0 = planes[0], *p1 = planes[1], *p2 = planes[2], *p3 = planes[3];
        for (size_t i = 0; i < n_elem; i++) {
            dst[4 * i] = p0[i];     dst[4 * i + 1] = p1[i];
            dst[4 * i + 2] = p2[i]; dst[4 * i + 3] = p3[i];
        }
    } else if (width == 8) {
        const uint8_t *p[8];
        for (int j = 0; j < 8; j++) p[j] = planes[j];
        for (size_t i = 0; i < n_elem; i++) {
            uint8_t *d = dst + 8 * i;
            d[0] = p[0][i]; d[1] = p[1][i]; d[2] = p[2][i]; d[3] = p[3][i];
            d[4] = p[4][i]; d[5] = p[5][i]; d[6] = p[6][i]; d[7] = p[7][i];
        }
    } else {
        for (int j = 0; j < width; j++) {
            const uint8_t *p = planes[j];
            uint8_t *d = dst + j;
            for (size_t i = 0; i < n_elem; i++) d[(size_t)i * width] = p[i];
        }
    }
}

// ===========================================================================
// COMPRESS
// ===========================================================================
typedef struct {
    const uint8_t *buf;   // whole input
    size_t total;
    int width;
    int level;
    size_t step;          // bytes per chunk (multiple of width)
    size_t n_chunks;
    // outputs (one per chunk):
    uint8_t **chunk_out;  // malloc'd frame bytes for each chunk
    size_t *chunk_len;
    // work distribution
    size_t start_chunk;
    size_t end_chunk;
    int error;            // set nonzero on failure
} comp_task;

// Build the per-chunk frame bytes for chunks [start_chunk, end_chunk).
static void *comp_worker(void *arg) {
    comp_task *t = (comp_task *)arg;
    const int width = t->width;
    // Scratch reused across this thread's chunks.
    uint8_t *planes = NULL;
    size_t planes_cap = 0;
    uint8_t *cbuf = NULL;
    size_t cbuf_cap = 0;

    for (size_t c = t->start_chunk; c < t->end_chunk; c++) {
        size_t off = c * t->step;
        size_t len = t->step;
        if (off + len > t->total) len = t->total - off;
        const uint8_t *src = t->buf + off;
        size_t n_elem = len / (size_t)width;
        size_t aligned = n_elem * (size_t)width;
        size_t tail_len = len - aligned;

        // Plane scratch.
        size_t need = aligned;  // width * n_elem
        if (need > planes_cap) {
            uint8_t *np = (uint8_t *)realloc(planes, need ? need : 1);
            if (!np) { t->error = 1; goto done; }
            planes = np; planes_cap = need;
        }
        if (width == 1) {
            if (n_elem) memcpy(planes, src, n_elem);
        } else if (n_elem) {
            split_planes(src, n_elem, width, planes);
        }

        // Compress each plane; keep the smaller of zstd vs raw store.
        // First pass: figure out total size to allocate the chunk buffer.
        // We compress into a temp per plane; store method/len/payload.
        // Upper bound for the chunk frame: header + tail + sum(5 + bound).
        size_t bound_each = ZSTD_compressBound(n_elem ? n_elem : 1);
        size_t cap = 5 + tail_len + (size_t)width * (5 + bound_each);
        if (cap > cbuf_cap) {
            uint8_t *nb = (uint8_t *)realloc(cbuf, cap);
            if (!nb) { t->error = 1; goto done; }
            cbuf = nb; cbuf_cap = cap;
        }
        uint8_t *w = cbuf;
        put_u32(w, (uint32_t)n_elem); w += 4;
        *w++ = (uint8_t)tail_len;
        if (tail_len) { memcpy(w, src + aligned, tail_len); w += tail_len; }

        for (int j = 0; j < width; j++) {
            const uint8_t *plane = planes + (size_t)j * n_elem;
            uint8_t *method_p = w; w += 1;
            uint8_t *clen_p = w; w += 4;
            size_t csz = 0;
            if (n_elem) {
                csz = ZSTD_compress(w, bound_each, plane, n_elem, t->level);
                if (ZSTD_isError(csz)) { t->error = 1; goto done; }
            }
            if (n_elem && csz < n_elem) {
                *method_p = METHOD_ZSTD;
                put_u32(clen_p, (uint32_t)csz);
                w += csz;
            } else {
                // store raw (incompressible or empty)
                *method_p = METHOD_STORE;
                put_u32(clen_p, (uint32_t)n_elem);
                if (n_elem) memcpy(w, plane, n_elem);
                w += n_elem;
            }
        }
        size_t clen = (size_t)(w - cbuf);
        uint8_t *outc = (uint8_t *)malloc(clen ? clen : 1);
        if (!outc) { t->error = 1; goto done; }
        memcpy(outc, cbuf, clen);
        t->chunk_out[c] = outc;
        t->chunk_len[c] = clen;
    }
done:
    free(planes);
    free(cbuf);
    return NULL;
}

static PyObject *py_compress(PyObject *self, PyObject *args) {
    Py_buffer view;
    int width, level, threads;
    Py_ssize_t chunk_size;
    if (!PyArg_ParseTuple(args, "y*iiin", &view, &width, &level, &threads,
                          &chunk_size))
        return NULL;
    if (width < 1) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "width must be >= 1");
        return NULL;
    }
    const uint8_t *buf = (const uint8_t *)view.buf;
    size_t total = (size_t)view.len;

    // Chunk granularity: a whole number of elements, mirroring chunked.py.
    size_t step = (size_t)((chunk_size / width) * width);
    if (step < (size_t)width) step = (size_t)width;
    size_t n_chunks = total ? (total + step - 1) / step : 0;

    if (threads < 1) threads = 1;
    if ((size_t)threads > n_chunks) threads = n_chunks ? (int)n_chunks : 1;

    uint8_t **chunk_out = (uint8_t **)calloc(n_chunks ? n_chunks : 1, sizeof(uint8_t *));
    size_t *chunk_len = (size_t *)calloc(n_chunks ? n_chunks : 1, sizeof(size_t));
    if (!chunk_out || !chunk_len) {
        free(chunk_out); free(chunk_len); PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    comp_task *tasks = (comp_task *)calloc(threads ? threads : 1, sizeof(comp_task));
    pthread_t *tids = (pthread_t *)calloc(threads ? threads : 1, sizeof(pthread_t));
    char *created = (char *)calloc(threads ? threads : 1, 1);
    if (!tasks || !tids || !created) {
        free(tasks); free(tids); free(created);
        free(chunk_out); free(chunk_len);
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    int err = 0;
    Py_BEGIN_ALLOW_THREADS
    size_t per = n_chunks ? (n_chunks + threads - 1) / threads : 0;
    for (int ti = 0; ti < threads; ti++) {
        size_t s = (size_t)ti * per;
        if (s >= n_chunks) break;
        size_t e = s + per; if (e > n_chunks) e = n_chunks;
        tasks[ti].buf = buf; tasks[ti].total = total; tasks[ti].width = width;
        tasks[ti].level = level; tasks[ti].step = step; tasks[ti].n_chunks = n_chunks;
        tasks[ti].chunk_out = chunk_out; tasks[ti].chunk_len = chunk_len;
        tasks[ti].start_chunk = s; tasks[ti].end_chunk = e; tasks[ti].error = 0;
        if (pthread_create(&tids[ti], NULL, comp_worker, &tasks[ti]) == 0) {
            created[ti] = 1;
        } else {
            comp_worker(&tasks[ti]);  // run inline if a thread cannot be spawned
        }
    }
    for (int ti = 0; ti < threads; ti++) {
        if (created[ti]) pthread_join(tids[ti], NULL);
    }
    Py_END_ALLOW_THREADS

    for (int ti = 0; ti < threads; ti++) {
        if (tasks[ti].error) err = 1;
    }

    if (err) {
        for (size_t c = 0; c < n_chunks; c++) free(chunk_out[c]);
        free(chunk_out); free(chunk_len); free(tasks); free(tids); free(created);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_RuntimeError, "native compress failed");
        return NULL;
    }

    // Assemble final frame: header + concatenated chunk frames.
    size_t body = 0;
    for (size_t c = 0; c < n_chunks; c++) body += chunk_len[c];
    size_t frame_len = HEADER_LEN + body;
    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)frame_len);
    if (!out) {
        for (size_t c = 0; c < n_chunks; c++) free(chunk_out[c]);
        free(chunk_out); free(chunk_len); free(tasks); free(tids); free(created);
        PyBuffer_Release(&view);
        return NULL;
    }
    uint8_t *o = (uint8_t *)PyBytes_AS_STRING(out);
    memcpy(o, MAGIC, 8);
    o[8] = 0; o[9] = (uint8_t)width;
    put_u32(o + 10, (uint32_t)n_chunks);
    uint8_t *p = o + HEADER_LEN;
    for (size_t c = 0; c < n_chunks; c++) {
        memcpy(p, chunk_out[c], chunk_len[c]);
        p += chunk_len[c];
        free(chunk_out[c]);
    }
    free(chunk_out); free(chunk_len); free(tasks); free(tids); free(created);
    PyBuffer_Release(&view);
    return out;
}

// ===========================================================================
// DECOMPRESS
// ===========================================================================
typedef struct {
    const uint8_t *frame;
    int width;
    // per chunk:
    size_t *chunk_hdr_off;   // offset of each chunk's header in the frame
    size_t *out_off;         // output offset of each chunk
    uint8_t *out;            // preallocated output
    size_t start_chunk, end_chunk;
    int error;
} decomp_task;

static void *decomp_worker(void *arg) {
    decomp_task *t = (decomp_task *)arg;
    const int width = t->width;
    const uint8_t *frame = t->frame;
    uint8_t *plane_buf = NULL;     // width * n_elem scratch for decoded planes
    size_t plane_cap = 0;
    const uint8_t **plane_ptrs = (const uint8_t **)malloc(sizeof(uint8_t *) * (width > 0 ? width : 1));
    if (!plane_ptrs) { t->error = 1; return NULL; }

    for (size_t c = t->start_chunk; c < t->end_chunk; c++) {
        const uint8_t *h = frame + t->chunk_hdr_off[c];
        uint32_t n_elem = get_u32(h); h += 4;
        uint8_t tail_len = *h++;
        const uint8_t *tail = h; h += tail_len;
        uint8_t *dst = t->out + t->out_off[c];

        if (width == 1) {
            // single plane
            uint8_t method = *h++;
            uint32_t clen = get_u32(h); h += 4;
            if (method == METHOD_ZSTD) {
                size_t got = ZSTD_decompress(dst, n_elem, h, clen);
                if (ZSTD_isError(got) || got != n_elem) { t->error = 1; goto done; }
            } else {
                if (clen != n_elem) { t->error = 1; goto done; }
                memcpy(dst, h, n_elem);
            }
            if (tail_len) memcpy(dst + n_elem, tail, tail_len);
            continue;
        }

        size_t need = (size_t)n_elem * (size_t)width;
        if (need > plane_cap) {
            uint8_t *nb = (uint8_t *)realloc(plane_buf, need ? need : 1);
            if (!nb) { t->error = 1; goto done; }
            plane_buf = nb; plane_cap = need;
        }
        for (int j = 0; j < width; j++) {
            uint8_t method = *h++;
            uint32_t clen = get_u32(h); h += 4;
            uint8_t *pj = plane_buf + (size_t)j * n_elem;
            if (method == METHOD_ZSTD) {
                size_t got = ZSTD_decompress(pj, n_elem, h, clen);
                if (ZSTD_isError(got) || got != n_elem) { t->error = 1; goto done; }
            } else {
                if (clen != n_elem) { t->error = 1; goto done; }
                if (n_elem) memcpy(pj, h, n_elem);
            }
            h += clen;
            plane_ptrs[j] = pj;
        }
        if (n_elem) join_planes(plane_ptrs, n_elem, width, dst);
        if (tail_len) memcpy(dst + need, tail, tail_len);
    }
done:
    free(plane_buf);
    free(plane_ptrs);
    return NULL;
}

static PyObject *py_decompress(PyObject *self, PyObject *args) {
    Py_buffer view;
    int threads;
    if (!PyArg_ParseTuple(args, "y*i", &view, &threads)) return NULL;
    const uint8_t *frame = (const uint8_t *)view.buf;
    size_t flen = (size_t)view.len;
    if (flen < HEADER_LEN || memcmp(frame, MAGIC, 8) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "not a Z4AIMF01 frame");
        return NULL;
    }
    int width = frame[9];
    uint32_t n_chunks = get_u32(frame + 10);
    if (width < 1) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "bad width");
        return NULL;
    }

    size_t *hdr_off = (size_t *)malloc(sizeof(size_t) * (n_chunks ? n_chunks : 1));
    size_t *out_off = (size_t *)malloc(sizeof(size_t) * (n_chunks ? n_chunks : 1));
    if (!hdr_off || !out_off) {
        free(hdr_off); free(out_off); PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    // Serial scan: locate each chunk and compute output offsets + total size.
    size_t off = HEADER_LEN;
    size_t total_out = 0;
    int parse_err = 0;
    for (uint32_t c = 0; c < n_chunks; c++) {
        if (off + 5 > flen) { parse_err = 1; break; }
        hdr_off[c] = off;
        uint32_t n_elem = get_u32(frame + off); off += 4;
        uint8_t tail_len = frame[off]; off += 1;
        off += tail_len;
        out_off[c] = total_out;
        total_out += (size_t)n_elem * (size_t)width + tail_len;
        for (int j = 0; j < width; j++) {
            if (off + 5 > flen) { parse_err = 1; break; }
            off += 1;
            uint32_t clen = get_u32(frame + off); off += 4;
            off += clen;
            if (off > flen) { parse_err = 1; break; }
        }
        if (parse_err) break;
    }
    if (parse_err) {
        free(hdr_off); free(out_off); PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "corrupt frame");
        return NULL;
    }

    PyObject *out = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_out);
    if (!out) {
        free(hdr_off); free(out_off); PyBuffer_Release(&view);
        return NULL;
    }
    uint8_t *obuf = (uint8_t *)PyBytes_AS_STRING(out);

    if (threads < 1) threads = 1;
    if ((size_t)threads > n_chunks) threads = n_chunks ? (int)n_chunks : 1;

    decomp_task *tasks = (decomp_task *)calloc(threads ? threads : 1, sizeof(decomp_task));
    pthread_t *tids = (pthread_t *)calloc(threads ? threads : 1, sizeof(pthread_t));
    char *created = (char *)calloc(threads ? threads : 1, 1);
    if (!tasks || !tids || !created) {
        free(tasks); free(tids); free(created);
        free(hdr_off); free(out_off); PyBuffer_Release(&view);
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    int err = 0;
    Py_BEGIN_ALLOW_THREADS
    size_t per = n_chunks ? (n_chunks + threads - 1) / threads : 0;
    for (int ti = 0; ti < threads; ti++) {
        size_t s = (size_t)ti * per;
        if (s >= n_chunks) break;
        size_t e = s + per; if (e > n_chunks) e = n_chunks;
        tasks[ti].frame = frame; tasks[ti].width = width;
        tasks[ti].chunk_hdr_off = hdr_off; tasks[ti].out_off = out_off;
        tasks[ti].out = obuf; tasks[ti].start_chunk = s; tasks[ti].end_chunk = e;
        tasks[ti].error = 0;
        if (pthread_create(&tids[ti], NULL, decomp_worker, &tasks[ti]) == 0) {
            created[ti] = 1;
        } else {
            decomp_worker(&tasks[ti]);  // run inline on failure
        }
    }
    for (int ti = 0; ti < threads; ti++) {
        if (created[ti]) pthread_join(tids[ti], NULL);
    }
    Py_END_ALLOW_THREADS

    for (int ti = 0; ti < threads; ti++) if (tasks[ti].error) err = 1;

    free(tasks); free(tids); free(created);
    free(hdr_off); free(out_off); PyBuffer_Release(&view);
    if (err) {
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "native decompress failed (corrupt frame?)");
        return NULL;
    }
    return out;
}

// ---------------------------------------------------------------------------
static PyMethodDef methods[] = {
    {"compress", py_compress, METH_VARARGS,
     "compress(buf, width, level, threads, chunk_size) -> bytes (Z4AIMF01 frame)"},
    {"decompress", py_decompress, METH_VARARGS,
     "decompress(frame, threads) -> bytes"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "z4ai._native_codec",
    "Fused multithreaded zstd chunked codec (Z4AIMF01).", -1, methods,
};

PyMODINIT_FUNC PyInit__native_codec(void) {
    return PyModule_Create(&moduledef);
}
