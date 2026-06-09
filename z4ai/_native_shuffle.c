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
 * SIMD-friendly byte-plane shuffle / unshuffle for z4ai.
 *
 * NumPy's strided copy tops out around 3 GB/s for the deinterleave, which is
 * the wall keeping the pure-Python codec below ZipNN's ~4.2 GB/s. These two
 * routines do the same transpose with tight, auto-vectorizable loops that the
 * compiler turns into SIMD (NEON/SSE), reaching memory bandwidth so the
 * transpose becomes negligible next to the entropy stage.
 *
 * Layout contract (matches z4ai/transforms.py):
 *   split(buf, width): given n = len(buf)//width whole elements, return a single
 *     buffer of length n*width where plane j occupies bytes [j*n : (j+1)*n] and
 *     plane_j[i] == buf[i*width + j].
 *   join(planes, width, n): inverse; planes is a list of `width` buffers each of
 *     length n, returns the interleaved buffer of length n*width.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

/* Specialized, branch-free transposes for the hot widths. Marking the inner
 * loops simple and contiguous lets clang/gcc vectorize them. */

static void split_w2(const unsigned char *src, unsigned char *dst, Py_ssize_t n) {
    unsigned char *p0 = dst, *p1 = dst + n;
    for (Py_ssize_t i = 0; i < n; i++) {
        p0[i] = src[2 * i];
        p1[i] = src[2 * i + 1];
    }
}

static void split_w4(const unsigned char *src, unsigned char *dst, Py_ssize_t n) {
    unsigned char *p0 = dst, *p1 = dst + n, *p2 = dst + 2 * n, *p3 = dst + 3 * n;
    for (Py_ssize_t i = 0; i < n; i++) {
        p0[i] = src[4 * i];
        p1[i] = src[4 * i + 1];
        p2[i] = src[4 * i + 2];
        p3[i] = src[4 * i + 3];
    }
}

/* fp64 / int64 / uint64.  Without this the 8-byte transpose fell to
 * split_generic, whose nested loop the compiler does not vectorize -- measured
 * ~3.4 GB/s, barely above NumPy, vs 36-55 GB/s for the unrolled w2/w4 paths. */
static void split_w8(const unsigned char *src, unsigned char *dst, Py_ssize_t n) {
    unsigned char *p0 = dst,         *p1 = dst + n,     *p2 = dst + 2 * n,
                  *p3 = dst + 3 * n, *p4 = dst + 4 * n, *p5 = dst + 5 * n,
                  *p6 = dst + 6 * n, *p7 = dst + 7 * n;
    for (Py_ssize_t i = 0; i < n; i++) {
        p0[i] = src[8 * i];     p1[i] = src[8 * i + 1];
        p2[i] = src[8 * i + 2]; p3[i] = src[8 * i + 3];
        p4[i] = src[8 * i + 4]; p5[i] = src[8 * i + 5];
        p6[i] = src[8 * i + 6]; p7[i] = src[8 * i + 7];
    }
}

static void split_generic(const unsigned char *src, unsigned char *dst,
                          Py_ssize_t n, int width) {
    for (int j = 0; j < width; j++) {
        unsigned char *pj = dst + (Py_ssize_t)j * n;
        const unsigned char *s = src + j;
        for (Py_ssize_t i = 0; i < n; i++) {
            pj[i] = s[(Py_ssize_t)i * width];
        }
    }
}

static void join_w2(unsigned char *const *planes, unsigned char *dst, Py_ssize_t n) {
    const unsigned char *p0 = planes[0], *p1 = planes[1];
    for (Py_ssize_t i = 0; i < n; i++) {
        dst[2 * i] = p0[i];
        dst[2 * i + 1] = p1[i];
    }
}

static void join_w4(unsigned char *const *planes, unsigned char *dst, Py_ssize_t n) {
    const unsigned char *p0 = planes[0], *p1 = planes[1];
    const unsigned char *p2 = planes[2], *p3 = planes[3];
    for (Py_ssize_t i = 0; i < n; i++) {
        dst[4 * i] = p0[i];
        dst[4 * i + 1] = p1[i];
        dst[4 * i + 2] = p2[i];
        dst[4 * i + 3] = p3[i];
    }
}

static void join_w8(unsigned char *const *planes, unsigned char *dst, Py_ssize_t n) {
    const unsigned char *p0 = planes[0], *p1 = planes[1], *p2 = planes[2],
                        *p3 = planes[3], *p4 = planes[4], *p5 = planes[5],
                        *p6 = planes[6], *p7 = planes[7];
    for (Py_ssize_t i = 0; i < n; i++) {
        dst[8 * i] = p0[i];     dst[8 * i + 1] = p1[i];
        dst[8 * i + 2] = p2[i]; dst[8 * i + 3] = p3[i];
        dst[8 * i + 4] = p4[i]; dst[8 * i + 5] = p5[i];
        dst[8 * i + 6] = p6[i]; dst[8 * i + 7] = p7[i];
    }
}

static void join_generic(unsigned char *const *planes, unsigned char *dst,
                         Py_ssize_t n, int width) {
    for (int j = 0; j < width; j++) {
        const unsigned char *pj = planes[j];
        unsigned char *d = dst + j;
        for (Py_ssize_t i = 0; i < n; i++) {
            d[(Py_ssize_t)i * width] = pj[i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Native bf16 field split / join.
 *
 * bf16 little-endian uint16 layout (MSB-first): [ S | E E E E E E E E | M*7 ].
 *   u    = src[2i] | (src[2i+1] << 8)
 *   sign = (u >> 15) & 1
 *   exp  = (u >> 7)  & 0xFF
 *   mant = u & 0x7F
 *
 * Stream layout (byte-exact with bitfield.py's NumPy path):
 *   sign     - bit-packed, big bit-order (np.packbits default): element i sets
 *              bit (7 - (i & 7)) of sign_packed[i >> 3]; length ceil(n/8).
 *   exponent - n bytes, one exp integer each.
 *   mantissa - n bytes (bf16 mantissa fits one byte), one mant integer each.
 *
 * Each routine uses tight, contiguous, compiler-vectorizable loops so it runs
 * near memory bandwidth and replaces the multi-pass NumPy
 * shift/mask/packbits/astype chain that was the top bf16 compress hotspot and a
 * bf16 decompress bottleneck.
 * ------------------------------------------------------------------------- */

static void bf16_split(const unsigned char *src, Py_ssize_t n,
                       unsigned char *sign_packed, unsigned char *exp,
                       unsigned char *mant) {
    /* Pass 1 - exponent + mantissa.  Two contiguous output streams from a
     * stride-2 input: a plain byte deinterleave the compiler vectorizes
     * (NEON vld2 / SSE).  The previous single loop folded the sign bit-pack
     * (`sign_packed[i>>3] |= ...`, a read-modify-write aliasing the same byte
     * across 8 iterations) into this loop, which serialized it - measured
     * ~3.7x slower than NumPy's separately-vectorized passes.  Splitting the
     * sign-pack into its own loop (pass 2) lets both vectorize. */
    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned int lo = src[2 * i];      /* E MMMMMMM (low byte)  */
        unsigned int hi = src[2 * i + 1];  /* S EEEEEEE (high byte) */
        exp[i] = (unsigned char)((hi << 1) | (lo >> 7));
        mant[i] = (unsigned char)(lo & 0x7F);
    }
    /* Pass 2 - pack sign bits, 8 elements per output byte, MSB-first
     * (np.packbits order: element i sets bit 7-(i&7)).  Each output byte is
     * independent (no cross-iteration aliasing), and the eight source reads are
     * the high bytes at stride 2, so this is branch-free and fully unrolled. */
    Py_ssize_t full = n >> 3;
    for (Py_ssize_t b = 0; b < full; b++) {
        const unsigned char *s = src + (b << 4) + 1;  /* high bytes, stride 2 */
        sign_packed[b] = (unsigned char)(
            (s[0]  & 0x80)        |
            ((s[2]  & 0x80) >> 1) |
            ((s[4]  & 0x80) >> 2) |
            ((s[6]  & 0x80) >> 3) |
            ((s[8]  & 0x80) >> 4) |
            ((s[10] & 0x80) >> 5) |
            ((s[12] & 0x80) >> 6) |
            ((s[14] & 0x80) >> 7));
    }
    Py_ssize_t rem = n & 7;
    if (rem) {
        unsigned char byte = 0;
        Py_ssize_t base = full << 3;
        for (Py_ssize_t k = 0; k < rem; k++) {
            byte |= (unsigned char)((src[2 * (base + k) + 1] & 0x80) >> k);
        }
        sign_packed[full] = byte;
    }
}

static void bf16_join(const unsigned char *sign_packed, const unsigned char *exp,
                      const unsigned char *mant, unsigned char *dst, Py_ssize_t n) {
    /* Process 8 elements per packed sign byte: load the sign byte ONCE and
     * extract its bits with constant shifts.  The scalar version's per-element
     * `sign_packed[i>>3] >> (7 - (i & 7))` reloaded the same byte every
     * iteration with a data-dependent shift, which blocked vectorization.  The
     * inner body is a stride-2 byte interleave the compiler turns into NEON
     * vst2 / SSE. */
    Py_ssize_t full = n >> 3;
    for (Py_ssize_t b = 0; b < full; b++) {
        unsigned int sb = sign_packed[b];
        const unsigned char *e = exp + (b << 3);
        const unsigned char *m = mant + (b << 3);
        unsigned char *d = dst + (b << 4);
        for (int k = 0; k < 8; k++) {
            unsigned int s = (sb >> (7 - k)) & 1;
            unsigned int u = (s << 15) | ((unsigned int)e[k] << 7) | (m[k] & 0x7F);
            d[2 * k] = (unsigned char)(u & 0xFF);
            d[2 * k + 1] = (unsigned char)(u >> 8);
        }
    }
    for (Py_ssize_t i = full << 3; i < n; i++) {
        unsigned int s = (sign_packed[i >> 3] >> (7 - (i & 7))) & 1;
        unsigned int u = (s << 15) | ((unsigned int)exp[i] << 7) | (mant[i] & 0x7F);
        dst[2 * i] = (unsigned char)(u & 0xFF);
        dst[2 * i + 1] = (unsigned char)((u >> 8) & 0xFF);
    }
}

static PyObject *py_bf16_split(PyObject *self, PyObject *args) {
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf)) {
        return NULL;
    }
    Py_ssize_t n = buf.len / 2;  /* whole bf16 elements */
    Py_ssize_t sbytes = (n + 7) / 8;
    PyObject *sign = PyBytes_FromStringAndSize(NULL, sbytes);
    PyObject *exp = PyBytes_FromStringAndSize(NULL, n);
    PyObject *mant = PyBytes_FromStringAndSize(NULL, n);
    if (sign == NULL || exp == NULL || mant == NULL) {
        Py_XDECREF(sign); Py_XDECREF(exp); Py_XDECREF(mant);
        PyBuffer_Release(&buf);
        return NULL;
    }
    const unsigned char *src = (const unsigned char *)buf.buf;
    unsigned char *sp = (unsigned char *)PyBytes_AS_STRING(sign);
    unsigned char *ep = (unsigned char *)PyBytes_AS_STRING(exp);
    unsigned char *mp = (unsigned char *)PyBytes_AS_STRING(mant);

    Py_BEGIN_ALLOW_THREADS
    bf16_split(src, n, sp, ep, mp);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    return Py_BuildValue("(NNN)", sign, exp, mant);
}

static PyObject *py_bf16_join(PyObject *self, PyObject *args) {
    Py_buffer sign, exp, mant;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "y*y*y*n", &sign, &exp, &mant, &n)) {
        return NULL;
    }
    if (exp.len < n || mant.len < n || sign.len < (n + 7) / 8) {
        PyBuffer_Release(&sign); PyBuffer_Release(&exp); PyBuffer_Release(&mant);
        PyErr_SetString(PyExc_ValueError, "bf16_join stream length too short for n");
        return NULL;
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, n * 2);
    if (out == NULL) {
        PyBuffer_Release(&sign); PyBuffer_Release(&exp); PyBuffer_Release(&mant);
        return NULL;
    }
    unsigned char *dst = (unsigned char *)PyBytes_AS_STRING(out);

    Py_BEGIN_ALLOW_THREADS
    bf16_join((const unsigned char *)sign.buf, (const unsigned char *)exp.buf,
              (const unsigned char *)mant.buf, dst, n);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&sign); PyBuffer_Release(&exp); PyBuffer_Release(&mant);
    return out;
}

static PyObject *py_split(PyObject *self, PyObject *args) {
    Py_buffer buf;
    int width;
    if (!PyArg_ParseTuple(args, "y*i", &buf, &width)) {
        return NULL;
    }
    if (width <= 0) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError, "width must be positive");
        return NULL;
    }
    Py_ssize_t n = buf.len / width;          /* whole elements */
    Py_ssize_t aligned = n * width;
    PyObject *out = PyBytes_FromStringAndSize(NULL, aligned);
    if (out == NULL) {
        PyBuffer_Release(&buf);
        return NULL;
    }
    const unsigned char *src = (const unsigned char *)buf.buf;
    unsigned char *dst = (unsigned char *)PyBytes_AS_STRING(out);

    Py_BEGIN_ALLOW_THREADS
    if (width == 2)      split_w2(src, dst, n);
    else if (width == 4) split_w4(src, dst, n);
    else if (width == 8) split_w8(src, dst, n);
    else if (width == 1) memcpy(dst, src, (size_t)aligned);
    else                 split_generic(src, dst, n, width);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    return out;
}

static PyObject *py_join(PyObject *self, PyObject *args) {
    PyObject *plane_seq;
    int width;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "Oin", &plane_seq, &width, &n)) {
        return NULL;
    }
    if (width <= 0) {
        PyErr_SetString(PyExc_ValueError, "width must be positive");
        return NULL;
    }
    PyObject *fast = PySequence_Fast(plane_seq, "planes must be a sequence");
    if (fast == NULL) return NULL;
    if (PySequence_Fast_GET_SIZE(fast) != width) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "len(planes) must equal width");
        return NULL;
    }

    /* Acquire a buffer view on every plane. */
    Py_buffer *views = (Py_buffer *)PyMem_Calloc(width, sizeof(Py_buffer));
    unsigned char **ptrs = (unsigned char **)PyMem_Calloc(width, sizeof(unsigned char *));
    if (views == NULL || ptrs == NULL) {
        PyMem_Free(views); PyMem_Free(ptrs); Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    int got = 0;
    for (int j = 0; j < width; j++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, j);  /* borrowed */
        if (PyObject_GetBuffer(item, &views[j], PyBUF_SIMPLE) != 0) goto fail;
        if (views[j].len != n) {
            PyErr_SetString(PyExc_ValueError, "plane length != n");
            PyBuffer_Release(&views[j]);
            goto fail;
        }
        ptrs[j] = (unsigned char *)views[j].buf;
        got = j + 1;
    }

    PyObject *out = PyBytes_FromStringAndSize(NULL, n * width);
    if (out == NULL) goto fail;
    unsigned char *dst = (unsigned char *)PyBytes_AS_STRING(out);

    Py_BEGIN_ALLOW_THREADS
    if (width == 2)      join_w2(ptrs, dst, n);
    else if (width == 4) join_w4(ptrs, dst, n);
    else if (width == 8) join_w8(ptrs, dst, n);
    else if (width == 1) memcpy(dst, ptrs[0], (size_t)n);
    else                 join_generic(ptrs, dst, n, width);
    Py_END_ALLOW_THREADS

    for (int j = 0; j < width; j++) PyBuffer_Release(&views[j]);
    PyMem_Free(views); PyMem_Free(ptrs); Py_DECREF(fast);
    return out;

fail:
    for (int j = 0; j < got; j++) PyBuffer_Release(&views[j]);
    PyMem_Free(views); PyMem_Free(ptrs); Py_DECREF(fast);
    return NULL;
}

static PyMethodDef methods[] = {
    {"split", py_split, METH_VARARGS,
     "split(buf, width) -> bytes: concatenated byte planes (plane j at [j*n:(j+1)*n])."},
    {"join", py_join, METH_VARARGS,
     "join(planes, width, n) -> bytes: interleave width planes of length n."},
    {"bf16_split", py_bf16_split, METH_VARARGS,
     "bf16_split(buf) -> (sign_packed, exponent, mantissa): split bf16 fields."},
    {"bf16_join", py_bf16_join, METH_VARARGS,
     "bf16_join(sign_packed, exponent, mantissa, n) -> bytes: rebuild bf16 buffer."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_native_shuffle",
    "SIMD byte-plane shuffle/unshuffle for z4ai.", -1, methods,
};

PyMODINIT_FUNC PyInit__native_shuffle(void) {
    return PyModule_Create(&moduledef);
}
