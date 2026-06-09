<!-- SPDX-License-Identifier: Apache-2.0 -->

# Installation

z4ai requires **Python >= 3.9** and depends only on `numpy` and `zstandard`.

```bash
pip install z4ai
```

From a checkout of the repository:

```bash
pip install .
# or, for development:
pip install -e ".[test]"
```

## Optional extras

| Extra | Installs | Enables |
|-------|----------|---------|
| `z4ai[maxratio]` | `brotli` | the `effort="max"` high-ratio tier (a chunk-parallel context-modeling backend). z4ai degrades gracefully to the fast default without it. |
| `z4ai[test]` | `pytest`, `brotli` | the test suite |
| `z4ai[benchmark]` | `zipnn`, `brotli` | the head-to-head benchmarks in `benchmarks/` |

## Optional native acceleration

z4ai ships small C sources (a SIMD byte-shuffle, an interleaved rANS coder, and a
fused chunked codec) that are compiled on demand. They are **optional** - without
a C toolchain z4ai runs as pure Python (NumPy + `zstandard`). To build them ahead
of time:

```bash
python -m z4ai.build_native
```

The on-disk frame format is identical with or without the native code, so frames
are byte-interoperable between the two.
