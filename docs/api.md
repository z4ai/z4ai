<!-- SPDX-License-Identifier: Apache-2.0 -->

# API reference

## Core codec

```{eval-rst}
.. currentmodule:: z4ai

.. autofunction:: compress
.. autofunction:: decompress
.. autofunction:: compress_array
.. autofunction:: decompress_array
.. autofunction:: peek_is_z4ai
.. autofunction:: dtype_width
```

## Checkpoint deltas

```{eval-rst}
.. autofunction:: z4ai.compress_delta
.. autofunction:: z4ai.decompress_delta
```

## Model-aware deltas (`.safetensors`)

```{eval-rst}
.. automodule:: z4ai.model_delta
   :members:
   :undoc-members:
```

## safetensors container (random access)

```{eval-rst}
.. automodule:: z4ai.safetensors
   :members: compress_bytes, decompress_bytes, compress_file, decompress_file, ZstnReader
```

## High-throughput chunked codec

```{eval-rst}
.. automodule:: z4ai.chunked
   :members: compress, decompress
```
