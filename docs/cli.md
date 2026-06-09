<!-- SPDX-License-Identifier: Apache-2.0 -->

# Command line

Installing z4ai puts a `z4ai` executable on your `PATH` (a console-script entry
point). It is a thin, dependency-light wrapper over the buffer API.

```bash
# Compress a raw weight dump as fp32 (4-byte elements) -> weights.z4ai
z4ai compress weights.bin -o weights.z4ai --dtype fp32

# Decompress back to the original bytes
z4ai decompress weights.z4ai -o weights.bin

# Inspect a blob without decompressing it (per-plane breakdown + ratio)
z4ai info weights.z4ai

# Print the version
z4ai --version
```

## Piping

Use `-` for either path to read stdin / write stdout:

```bash
cat weights.bin | z4ai compress - --dtype bf16 > weights.z4ai
z4ai decompress - < weights.z4ai > weights.bin
```

If z4ai is not on your `PATH` (e.g. an editable checkout without the entry point),
the same CLI is available as a module:

```bash
python -m z4ai compress weights.bin --dtype fp32
```

## Default output names

When `-o` is omitted, `compress` appends `.z4ai` and `decompress` strips a
trailing `.z4ai`. Decompression detects the format from the frame's magic bytes,
so the extension is a convenience, not a requirement.

Run `z4ai <command> --help` for the full flag list.
