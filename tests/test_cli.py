# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``z4ai`` command-line interface (z4ai/cli.py)."""

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from z4ai import cli  # noqa: E402


def _bf16_like_bytes(n_elems: int, seed: int = 0) -> bytes:
    """Deterministic, low-exponent-entropy 2-byte-element buffer.

    Mimics BF16 weights: a near-constant high byte (sign+exponent) and a
    pseudo-random low byte (mantissa).  No NumPy required.
    """
    out = bytearray()
    state = seed or 0x9E3779B1
    for _ in range(n_elems):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        mantissa = state & 0xFF
        # exponent byte clusters tightly around two values -> compressible
        hi = 0x3F if (state >> 8) & 1 else 0x3E
        out += struct.pack("<BB", mantissa, hi)
    return bytes(out)


# ---------------------------------------------------------------------------
# width resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("bf16", 2),
        ("fp16", 2),
        ("float16", 2),
        ("fp32", 4),
        ("float32", 4),
        ("fp64", 8),
        ("uint8", 1),
        ("fp8_e4m3", 1),
    ],
)
def test_resolve_width_from_dtype(dtype, expected):
    assert cli._resolve_width(dtype, None) == expected


def test_resolve_width_explicit_overrides_dtype():
    assert cli._resolve_width("bf16", 4) == 4


def test_resolve_width_default_is_one():
    assert cli._resolve_width(None, None) == 1


def test_resolve_width_unknown_dtype_errors():
    with pytest.raises(SystemExit):
        cli._resolve_width("not_a_dtype", None)


def test_resolve_width_bad_width_errors():
    with pytest.raises(SystemExit):
        cli._resolve_width(None, 0)


def test_default_out_paths():
    assert cli._default_out("m.bin", compressing=True) == "m.bin.z4ai"
    assert cli._default_out("m.bin.z4ai", compressing=False) == "m.bin"
    assert cli._default_out("m.bin", compressing=False) == "m.bin.out"
    assert cli._default_out("-", compressing=True) == "-"


# ---------------------------------------------------------------------------
# end-to-end main() round-trips on files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,width", [("bf16", 2), ("fp32", 4), ("uint8", 1)])
def test_main_roundtrip_files(tmp_path, dtype, width):
    original = _bf16_like_bytes(4096, seed=width)
    src = tmp_path / "data.bin"
    blob = tmp_path / "data.z4ai"
    restored = tmp_path / "restored.bin"
    src.write_bytes(original)

    rc = cli.main(["compress", str(src), "-o", str(blob), "--dtype", dtype, "-q"])
    assert rc == 0
    assert blob.read_bytes()[:4] == b"Z4AI"

    rc = cli.main(["decompress", str(blob), "-o", str(restored), "-q"])
    assert rc == 0
    assert restored.read_bytes() == original


def test_main_default_output_paths(tmp_path):
    original = _bf16_like_bytes(2048)
    src = tmp_path / "weights.bin"
    src.write_bytes(original)

    assert cli.main(["compress", str(src), "--dtype", "bf16", "-q"]) == 0
    blob = tmp_path / "weights.bin.z4ai"
    assert blob.exists() and blob.read_bytes()[:4] == b"Z4AI"

    assert cli.main(["decompress", str(blob), "-q"]) == 0
    # "weights.bin.z4ai" -> strips .z4ai -> "weights.bin" (overwrites src)
    assert (tmp_path / "weights.bin").read_bytes() == original


def test_main_compress_actually_shrinks_weights(tmp_path):
    original = _bf16_like_bytes(1 << 16)  # 128 KiB, low-entropy exponent plane
    src = tmp_path / "w.bin"
    blob = tmp_path / "w.z4ai"
    src.write_bytes(original)
    cli.main(["compress", str(src), "-o", str(blob), "--dtype", "bf16", "-q"])
    assert blob.stat().st_size < src.stat().st_size


def test_decompress_rejects_non_z4ai(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a z4ai file at all")
    with pytest.raises(SystemExit):
        cli.main(["decompress", str(junk), "-o", str(tmp_path / "x"), "-q"])


def test_info_command(tmp_path, capsys):
    original = _bf16_like_bytes(8192)
    src = tmp_path / "w.bin"
    blob = tmp_path / "w.z4ai"
    src.write_bytes(original)
    cli.main(["compress", str(src), "-o", str(blob), "--dtype", "bf16", "-q"])

    rc = cli.main(["info", str(blob)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Z4AI" in out
    # The codec's AUTO best-of picks whichever transform is smallest for the
    # given data (bit-field / byte-plane / opaque whole-buffer Zstd), so the
    # reported transform/plane count is data-dependent and not asserted here.
    # Check the stable, human-facing fields instead.
    assert "original:" in out
    assert "compressed:" in out
    assert "ratio:" in out


def test_no_escalate_flag(tmp_path):
    original = _bf16_like_bytes(4096)
    src = tmp_path / "w.bin"
    blob = tmp_path / "w.z4ai"
    restored = tmp_path / "r.bin"
    src.write_bytes(original)
    assert cli.main(
        ["compress", str(src), "-o", str(blob), "--dtype", "bf16",
         "--no-escalate", "-q"]
    ) == 0
    assert cli.main(["decompress", str(blob), "-o", str(restored), "-q"]) == 0
    assert restored.read_bytes() == original


# ---------------------------------------------------------------------------
# stdin/stdout and `python -m z4ai`
# ---------------------------------------------------------------------------

def test_python_dash_m_stdin_stdout_roundtrip(tmp_path):
    original = _bf16_like_bytes(4096, seed=7)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    comp = subprocess.run(
        [sys.executable, "-m", "z4ai", "compress", "-", "--dtype", "bf16", "-q"],
        input=original, stdout=subprocess.PIPE, env=env, check=True,
    )
    assert comp.stdout[:4] == b"Z4AI"

    deco = subprocess.run(
        [sys.executable, "-m", "z4ai", "decompress", "-", "-q"],
        input=comp.stdout, stdout=subprocess.PIPE, env=env, check=True,
    )
    assert deco.stdout == original


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
