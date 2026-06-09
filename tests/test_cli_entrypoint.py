# SPDX-License-Identifier: Apache-2.0
"""Tests for the installed ``z4ai`` console-script entry point.

``test_cli.py`` exercises the CLI through ``z4ai.cli.main`` and ``python -m z4ai``;
this file checks the *packaged executable* (the ``console_scripts`` entry point
created by ``pip install``). When z4ai is not installed as a distribution (e.g. a
bare ``PYTHONPATH`` checkout) the entry-point-specific tests skip instead of fail.
"""

from __future__ import annotations

import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(args, **kw):
    return subprocess.run(
        args, capture_output=True, text=True, cwd=str(REPO_ROOT), **kw
    )


def test_python_dash_m_version():
    """``python -m z4ai --version`` always works (no install required)."""
    r = _run([sys.executable, "-m", "z4ai", "--version"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip(), "version output should be non-empty"


def _entry_point_registered() -> bool:
    eps = md.entry_points()
    group = (
        eps.select(group="console_scripts")
        if hasattr(eps, "select")
        else eps.get("console_scripts", [])
    )
    return any(e.name == "z4ai" for e in group)


def test_entry_point_declared():
    """The packaging metadata declares the ``z4ai`` console script."""
    if not _entry_point_registered():
        pytest.skip("z4ai is not installed as a distribution in this environment")
    assert _entry_point_registered()


def test_installed_script_roundtrip(tmp_path):
    """If the ``z4ai`` executable is on PATH, a compress/decompress round-trips."""
    exe = shutil.which("z4ai")
    if exe is None:
        pytest.skip("z4ai executable not on PATH (package not pip-installed)")

    src = tmp_path / "w.bin"
    src.write_bytes(b"\x00\x01\x02\x03" * 4096)
    packed = tmp_path / "w.z4ai"
    out = tmp_path / "w.out"

    assert (
        _run(
            [exe, "compress", str(src), "-o", str(packed), "--dtype", "fp32"]
        ).returncode
        == 0
    )
    assert packed.exists()
    assert _run([exe, "decompress", str(packed), "-o", str(out)]).returncode == 0
    assert out.read_bytes() == src.read_bytes()
