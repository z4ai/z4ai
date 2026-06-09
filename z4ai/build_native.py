# SPDX-License-Identifier: Apache-2.0

"""Compile the optional native C extensions in place.

Run directly::

    python -m z4ai.build_native

Two extensions are built (both optional, both pure speed upgrades):

* ``_native_shuffle`` — SIMD byte-plane transpose (no external deps).
* ``_native_codec``   — fused, multithreaded chunked codec that links
  **libzstd** and runs the whole split -> zstd -> interleave loop in C with the
  GIL released (the throughput path that closes the gap to zipnn).  It requires
  the zstd development headers; if they are not found this one is skipped and
  :mod:`z4ai.chunked` transparently uses its pure-Python pipeline.

Everything is intentionally standalone (no pyproject build hook required) so the
accelerators can be built ad hoc on any machine that has a C compiler. If a
build fails or no compiler is present, z4ai still works — :mod:`z4ai._accel`
falls back to the NumPy transpose and :mod:`z4ai.chunked` to pure Python.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def build(verbose: bool = True) -> str:
    """Compile the extension next to this package; return the artifact path."""
    from setuptools import Extension
    from setuptools.dist import Distribution
    from setuptools.command.build_ext import build_ext

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "_native_shuffle.c")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    # -O3 + native arch unlocks the SIMD the inner loops are written for.
    extra = ["-O3", "-funroll-loops"]
    if sysconfig.get_platform().startswith(("macosx", "linux")):
        extra.append("-march=native")

    ext = Extension(
        "z4ai._native_shuffle",
        sources=[src],
        extra_compile_args=extra,
    )
    dist = Distribution({"name": "z4ai", "ext_modules": [ext]})
    cmd = build_ext(dist)
    cmd.build_lib = os.path.dirname(here)  # so z4ai/ ends up in the right place
    cmd.build_temp = tempfile.mkdtemp(prefix="z4ai_build_")
    cmd.inplace = 1
    if not verbose:
        cmd.verbose = 0
    cmd.ensure_finalized()
    cmd.run()
    return cmd.get_ext_fullpath("z4ai._native_shuffle")


def _zstd_flags():
    """Locate libzstd dev headers/libs.

    Tries, in order: ``pkg-config libzstd``, common Homebrew prefixes, then the
    bare ``-lzstd`` (system default search path).  Returns
    ``(include_dirs, library_dirs, libraries)`` or ``None`` when zstd headers
    cannot be found, so the caller can skip the codec extension gracefully.
    """
    # 1) pkg-config — the portable, correct source of truth.
    if shutil.which("pkg-config"):
        try:
            cflags = subprocess.check_output(
                ["pkg-config", "--cflags", "libzstd"], text=True
            ).split()
            libs = subprocess.check_output(
                ["pkg-config", "--libs", "libzstd"], text=True
            ).split()
            inc = [f[2:] for f in cflags if f.startswith("-I")]
            libdir = [f[2:] for f in libs if f.startswith("-L")]
            lib = [f[2:] for f in libs if f.startswith("-l")] or ["zstd"]
            return inc, libdir, lib
        except Exception:  # noqa: BLE001
            pass
    # 2) Homebrew / common prefixes.
    for prefix in ("/opt/homebrew/opt/zstd", "/opt/homebrew", "/usr/local",
                   "/usr/local/opt/zstd"):
        if os.path.exists(os.path.join(prefix, "include", "zstd.h")):
            return ([os.path.join(prefix, "include")],
                    [os.path.join(prefix, "lib")], ["zstd"])
    # 3) System default search path (e.g. Linux distro -dev package).
    for inc in ("/usr/include/zstd.h", "/usr/include/x86_64-linux-gnu/zstd.h"):
        if os.path.exists(inc):
            return ([], [], ["zstd"])
    return None


def build_codec(verbose: bool = True):
    """Compile ``_native_codec`` (links libzstd); return path or None if skipped."""
    from setuptools import Extension
    from setuptools.dist import Distribution
    from setuptools.command.build_ext import build_ext

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "_native_codec.c")
    if not os.path.exists(src):
        return None
    flags = _zstd_flags()
    if flags is None:
        if verbose:
            print("zstd dev headers not found; skipping _native_codec "
                  "(chunked path will use pure Python)")
        return None
    inc, libdir, lib = flags

    extra = ["-O3", "-funroll-loops"]
    if sysconfig.get_platform().startswith(("macosx", "linux")):
        extra.append("-march=native")

    ext = Extension(
        "z4ai._native_codec",
        sources=[src],
        include_dirs=inc,
        library_dirs=libdir,
        libraries=lib,
        extra_compile_args=extra,
        # pthread is in libc on macOS; -lpthread is the portable spelling.
        extra_link_args=["-lpthread"],
    )
    dist = Distribution({"name": "z4ai", "ext_modules": [ext]})
    cmd = build_ext(dist)
    cmd.build_lib = os.path.dirname(here)
    cmd.build_temp = tempfile.mkdtemp(prefix="z4ai_codec_build_")
    cmd.inplace = 1
    if not verbose:
        cmd.verbose = 0
    cmd.ensure_finalized()
    cmd.run()
    return cmd.get_ext_fullpath("z4ai._native_codec")


if __name__ == "__main__":
    failed = False
    try:
        print(f"built {build()}")
    except Exception as exc:  # noqa: BLE001
        print(f"_native_shuffle build failed ({exc!r}); NumPy transpose fallback")
        failed = True
    try:
        path = build_codec()
        if path:
            print(f"built {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"_native_codec build failed ({exc!r}); pure-Python chunked fallback")
    sys.exit(1 if failed else 0)
