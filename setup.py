# SPDX-License-Identifier: Apache-2.0

"""Build shim for the optional native accelerators.

Project metadata lives in ``pyproject.toml``; this file exists only to declare
the C extensions, which TOML cannot express. Every extension is marked
``optional=True`` so a missing/old compiler (or absent zstd headers) degrades
gracefully - the wheel still builds and z4ai falls back to NumPy / pure-Python at
import time. The accelerators are pure speed upgrades, never hard requirements.

Two extensions:
* ``_native_shuffle`` - SIMD byte-plane transpose (no external deps).
* ``_native_codec``   - fused multithreaded chunked codec linking **libzstd**
  (the throughput path; ``z4ai.chunked`` falls back to pure Python without it).
"""
import os
import platform
import shutil
import subprocess

from setuptools import Extension, setup

# -O3 + arch tuning unlock the SIMD the inner loops are written for.
# -march=native is gated to non-Windows where the flag exists.
extra_compile_args = ["-O3", "-funroll-loops"]
if platform.system() in ("Darwin", "Linux"):
    extra_compile_args.append("-march=native")

ext_modules = [
    Extension(
        "z4ai._native_shuffle",
        sources=["z4ai/_native_shuffle.c"],
        extra_compile_args=extra_compile_args,
        optional=True,  # build failure must not abort installation
    )
]


def _zstd_flags():
    """Locate libzstd dev headers/libs; return (inc, libdir, libs) or None."""
    if shutil.which("pkg-config"):
        try:
            cflags = subprocess.check_output(
                ["pkg-config", "--cflags", "libzstd"], text=True
            ).split()
            libs = subprocess.check_output(
                ["pkg-config", "--libs", "libzstd"], text=True
            ).split()
            return ([f[2:] for f in cflags if f.startswith("-I")],
                    [f[2:] for f in libs if f.startswith("-L")],
                    [f[2:] for f in libs if f.startswith("-l")] or ["zstd"])
        except Exception:  # noqa: BLE001
            pass
    for prefix in ("/opt/homebrew/opt/zstd", "/opt/homebrew", "/usr/local"):
        if os.path.exists(os.path.join(prefix, "include", "zstd.h")):
            return ([os.path.join(prefix, "include")],
                    [os.path.join(prefix, "lib")], ["zstd"])
    if os.path.exists("/usr/include/zstd.h"):
        return ([], [], ["zstd"])
    return None


_zf = _zstd_flags()
if _zf is not None:
    _inc, _libdir, _libs = _zf
    ext_modules.append(
        Extension(
            "z4ai._native_codec",
            sources=["z4ai/_native_codec.c"],
            include_dirs=_inc,
            library_dirs=_libdir,
            libraries=_libs,
            extra_compile_args=extra_compile_args,
            extra_link_args=["-lpthread"],
            optional=True,  # build failure must not abort installation
        )
    )

setup(ext_modules=ext_modules)
