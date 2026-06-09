# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures / path setup for the z4ai test suite."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))


@pytest.fixture(scope="session")
def z4ai():
    """Import the codec or skip the whole suite if core isn't landed yet."""
    return pytest.importorskip("z4ai", reason="z4ai core not implemented yet")
