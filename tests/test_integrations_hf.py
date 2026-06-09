# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional HuggingFace / safetensors integration.

Skipped entirely when torch + safetensors are not installed (the `hf` extra).
"""
import os

import pytest

torch = pytest.importorskip("torch", reason="HF integration needs the `hf` extra")
safetensors = pytest.importorskip(
    "safetensors", reason="HF integration needs the `hf` extra"
)
import safetensors.torch as st  # noqa: E402

import z4ai  # noqa: E402
from z4ai import integrations as zi  # noqa: E402


@pytest.fixture
def state_dict():
    torch.manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(128, 64),
        "model.lm_head.weight": torch.zeros(64),  # tied-ish / sparse tensor
    }


def test_save_then_load_roundtrips(tmp_path, state_dict):
    path = str(tmp_path / "m.z4ai.safetensors")
    stats = zi.save_file(state_dict, path, metadata={"format": "pt"})
    assert stats["compressed_bytes"] < stats["original_bytes"]
    assert stats["ratio"] > 1.0

    # File is a ZSTN container, not a plain safetensors.
    with open(path, "rb") as fh:
        assert fh.read(4) == b"ZSTN"

    back = zi.load_file(path)
    assert set(back) == set(state_dict)
    for k in state_dict:
        assert torch.equal(back[k], state_dict[k])


def test_top_level_lazy_exports(tmp_path, state_dict):
    path = str(tmp_path / "m.z4ai.safetensors")
    z4ai.save_file(state_dict, path)
    back = z4ai.load_file(path)
    for k in state_dict:
        assert torch.equal(back[k], state_dict[k])


def test_plain_safetensors_passthrough(tmp_path, state_dict):
    path = str(tmp_path / "plain.safetensors")
    st.save_file(state_dict, path, metadata={"format": "pt"})
    back = zi.load_file(path)
    for k in state_dict:
        assert torch.equal(back[k], state_dict[k])


def test_enable_hf_patches_load_file_and_safe_open(tmp_path, state_dict):
    zstn = str(tmp_path / "m.z4ai.safetensors")
    plain = str(tmp_path / "plain.safetensors")
    zi.save_file(state_dict, zstn, metadata={"format": "pt"})
    st.save_file(state_dict, plain, metadata={"format": "pt"})

    tmp_root = tmp_path  # ensure no temp leak under the system temp dir either
    before = set(os.listdir(os.environ.get("TMPDIR", "/tmp")))

    zi.enable_hf()
    try:
        assert zi.is_enabled()

        # Patched load_file decodes the compressed file.
        loaded = st.load_file(zstn)
        for k in state_dict:
            assert torch.equal(loaded[k], state_dict[k])

        # Patched safe_open behaves like transformers expects, on the ZSTN file.
        with safetensors.safe_open(zstn, framework="pt", device="cpu") as f:
            assert set(f.keys()) == set(state_dict)
            assert f.metadata() == {"format": "pt"}
            t = f.get_tensor("model.embed_tokens.weight")
            assert torch.equal(t, state_dict["model.embed_tokens.weight"])
            assert f.get_slice("model.embed_tokens.weight").get_shape() == [128, 64]

        # Plain files still load through the patch unchanged.
        with safetensors.safe_open(plain, framework="pt") as f:
            assert set(f.keys()) == set(state_dict)
    finally:
        zi.disable_hf()

    # No decompressed temp file left behind.
    after = set(os.listdir(os.environ.get("TMPDIR", "/tmp")))
    leaked = {x for x in (after - before) if x.endswith(".safetensors")}
    assert not leaked
    assert not zi.is_enabled()
    _ = tmp_root


def test_enable_hf_is_idempotent_and_reversible():
    orig = st.load_file
    zi.enable_hf()
    once = st.load_file
    zi.enable_hf()  # second call is a no-op
    assert st.load_file is once
    zi.disable_hf()
    assert st.load_file is orig
