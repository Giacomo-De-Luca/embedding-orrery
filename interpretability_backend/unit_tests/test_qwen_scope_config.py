"""Unit tests for the generalized Qwen-scope SAE config + loader.

Pure CPU tests — no model or SAE download. The loader tests fake the
HuggingFace weights with a tiny ``torch.save`` dict and monkeypatch
``huggingface_hub.hf_hub_download`` (the loader imports it inside the
function, so we patch it at the source module).
"""

from pathlib import Path

import pytest
import torch

from interpret.sae import HookType, QwenScopeSAEConfig
from interpret.sae.loading import (
    _config_to_key,
    _load_qwen_scope_sae,
    clear_sae_cache,
    load_sae,
)
from interpret.sae.source_ids import qwen_source_id


@pytest.fixture(autouse=True)
def _clear_cache():
    """Keep the module-level SAE cache from leaking across tests."""
    clear_sae_cache()
    yield
    clear_sae_cache()


# --------------------------------------------------------------------------- #
# Group A — config resolution (pure dataclass logic, no I/O)
# --------------------------------------------------------------------------- #


def test_repo_id_1_7b():
    cfg = QwenScopeSAEConfig(layer_index=15)
    assert cfg.repo_id == "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"


def test_repo_id_8b():
    cfg = QwenScopeSAEConfig(layer_index=10, model_size="8B")
    assert cfg.repo_id == "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"


def test_repo_id_27b_has_no_base_segment():
    cfg = QwenScopeSAEConfig(layer_index=30, model_size="27B")
    assert cfg.repo_id == "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
    assert "-Base" not in cfg.repo_id
    assert "Qwen3.5" in cfg.repo_id


def test_repo_id_k100_suffix():
    cfg = QwenScopeSAEConfig(layer_index=0, k=100)
    assert cfg.repo_id.endswith("-W32K-L0_100")


def test_default_width_filled_from_registry():
    assert QwenScopeSAEConfig(layer_index=0, model_size="1.7B").width == "32k"
    assert QwenScopeSAEConfig(layer_index=0, model_size="8B").width == "64k"
    assert QwenScopeSAEConfig(layer_index=0, model_size="27B").width == "80k"


def test_d_sae_per_width():
    assert QwenScopeSAEConfig(layer_index=0, model_size="1.7B").d_sae == 32768
    assert QwenScopeSAEConfig(layer_index=0, model_size="8B").d_sae == 65536
    assert QwenScopeSAEConfig(layer_index=0, model_size="27B").d_sae == 81920


def test_d_in_filled_from_registry():
    assert QwenScopeSAEConfig(layer_index=0, model_size="1.7B").d_in == 2048
    assert QwenScopeSAEConfig(layer_index=0, model_size="8B").d_in == 4096
    assert QwenScopeSAEConfig(layer_index=0, model_size="27B").d_in == 5120


def test_explicit_d_in_is_preserved():
    # An explicit d_in overrides the registry default (loader still validates).
    cfg = QwenScopeSAEConfig(layer_index=0, model_size="1.7B", d_in=1234)
    assert cfg.d_in == 1234


def test_weights_filename():
    cfg = QwenScopeSAEConfig(layer_index=15)
    assert cfg.weights_filename() == "layer15.sae.pt"
    assert cfg.weights_filename(3) == "layer3.sae.pt"


def test_variant_property():
    assert QwenScopeSAEConfig(layer_index=0, model_size="1.7B").variant == "Base"
    assert QwenScopeSAEConfig(layer_index=0, model_size="27B").variant is None


def test_unknown_model_size_raises():
    with pytest.raises(ValueError, match="Unknown Qwen model_size"):
        QwenScopeSAEConfig(layer_index=0, model_size="999B")


def test_bad_width_for_model_raises():
    with pytest.raises(ValueError, match="width"):
        QwenScopeSAEConfig(layer_index=0, model_size="1.7B", width="64k")


def test_bad_k_raises():
    with pytest.raises(ValueError, match="k=7"):
        QwenScopeSAEConfig(layer_index=0, k=7)


def test_non_resid_post_hook_raises():
    with pytest.raises(ValueError, match="RESID_POST"):
        QwenScopeSAEConfig(layer_index=0, hook_type=HookType.ATTN_OUT)


# --------------------------------------------------------------------------- #
# Group B — cache key
# --------------------------------------------------------------------------- #


def test_cache_key_includes_model_size_and_omits_variant():
    cfg = QwenScopeSAEConfig(layer_index=5, model_size="1.7B", k=50)
    key = _config_to_key(cfg)
    assert "1.7B" in key
    assert "32k" in key
    assert "50" in key
    # variant is no longer part of the key (it's derived from model_size)
    assert "Base" not in key


def test_cache_key_distinguishes_model_size():
    # Two models that would (in the MoE case) share a width must not collide.
    a = _config_to_key(QwenScopeSAEConfig(layer_index=5, model_size="8B"))
    b = _config_to_key(QwenScopeSAEConfig(layer_index=5, model_size="27B"))
    assert a != b


# --------------------------------------------------------------------------- #
# Group C — loader shape inference (fake weights, no download)
# --------------------------------------------------------------------------- #


def _write_fake_sae(tmp_path: Path, d_sae: int, d_in: int) -> Path:
    """Write a fake Qwen-scope .pt with on-disk orientation (W_enc: (d_sae, d_in))."""
    state = {
        "W_enc": torch.randn(d_sae, d_in, dtype=torch.float32),
        "W_dec": torch.randn(d_in, d_sae, dtype=torch.float32),
        "b_enc": torch.randn(d_sae, dtype=torch.float32),
        "b_dec": torch.randn(d_in, dtype=torch.float32),
    }
    path = tmp_path / "layer0.sae.pt"
    torch.save(state, path)
    return path


def test_loader_infers_dims_from_tensor(tmp_path, monkeypatch):
    pt = _write_fake_sae(tmp_path, d_sae=128, d_in=2048)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(pt))

    cfg = QwenScopeSAEConfig(layer_index=0, model_size="1.7B", k=50, dtype="float32", device="cpu")
    sae = _load_qwen_scope_sae(cfg)

    # Dims come from the tensor, not config.d_sae (which is 32768 here).
    assert sae.d_in == 2048
    assert sae.d_sae == 128
    assert tuple(sae.w_enc.shape) == (2048, 128)  # (d_in, d_sae) post-transpose
    assert tuple(sae.w_dec.shape) == (128, 2048)  # (d_sae, d_in) post-transpose


def test_loaded_topksae_l0_equals_k(tmp_path, monkeypatch):
    pt = _write_fake_sae(tmp_path, d_sae=128, d_in=2048)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(pt))

    cfg = QwenScopeSAEConfig(layer_index=0, model_size="1.7B", k=50, dtype="float32", device="cpu")
    sae = _load_qwen_scope_sae(cfg)

    feats, _ = sae(torch.randn(1, 4, 2048, dtype=torch.float32))
    per_token_l0 = (feats != 0).sum(dim=-1)
    assert (per_token_l0 == 50).all()


def test_loader_raises_on_d_in_mismatch(tmp_path, monkeypatch):
    # Disk d_in (999) disagrees with the 1.7B registry d_in (2048).
    pt = _write_fake_sae(tmp_path, d_sae=128, d_in=999)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo_id, filename: str(pt))

    cfg = QwenScopeSAEConfig(layer_index=0, model_size="1.7B", k=50, dtype="float32", device="cpu")
    with pytest.raises(ValueError, match="d_in mismatch"):
        _load_qwen_scope_sae(cfg)


def test_load_sae_caches_instance(tmp_path, monkeypatch):
    pt = _write_fake_sae(tmp_path, d_sae=128, d_in=2048)
    calls = {"n": 0}

    def _fake(repo_id, filename):
        calls["n"] += 1
        return str(pt)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake)

    cfg = QwenScopeSAEConfig(layer_index=0, model_size="1.7B", k=50, dtype="float32", device="cpu")
    first = load_sae(cfg)
    second = load_sae(cfg)
    assert first is second
    assert calls["n"] == 1  # second call served from cache


# --------------------------------------------------------------------------- #
# Group D — canonical qwen-scope source ids (DuckDB ``sae_id`` scheme)
# --------------------------------------------------------------------------- #


def test_qwen_source_id():
    cfg = QwenScopeSAEConfig(layer_index=14, model_size="1.7B")
    assert qwen_source_id(cfg) == "14-qwenscope-1-res-32k"


def test_qwen_source_id_width_follows_config():
    cfg = QwenScopeSAEConfig(layer_index=10, model_size="8B")  # only width: 64k
    assert qwen_source_id(cfg) == "10-qwenscope-1-res-64k"


def test_qwen_source_id_positional_parse_roundtrip():
    """The mutations layer parses sae_id positionally: split('-') → [0]=layer,
    [3]=hook abbrev, [4]=width. The qwen scheme must keep that 5-part shape."""
    cfg = QwenScopeSAEConfig(layer_index=14, model_size="1.7B")
    parts = qwen_source_id(cfg).split("-")
    assert int(parts[0]) == 14
    assert parts[3] == "res"
    assert parts[4] == "32k"
