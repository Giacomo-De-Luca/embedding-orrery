"""Tests for extract_token_residuals — fake Gemma/Qwen wrappers, no models."""

from contextlib import contextmanager

import pytest
import torch

from interpret.probing.configs.token_extraction import (
    TokenLevelExtractionConfig,
)
from interpret.probing.extraction.extract_token_residuals import (
    extract_token_residuals,
)
from interpret.sae.sae_config import HookType

HIDDEN = 8


def _fake_activation(layer: int, site_code: int, length: int) -> torch.Tensor:
    """[1, T, H] tensor whose values encode (layer, site, position)."""
    base = float(layer * 1000 + site_code * 100)
    positions = torch.arange(length, dtype=torch.float32).view(1, length, 1)
    return base + positions.expand(1, length, HIDDEN).clone()


class FakeGemmaWrapper:
    """Mimics GemmaPytorchInference's cache API. Token count = word count + 1 (BOS).

    The extractor must call `generate_from_template` (raw pass-through);
    the chat-wrapping `generate` is deliberately absent so a regression
    back to it fails loudly.
    """

    prepends_bos = True
    _SITE_CODES = {"pre_attn": 0, "attn_out": 1, "post_attn": 2, "mlp_out": 3, "post_mlp": 4}

    def __init__(self):
        self.configured = None
        self.cleared = False
        self._current: dict | None = None

    def configure_cache(self, layers, intermediates, prefill, last):
        self.configured = (set(layers), set(intermediates), prefill, last)

    def reset_prefill_cache(self):
        self._current = None

    def generate_from_template(self, formatted_prompt, output_len):
        layers, intermediates, _, _ = self.configured
        length = len(formatted_prompt.split()) + 1  # BOS
        self._current = {
            layer: {
                inter: _fake_activation(layer, self._SITE_CODES[inter], length)
                for inter in intermediates
            }
            for layer in layers
        }

    def get_cached_activations(self):
        return {"prefill": self._current}

    def clear_cache(self):
        self.cleared = True


class FakeQwenWrapper:
    """Mimics Qwen3Inference: cache_activations context, no BOS, HookType keys."""

    prepends_bos = False

    def __init__(self):
        self._layers = None
        self._hooks = None
        self._current = None

    @contextmanager
    def cache_activations(self, layers, hook_types, prefill_only=True):
        self._layers, self._hooks = set(layers), set(hook_types)
        yield lambda: self._current

    def generate_from_template(self, prompt, output_len, add_bos=True):
        length = len(prompt.split())
        self._current = {
            layer: {hook: _fake_activation(layer, 9, length) for hook in self._hooks}
            for layer in self._layers
        }


SAMPLES = ["one", "two words", "three word prompt"]


class TestGemmaPath:
    def test_ragged_layout_and_metadata(self):
        config = TokenLevelExtractionConfig(
            name="t",
            family="gemma",
            layers=[0, 2],
            sites=["resid_post", "attn_out"],
        )
        wrapper = FakeGemmaWrapper()
        ds = extract_token_residuals(config, SAMPLES, wrapper)

        # Gemma lengths: words + BOS -> 2, 3, 4.
        assert ds.metadata["token_offsets"] == [0, 2, 5, 9]
        assert ds.metadata["token_level"] is True
        assert ds.metadata["prepends_bos"] is True
        assert ds.metadata["n_tokens"] == 9
        assert ds.sample_ids == SAMPLES
        assert sorted(ds.activations) == [
            (0, "attn_out"),
            (0, "resid_post"),
            (2, "attn_out"),
            (2, "resid_post"),
        ]
        for tensor in ds.activations.values():
            assert tensor.shape == (9, HIDDEN)
            assert tensor.dtype == torch.bfloat16
        assert wrapper.cleared

    def test_site_mapping_reads_post_mlp_for_resid_post(self):
        config = TokenLevelExtractionConfig(
            name="t",
            family="gemma",
            layers=[1],
            sites=["resid_post"],
        )
        ds = extract_token_residuals(config, ["hi"], FakeGemmaWrapper())
        # post_mlp site code is 4 -> base 1000 + 400.
        assert float(ds.activations[(1, "resid_post")][0, 0]) == 1400.0

    def test_float32_storage(self):
        config = TokenLevelExtractionConfig(
            name="t",
            family="gemma",
            layers=[0],
            sites=["mlp_out"],
            storage_dtype="float32",
        )
        ds = extract_token_residuals(config, ["hi"], FakeGemmaWrapper())
        assert ds.activations[(0, "mlp_out")].dtype == torch.float32

    def test_wrong_wrapper_family_raises(self):
        config = TokenLevelExtractionConfig(
            name="t",
            family="gemma",
            layers=[0],
        )
        with pytest.raises(TypeError, match="configure_cache"):
            extract_token_residuals(config, SAMPLES, FakeQwenWrapper())


class TestQwenPath:
    def test_ragged_layout_no_bos(self):
        config = TokenLevelExtractionConfig(
            name="t",
            family="qwen",
            layers=[0, 5],
            sites=["resid_post"],
        )
        ds = extract_token_residuals(config, SAMPLES, FakeQwenWrapper())
        # Qwen lengths: word count (no BOS) -> 1, 2, 3.
        assert ds.metadata["token_offsets"] == [0, 1, 3, 6]
        assert ds.metadata["prepends_bos"] is False
        assert sorted(ds.activations) == [(0, "resid_post"), (5, "resid_post")]
        for tensor in ds.activations.values():
            assert tensor.shape == (6, HIDDEN)
            assert tensor.dtype == torch.bfloat16

    def test_string_hook_keys_accepted(self):
        """Qwen cache keys may be HookType members OR their string values."""

        class StringKeyQwen(FakeQwenWrapper):
            def generate_from_template(self, prompt, output_len, add_bos=True):
                super().generate_from_template(prompt, output_len, add_bos)
                self._current = {
                    layer: {HookType.RESID_POST.value: next(iter(hooks.values()))}
                    for layer, hooks in self._current.items()
                }

        config = TokenLevelExtractionConfig(
            name="t",
            family="qwen",
            layers=[0],
        )
        ds = extract_token_residuals(config, ["hi there"], StringKeyQwen())
        assert ds.activations[(0, "resid_post")].shape == (2, HIDDEN)

    def test_wrong_wrapper_family_raises(self):
        config = TokenLevelExtractionConfig(name="t", family="qwen", layers=[0])
        with pytest.raises(TypeError, match="no configure_cache"):
            extract_token_residuals(config, SAMPLES, FakeGemmaWrapper())

    def test_empty_samples_raise(self):
        config = TokenLevelExtractionConfig(name="t", family="qwen", layers=[0])
        with pytest.raises(ValueError, match="empty"):
            extract_token_residuals(config, [], FakeQwenWrapper())
