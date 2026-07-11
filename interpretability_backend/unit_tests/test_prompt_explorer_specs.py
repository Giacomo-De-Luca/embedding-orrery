"""Tests for multi-(layer, width) SAE spec support in PromptExplorer.

Pure dataclass/logic tests — no GPU, model weights, or SAE downloads.
Covers the pieces added for one-forward-pass multi-SAE prompt activations:
``PromptExplorerConfig.effective_saes``, the ``_store_read_plan`` helper
(which must mirror HookManager's write rule: single SAE at a site records
under ``sae_id=""``, co-attached SAEs record under their ``identity()``
slug), and the ``PromptResult.layers`` re-keying by ``(layer, width)``.
"""

from unittest.mock import MagicMock

import pytest
import torch

from interpret.sae.exploration.prompt_explorer import (
    LayerResult,
    PromptExplorerConfig,
    PromptResult,
    _store_read_plan,
)
from interpret.sae.sae_config import GemmaScopeSAEConfig


def _cfg(layer: int, width: str) -> GemmaScopeSAEConfig:
    return GemmaScopeSAEConfig(layer_index=layer, width=width, device="cpu")


def _layer_result(layer: int, width: str) -> LayerResult:
    return LayerResult(layer=layer, width=width, tokens=[], feature_acts=torch.zeros(1, 4))


class TestStoreReadPlan:
    def test_single_sae_per_site_reads_empty_sae_id(self):
        plan = _store_read_plan([_cfg(9, "16k"), _cfg(22, "16k")])
        assert [(c.layer_index, sid) for c, sid in plan] == [(9, ""), (22, "")]

    def test_shared_site_reads_identity_slugs(self):
        configs = [_cfg(9, "16k"), _cfg(9, "65k"), _cfg(22, "16k")]
        plan = _store_read_plan(configs)
        by_key = {(c.layer_index, c.width): sid for c, sid in plan}
        # Shared (layer 9, RESID_POST) site → each reads its identity slug
        assert by_key[(9, "16k")] == configs[0].identity()
        assert by_key[(9, "65k")] == configs[1].identity()
        assert by_key[(9, "16k")] != by_key[(9, "65k")]
        assert by_key[(9, "16k")] != ""
        # Unshared site keeps the legacy fast path
        assert by_key[(22, "16k")] == ""

    def test_preserves_config_order(self):
        configs = [_cfg(22, "16k"), _cfg(9, "65k"), _cfg(9, "16k")]
        plan = _store_read_plan(configs)
        assert [c for c, _ in plan] == configs


class TestEffectiveSaes:
    def test_derived_from_layers_and_width(self):
        config = PromptExplorerConfig(wrapper=MagicMock(), layers=[9, 22], width="65k")
        assert config.effective_saes() == [(9, "65k"), (22, "65k")]

    def test_explicit_saes_take_precedence(self):
        config = PromptExplorerConfig(
            wrapper=MagicMock(),
            layers=[17],
            width="16k",
            saes=[(9, "16k"), (9, "65k")],
        )
        assert config.effective_saes() == [(9, "16k"), (9, "65k")]


class TestPromptResultLayerAccess:
    def test_layer_returns_unique_match(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={
                (9, "16k"): _layer_result(9, "16k"),
                (22, "16k"): _layer_result(22, "16k"),
            },
        )
        assert result.layer(9).width == "16k"
        assert result.layer(22).layer == 22

    def test_layer_ambiguous_widths_raises(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={
                (9, "16k"): _layer_result(9, "16k"),
                (9, "65k"): _layer_result(9, "65k"),
            },
        )
        with pytest.raises(ValueError, match="16k"):
            result.layer(9)

    def test_layer_missing_raises_keyerror(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={(9, "16k"): _layer_result(9, "16k")},
        )
        with pytest.raises(KeyError):
            result.layer(3)
