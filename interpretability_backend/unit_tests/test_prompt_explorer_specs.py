"""Tests for multi-SAE spec support in PromptExplorer.

Pure dataclass/logic tests — no GPU, model weights, or SAE downloads.
Covers the pieces added for one-forward-pass multi-SAE prompt activations:
``SaeSpec`` (layer, width and hook site; a plain ``(layer, width)`` pair
means the residual stream), ``PromptExplorerConfig.effective_saes``, the
``_store_read_plan`` helper (which must mirror HookManager's write rule:
single SAE at a site records under ``sae_id=""``, co-attached SAEs record
under their ``identity()`` slug), and ``PromptResult.layers`` keyed by
``SaeSpec``, so a residual and an MLP SAE at the same layer and width stay
separate entries.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from interpret.sae.exploration import prompt_explorer
from interpret.sae.exploration.prompt_explorer import (
    LayerResult,
    PromptExplorer,
    PromptExplorerConfig,
    PromptResult,
    SaeSpec,
    _store_read_plan,
)
from interpret.sae.sae_config import GemmaScopeSAEConfig, HookType


def _cfg(layer: int, width: str, hook_type: HookType = HookType.RESID_POST) -> GemmaScopeSAEConfig:
    return GemmaScopeSAEConfig(layer_index=layer, width=width, hook_type=hook_type, device="cpu")


def _layer_result(layer: int, width: str, hook_type: HookType = HookType.RESID_POST) -> LayerResult:
    return LayerResult(
        layer=layer, width=width, tokens=[], feature_acts=torch.zeros(1, 4), hook_type=hook_type
    )


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

    def test_residual_and_mlp_at_one_layer_are_separate_sites(self):
        plan = _store_read_plan([_cfg(9, "16k"), _cfg(9, "16k", HookType.MLP_OUT)])
        assert [(c.hook_type, sid) for c, sid in plan] == [
            (HookType.RESID_POST, ""),
            (HookType.MLP_OUT, ""),
        ]


class TestSaeSpec:
    def test_hook_site_defaults_to_the_residual_stream(self):
        assert SaeSpec(9, "16k").hook_type is HookType.RESID_POST

    def test_of_reads_the_config_identity(self):
        cfg = _cfg(9, "65k", HookType.MLP_OUT)
        assert SaeSpec.of(cfg) == SaeSpec(9, "65k", HookType.MLP_OUT)

    def test_parse_takes_the_hook_site_as_a_value(self):
        assert SaeSpec.parse(9, "16k", "mlp_out") == SaeSpec(9, "16k", HookType.MLP_OUT)
        assert SaeSpec.parse(9, "16k", HookType.ATTN_OUT) == SaeSpec(9, "16k", HookType.ATTN_OUT)
        assert SaeSpec.parse(9, "16k") == SaeSpec(9, "16k")

    def test_parse_rejects_unknown_sites(self):
        with pytest.raises(ValueError, match="Unknown hook_type 'bogus'"):
            SaeSpec.parse(9, "16k", "bogus")

    def test_label_names_non_residual_sites_only(self):
        assert SaeSpec(9, "16k").label() == "9/16k"
        assert SaeSpec(9, "16k", HookType.MLP_OUT).label() == "9/16k/mlp_out"

    def test_sort_key_orders_layer_then_numeric_width_then_site(self):
        specs = [
            SaeSpec(22, "16k"),
            SaeSpec(9, "262k"),
            SaeSpec(9, "16k", HookType.ATTN_OUT),
            SaeSpec(9, "65k"),
            SaeSpec(9, "16k", HookType.MLP_OUT),
            SaeSpec(9, "16k"),
        ]
        assert sorted(specs, key=SaeSpec.sort_key) == [
            SaeSpec(9, "16k"),
            SaeSpec(9, "16k", HookType.MLP_OUT),
            SaeSpec(9, "16k", HookType.ATTN_OUT),
            SaeSpec(9, "65k"),
            SaeSpec(9, "262k"),
            SaeSpec(22, "16k"),
        ]


class TestEffectiveSaes:
    def test_derived_from_layers_and_width(self):
        config = PromptExplorerConfig(wrapper=MagicMock(), layers=[9, 22], width="65k")
        assert config.effective_saes() == [SaeSpec(9, "65k"), SaeSpec(22, "65k")]

    def test_explicit_saes_take_precedence(self):
        config = PromptExplorerConfig(
            wrapper=MagicMock(),
            layers=[17],
            width="16k",
            saes=[(9, "16k"), (9, "65k")],
        )
        assert config.effective_saes() == [SaeSpec(9, "16k"), SaeSpec(9, "65k")]

    def test_plain_pairs_mean_the_residual_stream(self):
        config = PromptExplorerConfig(
            wrapper=MagicMock(),
            saes=[(9, "16k"), (9, "16k", "mlp_out"), SaeSpec(9, "16k", HookType.ATTN_OUT)],
        )
        assert [s.hook_type for s in config.effective_saes()] == [
            HookType.RESID_POST,
            HookType.MLP_OUT,
            HookType.ATTN_OUT,
        ]


class TestLayerResult:
    def test_repr_names_the_hook_site(self):
        result = _layer_result(9, "16k", HookType.MLP_OUT)
        assert result.spec == SaeSpec(9, "16k", HookType.MLP_OUT)
        assert "sae=9/16k/mlp_out" in repr(result)


class TestPromptResultLayerAccess:
    def test_layer_returns_unique_match(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={
                SaeSpec(9, "16k"): _layer_result(9, "16k"),
                SaeSpec(22, "16k"): _layer_result(22, "16k"),
            },
        )
        assert result.layer(9).width == "16k"
        assert result.layer(22).layer == 22

    def test_layer_ambiguous_widths_raises(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={
                SaeSpec(9, "16k"): _layer_result(9, "16k"),
                SaeSpec(9, "65k"): _layer_result(9, "65k"),
            },
        )
        with pytest.raises(ValueError, match="16k"):
            result.layer(9)

    def test_layer_ambiguous_sites_raises(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={
                SaeSpec(9, "16k"): _layer_result(9, "16k"),
                SaeSpec(9, "16k", HookType.MLP_OUT): _layer_result(9, "16k", HookType.MLP_OUT),
            },
        )
        with pytest.raises(ValueError, match="mlp_out"):
            result.layer(9)

    def test_layer_missing_raises_keyerror(self):
        result = PromptResult(
            prompt="p",
            token_strings=[],
            layers={SaeSpec(9, "16k"): _layer_result(9, "16k")},
        )
        with pytest.raises(KeyError):
            result.layer(3)


class TestBuildSaeConfig:
    def test_default_config_hooks_the_requested_site(self):
        explorer = PromptExplorer(PromptExplorerConfig(wrapper=MagicMock(device="cpu")))
        cfg = explorer._build_sae_config(SaeSpec(9, "16k", HookType.MLP_OUT))
        assert (cfg.layer_index, cfg.width, cfg.hook_type) == (9, "16k", HookType.MLP_OUT)

    def test_factory_receives_the_spec(self):
        seen: list[SaeSpec] = []

        def factory(spec: SaeSpec) -> GemmaScopeSAEConfig:
            seen.append(spec)
            return _cfg(spec.layer, spec.width, spec.hook_type)

        explorer = PromptExplorer(
            PromptExplorerConfig(wrapper=MagicMock(device="cpu"), sae_config_factory=factory)
        )
        explorer._build_sae_config(SaeSpec(9, "16k", HookType.ATTN_OUT))
        assert seen == [SaeSpec(9, "16k", HookType.ATTN_OUT)]


def _prefill_acts(feature: int, seq_len: int = 2, d_sae: int = 4) -> torch.Tensor:
    """(batch, seq_len, d_sae) activations where one feature fires on every token."""
    acts = torch.zeros(1, seq_len, d_sae)
    acts[0, :, feature] = 1.0
    return acts


class _FakeHookManager:
    """HookManager stand-in: one prefill record per hook site, no SAE weights.

    The residual SAE fires feature 1 and the MLP SAE feature 2, so a result
    shows which site it was read from.
    """

    FEATURE_BY_SITE = {HookType.RESID_POST: 1, HookType.MLP_OUT: 2}

    def add_sae(self, config: GemmaScopeSAEConfig) -> None:
        pass

    @contextmanager
    def session(self, layers):
        def prefill(layer: int, hook_type: HookType, sae_id: str) -> SimpleNamespace:
            return SimpleNamespace(feature_acts=_prefill_acts(self.FEATURE_BY_SITE[hook_type]))

        yield SimpleNamespace(prefill=prefill)


class TestRunPromptKeysBySite:
    def test_residual_and_mlp_at_one_layer_and_width_stay_separate(self):
        wrapper = MagicMock(device="cpu")
        wrapper.format_prompt.return_value = "formatted"
        wrapper.token_strings.return_value = ["<bos>", "hi"]
        specs = [SaeSpec(9, "16k"), SaeSpec(9, "16k", HookType.MLP_OUT)]
        explorer = PromptExplorer(
            PromptExplorerConfig(wrapper=wrapper, saes=specs, skip_labels=True, top_k=1)
        )

        with patch.object(prompt_explorer, "HookManager", _FakeHookManager):
            result = explorer.run_prompt("hi", output_len=0)

        assert set(result.layers) == set(specs)
        residual = result.layers[SaeSpec(9, "16k")]
        mlp = result.layers[SaeSpec(9, "16k", HookType.MLP_OUT)]
        assert (residual.hook_type, mlp.hook_type) == (HookType.RESID_POST, HookType.MLP_OUT)
        assert [f.index for f in residual.tokens[0].features] == [1]
        assert [f.index for f in mlp.tokens[0].features] == [2]
