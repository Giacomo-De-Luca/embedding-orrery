"""Config-level tests for the token-level pipeline's extraction types.

Parses the real TREC experiment YAMLs (dispatch, topo order, skip_probes)
and exercises the new config dataclasses' validation + cache normalisation.
"""

from pathlib import Path

import pytest

from interpret.probing.caching import _normalise
from interpret.probing.configs.concat_extraction import ConcatExtractionConfig
from interpret.probing.configs.experiment import ExperimentConfig
from interpret.probing.configs.residual_pooled_extraction import (
    ResidualPooledExtractionConfig,
)
from interpret.probing.configs.sae_pooled_extraction import (
    SAEPooledExtractionConfig,
)
from interpret.probing.configs.token_extraction import (
    TokenLevelExtractionConfig,
)

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "trec_classification"
ALL_YAMLS = [
    "trec_gemma.yaml",
    "trec_qwen.yaml",
    "trec_gemma_smoke.yaml",
    "trec_qwen_smoke.yaml",
]


class TestYamlParsing:
    @pytest.mark.parametrize("yaml_name", ALL_YAMLS)
    def test_parses_and_topo_sorts(self, yaml_name):
        config = ExperimentConfig.from_yaml(EXPERIMENT_DIR / yaml_name)
        order = [e.name for e in config.topo_sorted_extractions()]
        positions = {name: i for i, name in enumerate(order)}
        for ext in config.extractions:
            if isinstance(ext, TokenLevelExtractionConfig):
                assert ext.skip_probes is True
            if isinstance(
                ext,
                (SAEPooledExtractionConfig, ResidualPooledExtractionConfig),
            ):
                assert positions[ext.source_extraction] < positions[ext.name]
            if isinstance(ext, ConcatExtractionConfig):
                for dep in ext.source_extractions:
                    assert positions[dep] < positions[ext.name]

    def test_gemma_yaml_types(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_DIR / "trec_gemma.yaml")
        by_name = {e.name: e for e in config.extractions}
        tokens = by_name["gemma_tokens"]
        assert isinstance(tokens, TokenLevelExtractionConfig)
        assert tokens.sites == ["resid_post", "mlp_out", "attn_out"]
        assert len(tokens.layers) == 34
        assert isinstance(by_name["gemma_sae_resid_max"], SAEPooledExtractionConfig)
        assert by_name["gemma_sae_resid_max"].intermediate_key == "sae_max"
        assert isinstance(by_name["gemma_res_last"], ResidualPooledExtractionConfig)
        assert by_name["gemma_res_last"].intermediate_key == "res_last"

    def test_sidecar_normalisation_round_trip(self):
        config = ExperimentConfig.from_yaml(EXPERIMENT_DIR / "trec_qwen.yaml")
        for ext in config.extractions:
            normalised = _normalise(ext)
            assert isinstance(normalised, dict)
            assert normalised == _normalise(ext)


class TestValidation:
    def test_missing_concat_source_raises(self):
        with pytest.raises(ValueError, match="not\\s+in extractions"):
            ExperimentConfig.from_dict(
                {
                    "name": "x",
                    "output_dir": "/tmp/x",
                    "manifest": {
                        "path": (
                            "interpret.probing.manifests.labeled_text:LabeledTextManifestBuilder"
                        ),
                    },
                    "extractions": [
                        {
                            "name": "c",
                            "type": "concat",
                            "source_extractions": ["missing"],
                        },
                    ],
                },
            )

    def test_qwen_non_resid_sites_raise(self):
        with pytest.raises(ValueError, match="resid_post"):
            TokenLevelExtractionConfig(
                name="t",
                family="qwen",
                layers=[0],
                sites=["mlp_out"],
            )

    def test_qwen_default_checkpoint_filled(self):
        cfg = TokenLevelExtractionConfig(name="t", family="qwen", layers=[0])
        assert cfg.checkpoint == "Qwen/Qwen3-1.7B"

    def test_token_config_requires_layers(self):
        with pytest.raises(ValueError, match="layers"):
            TokenLevelExtractionConfig(name="t", family="gemma", layers=[])

    def test_unknown_site_raises(self):
        with pytest.raises(ValueError, match="Unknown site"):
            SAEPooledExtractionConfig(
                name="s",
                source_extraction="t",
                site="post_attn",
            )

    def test_concat_requires_sources(self):
        with pytest.raises(ValueError, match="source_extractions"):
            ConcatExtractionConfig(name="c", source_extractions=[])
