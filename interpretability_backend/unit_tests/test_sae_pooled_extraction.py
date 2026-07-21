"""Tests for extract_sae_pooled / extract_residual_pooled on synthetic data.

A stub SAE (encode = first-3-columns slice) makes pooled values equal to
hand-computable residual values; `load_sae` / `clear_sae_cache` are
monkeypatched at the extractor module level so no weights are downloaded.
"""

import pytest
import torch

import interpret.probing.extraction.extract_sae_pooled as esp
from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.residual_pooled_extraction import (
    ResidualPooledExtractionConfig,
)
from interpret.probing.configs.sae_pooled_extraction import (
    SAEPooledExtractionConfig,
)
from interpret.probing.extraction.extract_residual_pooled import (
    extract_residual_pooled,
)
from interpret.probing.extraction.extract_sae_pooled import extract_sae_pooled
from interpret.sae.activation_store import max_pool_feature_acts

D_IN, D_SAE = 4, 3


class StubSAE:
    """encode(x) = x[:, :3] — feature values mirror the first residual dims."""

    w_dec = torch.zeros(D_SAE, D_IN)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :D_SAE].clone()


@pytest.fixture
def patched_sae(monkeypatch):
    calls = {"load": [], "clear": 0}

    def fake_load(config):
        calls["load"].append(config)
        return StubSAE()

    def fake_clear():
        calls["clear"] += 1

    monkeypatch.setattr(esp, "load_sae", fake_load)
    monkeypatch.setattr(esp, "clear_sae_cache", fake_clear)
    return calls


def _token_source(prepends_bos: bool = True, family: str = "gemma") -> ActivationDataset:
    """3 samples, lengths [3, 2, 1] (first position = BOS when applicable).

    BOS rows carry huge values (the activation sink); sample 2 is BOS-only
    and exercises the degenerate-range fallback.
    """
    residual = torch.tensor(
        [
            # sample 0: BOS + 2 tokens
            [100.0, 100.0, 100.0, 0.0],
            [1.0, -5.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 0.0],
            # sample 1: BOS + 1 token
            [100.0, 100.0, 100.0, 0.0],
            [7.0, -1.0, 0.0, 0.0],
            # sample 2: BOS only
            [2.0, 0.0, 5.0, 0.0],
        ],
    )
    return ActivationDataset(
        activations={(0, "resid_post"): residual, (2, "resid_post"): residual * 2},
        targets=torch.empty(0),
        sample_ids=["s0", "s1", "s2"],
        metadata={
            "extraction_type": "token_residuals",
            "token_level": True,
            "family": family,
            "token_offsets": [0, 3, 5, 6],
            "prepends_bos": prepends_bos,
            "storage_dtype": "float32",
        },
    )


class TestSAEPooledMax:
    def test_max_excludes_bos_with_single_token_fallback(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="max",
            layers=[0],
        )
        ds = extract_sae_pooled(_token_source(), config)
        pooled = ds.activations[(0, "sae_max")]
        # Sample 0: max over non-BOS rows -> [3, 2, 0]; feature 2 (only >0
        # via the BOS rows and sample 2) survives thanks to sample 2's
        # BOS-only fallback.
        assert pooled.shape[0] == 3
        kept = ds.metadata["kept_by_layer"][0]
        dense = torch.zeros(3, D_SAE)
        dense[:, kept] = pooled
        assert dense[0].tolist() == [3.0, 2.0, 0.0]
        assert dense[1].tolist() == [7.0, -1.0, 0.0]
        assert dense[2].tolist() == [2.0, 0.0, 5.0]

    def test_max_keeps_bos_when_not_prepended(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="max",
            layers=[0],
            drop_dead_features=False,
        )
        ds = extract_sae_pooled(_token_source(prepends_bos=False), config)
        pooled = ds.activations[(0, "sae_max")]
        assert pooled[0].tolist() == [100.0, 100.0, 100.0]

    def test_chunked_encoding_matches_unchunked(self, patched_sae):
        source = _token_source()
        big = extract_sae_pooled(
            source,
            SAEPooledExtractionConfig(
                name="a",
                source_extraction="t",
                pooling="max",
                layers=[0],
                drop_dead_features=False,
            ),
        )
        small = extract_sae_pooled(
            source,
            SAEPooledExtractionConfig(
                name="b",
                source_extraction="t",
                pooling="max",
                layers=[0],
                drop_dead_features=False,
                batch_size_tokens=2,
            ),
        )
        assert torch.equal(
            big.activations[(0, "sae_max")],
            small.activations[(0, "sae_max")],
        )

    def test_matches_production_max_pool_semantics(self, patched_sae):
        """Dense pooling equals max_pool_feature_acts on a single sample."""
        source = _token_source()
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="max",
            layers=[0],
            drop_dead_features=False,
        )
        pooled = extract_sae_pooled(source, config).activations[(0, "sae_max")]

        sample0_tokens = source.activations[(0, "resid_post")][0:3]
        record_acts = StubSAE().encode(sample0_tokens).unsqueeze(0)  # (1, T, d)
        record = type("R", (), {"feature_acts": record_acts})()
        pairs = max_pool_feature_acts(record, start=1)  # skip BOS

        dense_positive = {i: float(v) for i, v in enumerate(pooled[0]) if v > 0}
        assert dict(pairs) == dense_positive


class TestSAEPooledLastAndFiltering:
    def test_last_picks_final_token(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="last",
            layers=[0],
            drop_dead_features=False,
        )
        ds = extract_sae_pooled(_token_source(), config)
        pooled = ds.activations[(0, "sae_last")]
        assert pooled[0].tolist() == [3.0, 2.0, 0.0]
        assert pooled[1].tolist() == [7.0, -1.0, 0.0]
        assert pooled[2].tolist() == [2.0, 0.0, 5.0]
        assert ds.metadata["pooling"] == "last"

    def test_min_active_samples_filters(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="max",
            layers=[0],
            min_active_samples=2,
        )
        ds = extract_sae_pooled(_token_source(), config)
        # Post-BOS-mask activity counts: f0 in 3 samples, f1 in 1, f2 in 1.
        assert ds.metadata["kept_by_layer"][0] == [0]
        assert ds.activations[(0, "sae_max")].shape == (3, 1)

    def test_all_layers_default_and_clear_cache_per_layer(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            pooling="max",
        )
        ds = extract_sae_pooled(_token_source(), config)
        assert sorted(k[0] for k in ds.activations) == [0, 2]
        assert patched_sae["clear"] == 2
        assert all(c.dtype == "float32" for c in patched_sae["load"])

    def test_qwen_requires_resid_post(self, patched_sae):
        config = SAEPooledExtractionConfig(
            name="p",
            source_extraction="t",
            site="mlp_out",
            pooling="max",
        )
        with pytest.raises(ValueError, match="resid_post"):
            extract_sae_pooled(_token_source(family="qwen"), config)

    def test_non_token_level_source_raises(self, patched_sae):
        source = ActivationDataset(
            activations={(0, "resid_post"): torch.zeros(3, D_IN)},
            sample_ids=["a", "b", "c"],
            metadata={"extraction_type": "gemma"},
        )
        config = SAEPooledExtractionConfig(name="p", source_extraction="t")
        with pytest.raises(ValueError, match="token_level"):
            extract_sae_pooled(source, config)


class TestResidualPooled:
    def test_last(self):
        config = ResidualPooledExtractionConfig(
            name="r",
            source_extraction="t",
            pooling="last",
            layers=[0],
        )
        ds = extract_residual_pooled(_token_source(), config)
        pooled = ds.activations[(0, "res_last")]
        assert pooled.shape == (3, D_IN)
        assert pooled[0].tolist() == [3.0, 2.0, 0.0, 0.0]

    def test_max_excludes_bos(self):
        config = ResidualPooledExtractionConfig(
            name="r",
            source_extraction="t",
            pooling="max",
            layers=[0],
        )
        pooled = extract_residual_pooled(_token_source(), config).activations[(0, "res_max")]
        assert pooled[0].tolist() == [3.0, 2.0, 0.0, 0.0]
        assert pooled[2].tolist() == [2.0, 0.0, 5.0, 0.0]  # BOS-only fallback

    def test_mean_excludes_bos(self):
        config = ResidualPooledExtractionConfig(
            name="r",
            source_extraction="t",
            pooling="mean",
            layers=[0],
        )
        pooled = extract_residual_pooled(_token_source(), config).activations[(0, "res_mean")]
        assert pooled[0].tolist() == [2.0, -1.5, 0.0, 0.0]
