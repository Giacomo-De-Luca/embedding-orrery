"""Tests for extract_concat_activations on synthetic pooled datasets."""

import pytest
import torch

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.concat_extraction import ConcatExtractionConfig
from interpret.probing.extraction.extract_concat_activations import (
    extract_concat_activations,
)

IDS = ["s0", "s1"]


def _pooled(site: str, keys: dict[int, int], kept: dict[int, list[int]], base: float):
    """Pooled dataset: `keys` maps layer -> column count."""
    activations = {
        (layer, "sae_max"): base + torch.arange(2 * cols, dtype=torch.float32).reshape(2, cols)
        for layer, cols in keys.items()
    }
    return ActivationDataset(
        activations=activations,
        sample_ids=list(IDS),
        metadata={
            "extraction_type": "sae_pooled",
            "sae_site": site,
            "kept_by_layer": kept,
        },
    )


class TestConcat:
    def test_names_spans_and_order(self):
        ds_a = _pooled(
            "resid_post",
            {0: 2, 1: 1},
            {0: [5, 9], 1: [7]},
            base=0.0,
        )
        ds_b = _pooled("mlp_out", {0: 1}, {0: [2]}, base=100.0)
        config = ConcatExtractionConfig(
            name="c",
            source_extractions=["a", "b"],
        )
        out = extract_concat_activations([("a", ds_a), ("b", ds_b)], config)

        matrix = out.activations[(0, "concat")]
        assert matrix.shape == (2, 4)
        assert out.metadata["feature_names"] == [
            "L0_resid_post_f5",
            "L0_resid_post_f9",
            "L1_resid_post_f7",
            "L0_mlp_out_f2",
        ]
        assert out.metadata["concat_spans"] == [
            ("a", 0, "resid_post", 0, 2),
            ("a", 1, "resid_post", 2, 3),
            ("b", 0, "mlp_out", 3, 4),
        ]
        # Column content follows source order.
        assert matrix[:, 3].tolist() == [100.0, 101.0]
        assert out.sample_ids == IDS

    def test_layers_filter(self):
        ds_a = _pooled(
            "resid_post",
            {0: 2, 1: 1},
            {0: [5, 9], 1: [7]},
            base=0.0,
        )
        config = ConcatExtractionConfig(
            name="c",
            source_extractions=["a"],
            layers=[1],
        )
        out = extract_concat_activations([("a", ds_a)], config)
        assert out.activations[(0, "concat")].shape == (2, 1)
        assert out.metadata["feature_names"] == ["L1_resid_post_f7"]

    def test_missing_kept_by_layer_falls_back_to_positional(self):
        ds = _pooled("resid_post", {3: 2}, {}, base=0.0)
        config = ConcatExtractionConfig(name="c", source_extractions=["a"])
        out = extract_concat_activations([("a", ds)], config)
        assert out.metadata["feature_names"] == [
            "L3_resid_post_f0",
            "L3_resid_post_f1",
        ]

    def test_mismatched_sample_ids_raise(self):
        ds_a = _pooled("resid_post", {0: 1}, {0: [1]}, base=0.0)
        ds_b = _pooled("resid_post", {0: 1}, {0: [1]}, base=0.0)
        ds_b.sample_ids = ["other", "ids"]
        config = ConcatExtractionConfig(name="c", source_extractions=["a", "b"])
        with pytest.raises(ValueError, match="sample_ids"):
            extract_concat_activations([("a", ds_a), ("b", ds_b)], config)

    def test_empty_after_filter_raises(self):
        ds = _pooled("resid_post", {0: 1}, {0: [1]}, base=0.0)
        config = ConcatExtractionConfig(
            name="c",
            source_extractions=["a"],
            layers=[99],
        )
        with pytest.raises(ValueError, match="layer filter"):
            extract_concat_activations([("a", ds)], config)

    def test_kept_by_layer_size_mismatch_raises(self):
        ds = _pooled("resid_post", {0: 2}, {0: [1]}, base=0.0)
        config = ConcatExtractionConfig(name="c", source_extractions=["a"])
        with pytest.raises(ValueError, match="kept_by_layer"):
            extract_concat_activations([("a", ds)], config)
