"""Tests for LabeledTextManifestBuilder — synthetic files + real trec.tsv."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interpret.probing.manifests.labeled_text import LabeledTextManifestBuilder

TREC_PATH = Path(__file__).parents[1] / "resources" / "datasets" / "SAE" / "trec.tsv"


@pytest.fixture
def tsv_path(tmp_path):
    df = pd.DataFrame(
        {
            "text": ["alpha", "beta", "alpha", "gamma", "delta", "epsilon"],
            "coarse_label": [0, 1, 2, 1, 0, 1],
            "cat_label": ["yes", "no", "yes", "no", "maybe", "no"],
            "split": ["train", "train", "train", "test", "test", "train"],
        },
    )
    path = tmp_path / "data.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


class TestSyntheticData:
    def test_dedupe_keeps_first(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
        )
        assert builder.samples == ["alpha", "beta", "gamma", "delta", "epsilon"]
        samples, values = builder.get_rated_samples("labeled_text", "coarse_label")
        # "alpha" keeps its FIRST row's label (0, not the duplicate's 2).
        assert dict(zip(samples, values.tolist(), strict=True))["alpha"] == 0

    def test_no_dedupe(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
            dedupe=False,
        )
        assert len(builder.samples) == 6

    def test_split_filter(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
            split_column="split",
            splits=["test"],
        )
        assert builder.samples == ["gamma", "delta"]

    def test_splits_without_split_column_raises(self, tsv_path):
        with pytest.raises(ValueError, match="split_column"):
            LabeledTextManifestBuilder(
                tsv_path,
                target_columns=["coarse_label"],
                splits=["train"],
            )

    def test_limit_after_dedupe(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
            limit=3,
        )
        assert builder.samples == ["alpha", "beta", "gamma"]

    def test_int_targets_pass_through_as_int64(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
        )
        _, values = builder.get_rated_samples("labeled_text", "coarse_label")
        assert values.dtype == np.int64
        assert "coarse_label" not in builder.target_label_maps

    def test_categorical_targets_encoded_alphabetically(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["cat_label"],
        )
        assert builder.target_label_maps["cat_label"] == {
            "maybe": 0,
            "no": 1,
            "yes": 2,
        }
        samples, values = builder.get_rated_samples("labeled_text", "cat_label")
        assert values.dtype == np.int64
        assert dict(zip(samples, values.tolist(), strict=True))["delta"] == 0

    def test_min_class_count_filters_only_that_target(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label", "cat_label"],
            min_class_count={"coarse_label": 2},
        )
        # After dedupe: coarse_label counts are {0: 2, 1: 3} (class 2 gone
        # with the duplicate row) — nothing below 2, all 5 rows survive...
        samples, _ = builder.get_rated_samples("labeled_text", "coarse_label")
        assert len(samples) == 5
        # ...but with the bar at 3 only class 1 survives.
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label", "cat_label"],
            min_class_count={"coarse_label": 3},
        )
        samples, values = builder.get_rated_samples("labeled_text", "coarse_label")
        assert set(values.tolist()) == {1}
        # The other target is untouched.
        samples, _ = builder.get_rated_samples("labeled_text", "cat_label")
        assert len(samples) == 5

    def test_min_class_count_unknown_column_raises(self, tsv_path):
        with pytest.raises(ValueError, match="min_class_count"):
            LabeledTextManifestBuilder(
                tsv_path,
                target_columns=["coarse_label"],
                min_class_count={"nope": 2},
            )

    def test_wrong_source_raises(self, tsv_path):
        builder = LabeledTextManifestBuilder(
            tsv_path,
            target_columns=["coarse_label"],
            source_name="trec",
        )
        with pytest.raises(ValueError, match="Unknown source"):
            builder.get_rated_samples("labeled_text", "coarse_label")
        builder.get_rated_samples("trec", "coarse_label")

    def test_missing_columns_raise(self, tsv_path):
        with pytest.raises(ValueError, match="missing"):
            LabeledTextManifestBuilder(tsv_path, target_columns=["nope"])
        with pytest.raises(ValueError, match="target_columns"):
            LabeledTextManifestBuilder(tsv_path, target_columns=[])

    def test_csv_and_parquet_loaders(self, tmp_path, tsv_path):
        df = pd.read_csv(tsv_path, sep="\t")
        csv_path = tmp_path / "data.csv"
        parquet_path = tmp_path / "data.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        for path in (csv_path, parquet_path):
            builder = LabeledTextManifestBuilder(
                path,
                target_columns=["coarse_label"],
            )
            assert len(builder.samples) == 5


@pytest.mark.skipif(not TREC_PATH.exists(), reason="trec.tsv not present")
class TestRealTrec:
    def test_trec_shape_and_targets(self):
        builder = LabeledTextManifestBuilder(
            TREC_PATH,
            target_columns=["coarse_label", "fine_label"],
            min_class_count={"fine_label": 5},
        )
        # 5952 rows minus 81 duplicate texts.
        assert len(builder.samples) == 5871
        assert len(set(builder.samples)) == len(builder.samples)

        _, coarse = builder.get_rated_samples("labeled_text", "coarse_label")
        assert coarse.dtype == np.int64
        assert set(coarse.tolist()) == set(range(6))

        _, fine = builder.get_rated_samples("labeled_text", "fine_label")
        counts = pd.Series(fine).value_counts()
        assert counts.min() >= 5  # StratifiedKFold(5)-safe
