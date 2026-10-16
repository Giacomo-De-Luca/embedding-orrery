"""Categorical probe targets (backend/services/probe_targets.py).

Covers label canonicalisation (must match how the browser renders the same
metadata value — String(JSON.parse(..)) — so predicted labels line up with
the target field's categories), class ordering, the min-class threshold,
and the binary/multiclass task split.
"""

import math

import numpy as np
import pytest

from backend.services.probe_targets import (
    CategoricalLabel,
    CategoricalTarget,
    ProbeTargetError,
    class_count_threshold,
)
from backend.services.probing_types import (
    CLASSIFICATION_KINDS,
    MAX_PROBE_CLASSES,
    PROBE_KINDS,
    ProbeConfig,
    ProbeTask,
    derive_probe_task,
    score_field_names,
)

# Real class counts from the `combined` collection (combined.csv).
CATEGORY_COUNTS = {
    "EX:OXYMORON PARADOX": 276,
    "ANALOGY": 261,
    "RHETORICAL QUESTION": 221,
    "OTHER": 196,
    "EX:CONTEXT SHIFT": 185,
    "IM:FALSE ASSERTION": 117,
    "EUPHEMISM": 84,
    "HYPERBOLE": 81,
}
SOURCE_COUNTS = {
    "[SENTIPOLC_16]": 7538,
    "[IRONITA-HSC]": 1740,
    "[TWITTIRO]": 1034,
    "[IRONITA-TW-SENTIPOLC, SENTIPOLC_16]": 877,
    "[Multipico-it-0.75Majority]": 770,
    "[IRONITA-TW-BS, SENTIPOLC_16]": 699,
    "[IRONITA-TW-BS]": 541,
    "[IRONITA-TW-SPINO]": 391,
    "[IRONITA-TWITA]": 223,
    "[IRONITA-TW-SENTIPOLC, SENTIPOLC_16, TWITTIRO]": 191,
    "[IRONITA-TW-BS, TWITTIRO]": 114,
    "[IRONITA-TW-BS, SENTIPOLC_16, TWITTIRO]": 29,
    "[SENTIPOLC_16, TWITTIRO]": 26,
    "[IRONITA-TW-SPINO, TWITTIRO]": 24,
    "[IRONITA-TW-SENTIPOLC]": 13,
    "[IRONITA-TW-SENTIPOLC, TWITTIRO]": 3,
}


def _labels(counts: dict[str, int], n_missing: int = 0) -> list[str | None]:
    labels: list[str | None] = []
    for label, count in counts.items():
        labels.extend([label] * count)
    labels.extend([None] * n_missing)
    return labels


# ------------------------------------------------------------------
# Label canonicalisation
# ------------------------------------------------------------------


class TestCanonicalLabel:
    @pytest.mark.parametrize(
        ("text", "json_type", "expected"),
        [
            ("3.0", "DOUBLE", "3"),  # JS: String(JSON.parse("3.0")) === "3"
            ("-0.0", "DOUBLE", "0"),
            ("2.5", "DOUBLE", "2.5"),
            ("3", "BIGINT", "3"),
            ("18446744073709551615", "UBIGINT", "18446744073709551615"),
            ("03", "VARCHAR", "03"),  # strings stay verbatim
            ("3.0", "VARCHAR", "3.0"),
            ("safe", "VARCHAR", "safe"),
            ("true", "BOOLEAN", "true"),
            ("", "VARCHAR", None),  # the browser treats "" as Missing
            ("[1,2]", "ARRAY", None),
            ('{"a":1}', "OBJECT", None),
            (None, None, None),
            (None, "NULL", None),
        ],
    )
    def test_canonical(self, text, json_type, expected):
        assert CategoricalLabel.canonical(text, json_type) == expected

    @pytest.mark.parametrize("value", ["0", "-3", "2.5", "+1", ".5", "1e3", "1E-2"])
    def test_numeric_accepts_plain_numbers(self, value):
        assert CategoricalLabel.is_numeric(value)

    @pytest.mark.parametrize("value", ["1_000", "nan", "inf", "-inf", "0x1A", "", " ", "1 ", "abc"])
    def test_numeric_rejects_lenient_parses(self, value):
        assert not CategoricalLabel.is_numeric(value)

    def test_numeric_labels_sort_numerically(self):
        assert CategoricalLabel.sort(["10", "2", "1"]) == ["1", "2", "10"]

    def test_text_labels_sort_by_code_point(self):
        assert CategoricalLabel.sort(["b", "B", "a", "10", "2"]) == ["10", "2", "B", "a", "b"]


# ------------------------------------------------------------------
# Threshold
# ------------------------------------------------------------------


class TestThreshold:
    def test_float_noise_does_not_round_up(self):
        # 0.01 * 1400 == 14.000000000000002 in IEEE-754.
        assert class_count_threshold(1400, 0.01) == 14

    def test_fractional_counts_round_up(self):
        assert class_count_threshold(1421, 0.01) == 15

    def test_floor_applies(self):
        assert class_count_threshold(100, 0.01) == 5
        assert class_count_threshold(10_000, 0.0) == 5


# ------------------------------------------------------------------
# Target construction
# ------------------------------------------------------------------


class TestCategoricalTarget:
    def test_combined_category_keeps_all_eight(self):
        labels = _labels(CATEGORY_COUNTS, n_missing=12_792)
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.01)
        assert target.task == ProbeTask.MULTICLASS
        assert target.threshold == 15
        assert target.classes == sorted(CATEGORY_COUNTS)
        assert target.excluded == {}
        assert target.n_labelled == 1421

    def test_combined_source_drops_six_rare_combinations(self):
        target = CategoricalTarget.from_labels(_labels(SOURCE_COUNTS), min_class_fraction=0.01)
        assert target.threshold == 143
        assert len(target.classes) == 10
        assert target.excluded == {k: v for k, v in SOURCE_COUNTS.items() if v < 143}

    def test_codes_index_kept_classes_and_nan_otherwise(self):
        labels = ["b", "a", None, "c", "rare"] + ["a"] * 10 + ["b"] * 10 + ["c"] * 10
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.0)
        assert target.classes == ["a", "b", "c"]
        assert target.excluded == {"rare": 1}
        assert target.codes[:5].tolist()[:2] == [1.0, 0.0]
        assert math.isnan(target.codes[2])  # unlabelled
        assert target.codes[3] == 2.0
        assert math.isnan(target.codes[4])  # excluded class
        assert target.codes.dtype == np.float64

    def test_class_counts_cover_kept_classes(self):
        target = CategoricalTarget.from_labels(_labels(CATEGORY_COUNTS), min_class_fraction=0.01)
        assert target.class_counts == {k: CATEGORY_COUNTS[k] for k in target.classes}

    def test_two_values_are_binary_without_threshold(self):
        labels = ["unsafe", "safe", None, "safe"] + ["safe"] * 200 + ["unsafe"]
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.1)
        assert target.task == ProbeTask.BINARY
        assert target.classes == ["safe", "unsafe"]
        assert target.target_mapping == {"safe": 0.0, "unsafe": 1.0}
        assert target.excluded == {}

    def test_three_values_dropping_to_two_is_binary_with_exclusions(self):
        labels = ["a"] * 50 + ["b"] * 50 + ["c"] * 2
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.0)
        assert target.task == ProbeTask.BINARY
        assert target.classes == ["a", "b"]
        assert target.excluded == {"c": 2}

    def test_numeric_binary_matches_legacy_mapping(self):
        # Legacy logreg mapped the larger of two numeric values to 1.
        target = CategoricalTarget.from_labels(["1", "0", "1", None, "0"], min_class_fraction=0.01)
        assert target.classes == ["0", "1"]
        assert target.codes[:2].tolist() == [1.0, 0.0]
        assert target.target_mapping is None  # numeric targets carry no mapping

    def test_numeric_labels_order_numerically(self):
        labels = [str(v) for v in [10, 2, 1] * 20]
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.0)
        assert target.classes == ["1", "2", "10"]
        assert target.numeric

    def test_from_numeric_canonicalises_integral_floats(self):
        y = np.array([0.0, 1.0, 2.0, np.nan, 2.0] * 10)
        target = CategoricalTarget.from_numeric(y, min_class_fraction=0.0)
        assert target.classes == ["0", "1", "2"]
        assert math.isnan(target.codes[3])
        assert target.codes[2] == 2.0

    def test_empty_strings_are_unlabelled(self):
        labels = ["", "a", "b", ""] + ["a"] * 10
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.0)
        assert target.classes == ["a", "b"]
        assert target.n_labelled == 12


class TestCategoricalTargetErrors:
    def test_single_class_raises(self):
        with pytest.raises(ProbeTargetError, match="two"):
            CategoricalTarget.from_labels(["a", "a", None], min_class_fraction=0.0)

    def test_fewer_than_two_kept_raises(self):
        labels = ["a"] * 100 + ["b"] * 2 + ["c"] * 2
        with pytest.raises(ProbeTargetError, match="[Mm]in class"):
            CategoricalTarget.from_labels(labels, min_class_fraction=0.0)

    def test_too_many_kept_classes_mentions_setting(self):
        labels = [f"c{i}" for i in range(MAX_PROBE_CLASSES + 1) for _ in range(10)]
        with pytest.raises(ProbeTargetError, match="[Mm]in class"):
            CategoricalTarget.from_labels(labels, min_class_fraction=0.0)

    def test_max_classes_exactly_is_allowed(self):
        labels = [f"c{i}" for i in range(MAX_PROBE_CLASSES) for _ in range(10)]
        target = CategoricalTarget.from_labels(labels, min_class_fraction=0.0)
        assert len(target.classes) == MAX_PROBE_CLASSES

    def test_continuous_numeric_raises(self):
        labels = [str(v) for v in np.linspace(0.5, 9.5, 60)]
        with pytest.raises(ProbeTargetError, match="continuous"):
            CategoricalTarget.from_labels(labels, min_class_fraction=0.0)

    def test_two_non_integral_values_still_binary(self):
        target = CategoricalTarget.from_labels(["0.5", "1.5"] * 30, min_class_fraction=0.0)
        assert target.task == ProbeTask.BINARY
        assert target.classes == ["0.5", "1.5"]


# ------------------------------------------------------------------
# Kind / task helpers (probing_types)
# ------------------------------------------------------------------


class TestProbeTypes:
    def test_new_kinds_registered(self):
        assert {"svc", "mlp_clf"} <= set(PROBE_KINDS)
        assert set(CLASSIFICATION_KINDS) == {"logreg", "svc", "mlp_clf"}

    def test_default_min_class_fraction(self):
        assert ProbeConfig(collection_name="c", target_field="f").min_class_fraction == 0.01

    @pytest.mark.parametrize("kind", ["logreg", "svc", "mlp_clf"])
    def test_classification_kinds_have_no_residual(self, kind):
        score, residual = score_field_names("category", kind)
        assert score == f"probe_category_{kind}_score"
        assert residual is None

    def test_derive_task_prefers_recorded_task(self):
        assert derive_probe_task("logreg", {"task": "multiclass"}) == ProbeTask.MULTICLASS

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("logreg", ProbeTask.BINARY),
            ("svc", ProbeTask.BINARY),
            ("ridge", ProbeTask.REGRESSION),
            ("mlp", ProbeTask.REGRESSION),
        ],
    )
    def test_derive_task_for_legacy_configs(self, kind, expected):
        assert derive_probe_task(kind, None) == expected
        assert derive_probe_task(kind, {}) == expected
