"""Unit tests for the probe converters.

`build_probe_config` (TrainProbeInput -> ProbeConfig) is duck-typed over the
GraphQL input, so a SimpleNamespace stands in for TrainProbeInput.
`probe_info_from_record` maps a stored probe row to the GraphQL ProbeInfo,
shared by the collectionProbes query and the trainProbe mutation.
"""

from types import SimpleNamespace

import pytest

from backend.API.converters import build_probe_config, probe_info_from_record


def _input(**overrides):
    base = dict(
        collection_name="col",
        target_field="rating",
        kind="ridge",
        alpha=None,
        c=None,
        kernel=None,
        class_weight=None,
        epochs=None,
        hidden_dims=None,
        patience=None,
        dev_split=None,
        activation=None,
        seed=None,
        train_split=None,
        max_train_samples=None,
        min_class_fraction=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildProbeConfig:
    def test_defaults_applied(self):
        config = build_probe_config(_input())
        assert config.kind == "ridge"
        assert config.patience == 10
        assert config.dev_split == 0.2
        assert config.max_train_samples == 50_000
        assert config.seed == 7

    def test_patience_threads_through(self):
        assert build_probe_config(_input(kind="mlp", patience=5)).patience == 5

    def test_patience_below_one_raises(self):
        with pytest.raises(ValueError, match="patience"):
            build_probe_config(_input(kind="mlp", patience=0))

    def test_dev_split_threads_through(self):
        assert build_probe_config(_input(kind="mlp", dev_split=0.3)).dev_split == 0.3
        # 0 is legal: disables early stopping.
        assert build_probe_config(_input(kind="mlp", dev_split=0.0)).dev_split == 0.0

    def test_dev_split_out_of_range_raises(self):
        with pytest.raises(ValueError, match="dev_split"):
            build_probe_config(_input(kind="mlp", dev_split=1.0))
        with pytest.raises(ValueError, match="dev_split"):
            build_probe_config(_input(kind="mlp", dev_split=-0.1))

    def test_activation_threads_through(self):
        assert build_probe_config(_input()).activation == "relu"
        assert build_probe_config(_input(kind="mlp", activation="gelu")).activation == "gelu"

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            build_probe_config(_input(kind="mlp", activation="swish"))

    def test_max_train_samples_threads_through(self):
        assert build_probe_config(_input(max_train_samples=1000)).max_train_samples == 1000

    def test_zero_epochs_raises(self):
        with pytest.raises(ValueError, match="epochs"):
            build_probe_config(_input(kind="mlp", epochs=0))

    def test_invalid_hidden_dims_raise(self):
        with pytest.raises(ValueError, match="hidden_dims"):
            build_probe_config(_input(kind="mlp", hidden_dims=[256, 0]))

    def test_new_kinds_accepted(self):
        assert build_probe_config(_input(kind="massmean_cov")).kind == "massmean_cov"
        assert build_probe_config(_input(kind="lasso", alpha=0.01)).alpha == 0.01

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            build_probe_config(_input(kind="pls"))

    def test_classifier_kinds_accept_balanced_weights(self):
        for kind in ("svc", "mlp_clf"):
            config = build_probe_config(_input(kind=kind, class_weight="balanced"))
            assert config.kind == kind
            assert config.class_weight == "balanced"

    def test_min_class_fraction_threads_through(self):
        assert build_probe_config(_input()).min_class_fraction == 0.01
        config = build_probe_config(_input(kind="logreg", min_class_fraction=0.025))
        assert config.min_class_fraction == 0.025
        assert build_probe_config(_input(min_class_fraction=0.0)).min_class_fraction == 0.0

    @pytest.mark.parametrize("value", [-0.01, 0.5, 1.0])
    def test_min_class_fraction_out_of_range_raises(self, value):
        with pytest.raises(ValueError, match="min_class_fraction"):
            build_probe_config(_input(kind="logreg", min_class_fraction=value))


def _record(kind: str, config: dict | None, **overrides) -> dict:
    record = dict(
        collection_name="col",
        target_field="category",
        kind=kind,
        config=config,
        metrics={"val_accuracy": 0.6},
        direction=None,
        scaler_mean=None,
        scaler_scale=None,
        intercept=None,
        artifact_path=None,
        n_train=80,
        n_val=20,
        created_at="2026-10-10 12:00:00",
    )
    record.update(overrides)
    return record


class TestProbeInfoFromRecord:
    def test_legacy_logreg_is_binary(self):
        info = probe_info_from_record(_record("logreg", {"c": 1.0}))
        assert info.task == "binary"
        assert info.classes is None
        assert info.score_field == "probe_category_logreg_score"
        assert info.residual_field is None
        assert info.created_at == "2026-10-10 12:00:00"

    def test_ridge_is_regression_with_residual(self):
        # Every config snapshot records min_class_fraction; only classifiers use it.
        info = probe_info_from_record(
            _record("ridge", {"min_class_fraction": 0.01}, target_field="Conc.M")
        )
        assert info.task == "regression"
        assert info.score_field == "probe_Conc_M_ridge_score"
        assert info.residual_field == "probe_Conc_M_ridge_residual"
        assert info.min_class_fraction is None

    def test_multiclass_record(self):
        config = {
            "task": "multiclass",
            "classes": ["ANALOGY", "EUPHEMISM", "HYPERBOLE"],
            "excluded_classes": {"RARE": 3},
            "min_class_fraction": 0.01,
            "target_mapping": None,
        }
        info = probe_info_from_record(_record("svc", config))
        assert info.task == "multiclass"
        assert info.classes == ["ANALOGY", "EUPHEMISM", "HYPERBOLE"]
        assert info.excluded_classes == {"RARE": 3}
        assert info.min_class_fraction == 0.01
        assert info.target_mapping is None
        assert info.metrics == {"val_accuracy": 0.6}
        assert (info.n_train, info.n_val) == (80, 20)

    def test_binary_text_record_keeps_target_mapping(self):
        config = {"task": "binary", "classes": ["safe", "unsafe"], "target_mapping": {"safe": 0.0, "unsafe": 1.0}}
        info = probe_info_from_record(_record("logreg", config))
        assert info.classes == ["safe", "unsafe"]
        assert info.target_mapping == {"safe": 0.0, "unsafe": 1.0}
