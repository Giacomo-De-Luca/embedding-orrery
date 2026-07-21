"""Unit tests for the TrainProbeInput -> ProbeConfig converter.

`build_probe_config` is duck-typed over the GraphQL input, so a
SimpleNamespace stands in for TrainProbeInput.
"""

from types import SimpleNamespace

import pytest

from backend.API.converters import build_probe_config


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
        seed=None,
        train_split=None,
        max_train_samples=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildProbeConfig:
    def test_defaults_applied(self):
        config = build_probe_config(_input())
        assert config.kind == "ridge"
        assert config.patience == 10
        assert config.max_train_samples == 50_000
        assert config.seed == 42

    def test_patience_threads_through(self):
        assert build_probe_config(_input(kind="mlp", patience=5)).patience == 5

    def test_patience_below_one_raises(self):
        with pytest.raises(ValueError, match="patience"):
            build_probe_config(_input(kind="mlp", patience=0))

    def test_max_train_samples_threads_through(self):
        assert build_probe_config(_input(max_train_samples=1000)).max_train_samples == 1000

    def test_new_kinds_accepted(self):
        assert build_probe_config(_input(kind="massmean_cov")).kind == "massmean_cov"
        assert build_probe_config(_input(kind="lasso", alpha=0.01)).alpha == 0.01

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            build_probe_config(_input(kind="pls"))
