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
        dev_split=None,
        activation=None,
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
