"""Unit tests for the probing service.

The pure core (run_probe_core) is tested on synthetic data with no databases;
the orchestrator (train_probe_for_collection) is tested with a fake DuckDB
client and a monkeypatched embedding loader.
"""

import json

import numpy as np
import pytest

from backend.services import probing_service
from backend.services.probing_service import (
    MIN_VALID_SAMPLES,
    ProbeConfig,
    ProbeTrainingError,
    _build_spec,
    binary_target_mapping,
    run_probe_core,
    sanitize_field_key,
    score_field_names,
    train_probe_for_collection,
)
from interpret.probing.configs.probe import MLPProbeSpec, SklearnProbeSpec
from interpret.probing.probes.sklearn_probes import _mass_mean_cov_direction


def _linear_data(n: int = 500, d: int = 32, seed: int = 0, noise: float = 0.05):
    """Synthetic X, y with a strong linear relationship."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = X.astype(np.float64) @ w + rng.normal(scale=noise, size=n)
    return X, y


def _anisotropic_data(n: int = 600, seed: int = 0):
    """Signal buried under a high-variance shared nuisance direction.

    dim0 = signal + nuisance, dim1 = nuisance, dims 2-7 = iid noise. The
    difference-of-means direction points at dim0 and its projection is
    dominated by the nuisance; the covariance-corrected direction recovers
    dim0 − dim1 = signal.
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    nuis = rng.normal(scale=8.0, size=n)
    rest = rng.normal(size=(n, 6))
    X = np.column_stack([s + nuis, nuis, rest]).astype(np.float32)
    return X, s.copy()


def _ids(n: int) -> list[str]:
    return [f"item_{i}" for i in range(n)]


# ------------------------------------------------------------------
# Field naming
# ------------------------------------------------------------------


class TestFieldNaming:
    def test_sanitize_field_key(self):
        assert sanitize_field_key("Conc.M") == "Conc_M"
        assert sanitize_field_key("plain_field") == "plain_field"
        assert sanitize_field_key("weird key!") == "weird_key_"

    def test_score_field_names_ridge(self):
        score, residual = score_field_names("Conc.M", "ridge")
        assert score == "probe_Conc_M_ridge_score"
        assert residual == "probe_Conc_M_ridge_residual"

    def test_score_field_names_massmean_has_calibrated_residual(self):
        score, residual = score_field_names("rating", "massmean")
        assert score == "probe_rating_massmean_score"
        assert residual == "probe_rating_massmean_residual"

    def test_score_field_names_logreg_has_no_residual(self):
        # Probability scores are not target-unit predictions.
        _, residual = score_field_names("label", "logreg")
        assert residual is None


# ------------------------------------------------------------------
# Spec mapping
# ------------------------------------------------------------------


class TestBuildSpec:
    def test_ridge_spec(self):
        spec = _build_spec(ProbeConfig(collection_name="c", target_field="f"))
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "ridge"
        assert spec.save_directions is True
        assert spec.standardise is True

    def test_massmean_spec(self):
        spec = _build_spec(ProbeConfig(collection_name="c", target_field="f", kind="massmean"))
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "massmean"
        assert spec.save_directions is True

    def test_lasso_spec(self):
        spec = _build_spec(
            ProbeConfig(collection_name="c", target_field="f", kind="lasso", alpha=0.01)
        )
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "lasso"
        assert spec.alpha == 0.01
        assert spec.save_directions is True

    def test_massmean_cov_spec(self):
        spec = _build_spec(ProbeConfig(collection_name="c", target_field="f", kind="massmean_cov"))
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "massmean_cov"
        assert spec.save_directions is True
        assert spec.standardise is True

    def test_mlp_spec(self):
        spec = _build_spec(
            ProbeConfig(collection_name="c", target_field="f", kind="mlp", hidden_dims=[8])
        )
        assert isinstance(spec, MLPProbeSpec)
        assert spec.hidden_dims == [8]

    def test_mlp_default_hidden_dims(self):
        spec = _build_spec(ProbeConfig(collection_name="c", target_field="f", kind="mlp"))
        assert spec.hidden_dims == [256]

    def test_svr_spec(self):
        spec = _build_spec(
            ProbeConfig(collection_name="c", target_field="f", kind="svr", c=2.0, kernel="rbf")
        )
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "svr"
        assert spec.C == 2.0
        assert spec.kernel == "rbf"
        assert spec.save_models is True  # SVR has no coef_; estimator persisted

    def test_logreg_spec(self):
        spec = _build_spec(
            ProbeConfig(
                collection_name="c", target_field="f", kind="logreg",
                c=0.5, class_weight="balanced",
            )
        )
        assert isinstance(spec, SklearnProbeSpec)
        assert spec.kind == "logreg"
        assert spec.C == 0.5
        assert spec.class_weight == "balanced"
        assert spec.save_directions is True  # hyperplane normal for scoring

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            _build_spec(ProbeConfig(collection_name="c", target_field="f", kind="svm"))


# ------------------------------------------------------------------
# Pure core: ridge
# ------------------------------------------------------------------


class TestRunProbeCoreRidge:
    def test_recovers_linear_signal(self, tmp_path):
        X, y = _linear_data()
        config = ProbeConfig(collection_name="c", target_field="rating")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert out.metrics["val_r2"] > 0.9
        assert out.metrics["val_spearman"] > 0.9
        assert out.direction is not None and len(out.direction) == X.shape[1]
        assert out.scaler_mean is not None and len(out.scaler_mean) == X.shape[1]
        assert out.scaler_scale is not None
        assert out.intercept is not None
        assert len(out.scores) == len(y)
        assert out.residuals is not None
        assert all(r is not None for r in out.residuals)
        assert out.n_train == 400
        assert out.n_val == 100

    def test_scores_track_targets(self, tmp_path):
        X, y = _linear_data(n=300)
        config = ProbeConfig(collection_name="c", target_field="rating")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        corr = np.corrcoef(np.array(out.scores), y)[0, 1]
        assert corr > 0.9

    def test_null_targets_scored_but_not_trained(self, tmp_path):
        X, y = _linear_data(n=200)
        y[:20] = np.nan
        config = ProbeConfig(collection_name="c", target_field="rating")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert len(out.scores) == 200
        assert all(s is not None for s in out.scores)
        assert out.residuals[0] is None  # no target -> no residual
        assert out.residuals[50] is not None
        assert out.n_train + out.n_val == 180

    def test_artifacts_written(self, tmp_path):
        X, y = _linear_data(n=100)
        config = ProbeConfig(collection_name="c", target_field="rating")
        run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert (tmp_path / "probe_results.csv").exists()
        assert (tmp_path / "directions" / "L0_embedding_ridge.npz").exists()


# ------------------------------------------------------------------
# Pure core: massmean
# ------------------------------------------------------------------


class TestRunProbeCoreMassmean:
    def test_metrics_and_calibrated_residuals(self, tmp_path):
        X, y = _linear_data()
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert out.metrics["val_spearman"] > 0.5
        assert out.direction is not None
        assert out.intercept == 0.0
        # Residuals come from the calibrated readout: pred = slope·score + b,
        # residual = pred − y. On clean linear data the implied predictions
        # must track the targets.
        assert out.residuals is not None
        assert all(r is not None for r in out.residuals)
        preds = y + np.array(out.residuals, dtype=float)
        assert np.corrcoef(preds, y)[0, 1] > 0.5
        assert out.calibration is not None
        assert set(out.calibration) == {"slope", "intercept"}

    def test_null_targets_have_no_residual(self, tmp_path):
        X, y = _linear_data(n=200)
        y[:20] = np.nan
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert len(out.scores) == 200  # every row scored
        assert out.residuals[0] is None
        assert out.residuals[50] is not None

    def test_calibrated_r2_reported(self, tmp_path):
        """Massmean gets a calibrated R²: slope/intercept fitted on the train
        projections, R² evaluated on validation. On strongly linear data it
        must be high; it can never beat a perfect fit."""
        X, y = _linear_data()
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert out.metrics["val_r2"] is not None
        assert 0.2 < out.metrics["val_r2"] <= 1.0
        assert out.metrics["train_r2"] is not None

    def test_degenerate_targets_raise(self, tmp_path):
        """Constant targets -> NaN massmean direction -> non-finite scores.

        A probe whose scores are non-finite is useless (and NaN would corrupt
        JSON downstream), so the core must fail cleanly instead of persisting.
        """
        X, _ = _linear_data(n=100)
        y = np.full(100, 3.0)  # constant targets -> degenerate direction
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean")
        with pytest.raises(ProbeTrainingError, match="degenerate"):
            run_probe_core(X, y, _ids(100), config, tmp_path)

    def test_metrics_json_safe(self, tmp_path):
        """Whatever metrics survive parsing must serialize to strict JSON."""
        X, y = _linear_data(n=100)
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean")
        out = run_probe_core(X, y, _ids(100), config, tmp_path)
        dumped = json.dumps(out.metrics)
        assert "NaN" not in dumped and "Infinity" not in dumped


# ------------------------------------------------------------------
# Pure core: massmean_cov (Geometry-of-Truth covariance correction)
# ------------------------------------------------------------------


class TestRunProbeCoreMassmeanCov:
    def test_direction_recovers_discriminant(self):
        """Raw-space check: θ = Σ⁺(μ⁺ − μ⁻) points at dim0 − dim1 (= signal)."""
        X, y = _anisotropic_data()
        direction = _mass_mean_cov_direction(X.astype(np.float64), y)
        assert abs(float(np.linalg.norm(direction)) - 1.0) < 1e-8
        expected = np.zeros(X.shape[1])
        expected[0], expected[1] = 1.0, -1.0
        expected /= np.linalg.norm(expected)
        assert abs(float(direction @ expected)) > 0.8

    def test_beats_plain_massmean_on_anisotropic_data(self, tmp_path):
        X, y = _anisotropic_data()
        base = run_probe_core(
            X, y, _ids(len(y)),
            ProbeConfig(collection_name="c", target_field="rating", kind="massmean"),
            tmp_path / "mm",
        )
        cov = run_probe_core(
            X, y, _ids(len(y)),
            ProbeConfig(collection_name="c", target_field="rating", kind="massmean_cov"),
            tmp_path / "cov",
        )
        assert cov.metrics["val_spearman"] > base.metrics["val_spearman"] + 0.2
        assert cov.metrics["val_r2"] > 0.5

    def test_full_output_shape(self, tmp_path):
        """Same output contract as massmean: direction, calibration, residuals."""
        X, y = _linear_data()
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean_cov")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert out.direction is not None and len(out.direction) == X.shape[1]
        assert abs(float(np.linalg.norm(np.array(out.direction))) - 1.0) < 1e-6
        assert out.intercept == 0.0
        assert out.residuals is not None
        assert out.calibration is not None
        assert (tmp_path / "directions" / "L0_embedding_massmean_cov.npz").exists()

    def test_degenerate_targets_raise(self, tmp_path):
        X, _ = _linear_data(n=100)
        y = np.full(100, 3.0)
        config = ProbeConfig(collection_name="c", target_field="rating", kind="massmean_cov")
        with pytest.raises(ProbeTrainingError):
            run_probe_core(X, y, _ids(100), config, tmp_path)


# ------------------------------------------------------------------
# Pure core: lasso
# ------------------------------------------------------------------


class TestRunProbeCoreLasso:
    def test_recovers_linear_signal_with_residuals(self, tmp_path):
        X, y = _linear_data()
        config = ProbeConfig(
            collection_name="c", target_field="rating", kind="lasso", alpha=0.01
        )
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert out.metrics["val_r2"] > 0.85
        assert out.direction is not None and len(out.direction) == X.shape[1]
        assert out.residuals is not None  # predictive kind
        assert all(r is not None for r in out.residuals)
        assert (tmp_path / "directions" / "L0_embedding_lasso.npz").exists()

    def test_field_names_have_residual(self):
        score, residual = score_field_names("rating", "lasso")
        assert score == "probe_rating_lasso_score"
        assert residual == "probe_rating_lasso_residual"


# ------------------------------------------------------------------
# Pure core: MLP
# ------------------------------------------------------------------


class TestRunProbeCoreMlp:
    def test_tiny_mlp_runs(self, tmp_path):
        X, y = _linear_data(n=200, d=16)
        config = ProbeConfig(
            collection_name="c",
            target_field="rating",
            kind="mlp",
            hidden_dims=[8],
            epochs=3,
            patience=2,
        )
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)

        assert "val_r2" in out.metrics
        assert "val_spearman" in out.metrics
        assert len(out.scores) == 200
        assert out.residuals is not None
        assert out.direction is None
        assert out.scaler_mean is None
        assert out.intercept is None
        assert (tmp_path / "checkpoints" / "layer_0_embedding.pt").exists()


# ------------------------------------------------------------------
# Guardrails
# ------------------------------------------------------------------


class TestGuardrails:
    def test_too_few_valid_targets_raises(self, tmp_path):
        X, y = _linear_data(n=100)
        y[MIN_VALID_SAMPLES - 10 :] = np.nan  # only 40 valid
        config = ProbeConfig(collection_name="c", target_field="rating")
        with pytest.raises(ProbeTrainingError, match=str(MIN_VALID_SAMPLES)):
            run_probe_core(X, y, _ids(len(y)), config, tmp_path)

    def test_training_subsample_cap(self, tmp_path):
        X, y = _linear_data(n=200)
        config = ProbeConfig(collection_name="c", target_field="rating", max_train_samples=100)
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert out.n_train + out.n_val == 100
        assert len(out.scores) == 200

    def test_empty_metrics_raise(self, tmp_path, monkeypatch):
        """A fit that yields no metrics (errored or degenerate) must not persist."""
        X, y = _linear_data(n=100)
        config = ProbeConfig(collection_name="c", target_field="rating")
        monkeypatch.setattr(probing_service, "_parse_metrics", lambda _: {})
        with pytest.raises(ProbeTrainingError, match="no usable metrics"):
            run_probe_core(X, y, _ids(100), config, tmp_path)

    def test_stale_artifact_not_reused(self, tmp_path):
        """Retrains must never score with a previous run's direction file.

        Seeds the artifact path with an all-zero direction; if training didn't
        replace it, scores would be constant and metrics couldn't be high.
        """
        X, y = _linear_data(n=200)
        directions = tmp_path / "directions"
        directions.mkdir(parents=True)
        d = X.shape[1]
        np.savez(
            directions / "L0_embedding_ridge.npz",
            coef=np.zeros(d),
            intercept=np.atleast_1d(0.0),
            scaler_mean=np.zeros(d),
            scaler_scale=np.ones(d),
        )
        config = ProbeConfig(collection_name="c", target_field="rating")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert out.metrics["val_r2"] > 0.9
        assert np.std(np.array(out.scores)) > 0.1  # not the stale zero direction


# ------------------------------------------------------------------
# Orchestrator (fake DB + patched loader)
# ------------------------------------------------------------------


class FakeDB:
    def __init__(self, field_rows, dataset_name="ds", text_rows=None):
        self._field_rows = field_rows
        self._text_rows = text_rows
        self._dataset_name = dataset_name
        self.upserted = []
        self.score_inserts = []

    def get_dataset_name_for_collection(self, collection_name):
        return self._dataset_name

    def get_numeric_metadata_field(self, dataset_name, field):
        return self._field_rows

    def get_text_metadata_field(self, dataset_name, field):
        if self._text_rows is not None:
            return self._text_rows
        # Default: what the numeric rows look like as raw JSON strings.
        return [(i, None if v is None else str(v)) for i, v in self._field_rows]

    def upsert_probe(self, *args, **kwargs):
        self.upserted.append((args, kwargs))

    def insert_probe_scores_bulk(self, collection_name, target_field, kind, df):
        self.score_inserts.append(df)
        return len(df)


@pytest.fixture
def linear_setup(tmp_path, monkeypatch):
    """Fake db with 80 valid rows + loader returning matching linear X."""
    n, d = 80, 8
    X, y = _linear_data(n=n, d=d, seed=1)
    ids = _ids(n)
    rows = [(ids[i], float(y[i])) for i in range(n)]
    fake_db = FakeDB(rows)
    monkeypatch.setattr(probing_service, "_get_duckdb", lambda: fake_db)
    monkeypatch.setattr(probing_service, "load_embeddings_for_ids", lambda collection, req_ids: X)
    monkeypatch.setattr(probing_service, "PROBING_RESULTS_DIR", tmp_path)
    return fake_db


class TestOrchestrator:
    def test_happy_path_persists(self, linear_setup):
        fake_db = linear_setup
        config = ProbeConfig(collection_name="col", target_field="rating")
        result = train_probe_for_collection(config)

        assert result.error is None
        assert result.metrics["val_r2"] is not None
        assert result.score_field == "probe_rating_ridge_score"
        assert result.residual_field == "probe_rating_ridge_residual"
        assert result.n_scored == 80
        assert len(fake_db.upserted) == 1
        assert len(fake_db.score_inserts) == 1
        assert len(fake_db.score_inserts[0]) == 80

    def test_too_few_valid_returns_error_result(self, tmp_path, monkeypatch):
        rows = [(f"item_{i}", 1.0 if i < 10 else None) for i in range(100)]
        fake_db = FakeDB(rows)
        monkeypatch.setattr(probing_service, "_get_duckdb", lambda: fake_db)
        monkeypatch.setattr(probing_service, "PROBING_RESULTS_DIR", tmp_path)
        config = ProbeConfig(collection_name="col", target_field="rating")
        result = train_probe_for_collection(config)

        assert result.error is not None
        assert str(MIN_VALID_SAMPLES) in result.error
        assert fake_db.upserted == []

    def test_massmean_persists_calibration(self, linear_setup):
        """The calibration line lands in the persisted config snapshot."""
        fake_db = linear_setup
        config = ProbeConfig(collection_name="col", target_field="rating", kind="massmean")
        result = train_probe_for_collection(config)

        assert result.error is None
        _, kwargs = fake_db.upserted[0]
        cal = kwargs["config"]["calibration"]
        assert set(cal) == {"slope", "intercept"}

    def test_binary_categorical_fallback(self, tmp_path, monkeypatch):
        """A "safe"/"unsafe" column trains via the 0/1 mapping end-to-end."""
        n, d = 80, 8
        rng = np.random.default_rng(3)
        X = rng.normal(size=(n, d)).astype(np.float32)
        w = rng.normal(size=d)
        labels = np.where(X.astype(np.float64) @ w > 0, "unsafe", "safe")
        ids = _ids(n)
        numeric_rows = [(ids[i], None) for i in range(n)]  # TRY_CAST all-null
        text_rows = [(ids[i], str(labels[i])) for i in range(n)]
        fake_db = FakeDB(numeric_rows, text_rows=text_rows)
        monkeypatch.setattr(probing_service, "_get_duckdb", lambda: fake_db)
        monkeypatch.setattr(probing_service, "load_embeddings_for_ids", lambda c, i: X)
        monkeypatch.setattr(probing_service, "PROBING_RESULTS_DIR", tmp_path)
        config = ProbeConfig(collection_name="col", target_field="safety")
        result = train_probe_for_collection(config)

        assert result.error is None
        assert result.target_mapping == {"safe": 0.0, "unsafe": 1.0}
        assert result.metrics["val_spearman"] > 0.5
        # Mapping is recorded in the persisted config snapshot.
        _, kwargs = fake_db.upserted[0]
        assert kwargs["config"]["target_mapping"] == {"safe": 0.0, "unsafe": 1.0}

    def test_unknown_collection_returns_error(self, tmp_path, monkeypatch):
        fake_db = FakeDB([], dataset_name=None)
        monkeypatch.setattr(probing_service, "_get_duckdb", lambda: fake_db)
        monkeypatch.setattr(probing_service, "PROBING_RESULTS_DIR", tmp_path)
        config = ProbeConfig(collection_name="nope", target_field="rating")
        result = train_probe_for_collection(config)

        assert result.error is not None
        assert "nope" in result.error

    def test_missing_embeddings_returns_error(self, tmp_path, monkeypatch):
        rows = [(f"item_{i}", float(i)) for i in range(80)]
        fake_db = FakeDB(rows)
        monkeypatch.setattr(probing_service, "_get_duckdb", lambda: fake_db)
        monkeypatch.setattr(probing_service, "load_embeddings_for_ids", lambda c, i: None)
        monkeypatch.setattr(probing_service, "PROBING_RESULTS_DIR", tmp_path)
        config = ProbeConfig(collection_name="col", target_field="rating")
        result = train_probe_for_collection(config)

        assert result.error is not None
        assert "embedding" in result.error.lower()
        assert fake_db.upserted == []


# ------------------------------------------------------------------
# Binary categorical targets
# ------------------------------------------------------------------


class TestBinaryTargetMapping:
    def test_two_classes_map_sorted(self):
        vals = ["unsafe", "safe", None, "safe", "unsafe"]
        assert binary_target_mapping(vals) == {"safe": 0.0, "unsafe": 1.0}

    def test_non_binary_returns_none(self):
        assert binary_target_mapping(["a", "b", "c"]) is None  # 3 classes
        assert binary_target_mapping(["a", "a", None]) is None  # 1 class
        assert binary_target_mapping([None, None]) is None  # no values
        assert binary_target_mapping([]) is None

    def test_case_sensitive_classes(self):
        # "Safe" and "safe" are distinct values; with a third value this
        # is not binary.
        assert binary_target_mapping(["Safe", "safe", "unsafe"]) is None

    def test_massmean_imbalanced_binary_fails_cleanly(self, tmp_path):
        """Majority-0 binary targets degenerate massmean's median split.

        With >50% zeros the median is 0 and ">= median" puts every sample in
        the high group (empty low group -> NaN direction). We deliberately do
        NOT paper over this in the toolkit — the probe must fail with the
        clear degenerate-target error so the user picks ridge/mlp instead.
        """
        rng = np.random.default_rng(0)
        n, d = 300, 16
        w = rng.normal(size=d)
        X = rng.normal(size=(n, d))
        logits = X @ w
        y = (logits > np.quantile(logits, 0.7)).astype(float)  # ~70% zeros
        config = ProbeConfig(collection_name="c", target_field="label", kind="massmean")
        with pytest.raises(ProbeTrainingError):
            run_probe_core(X, y, _ids(n), config, tmp_path)

    def test_ridge_on_binary_targets(self, tmp_path):
        rng = np.random.default_rng(1)
        n, d = 300, 16
        w = rng.normal(size=d)
        X = rng.normal(size=(n, d))
        y = (X @ w > 0).astype(float)
        config = ProbeConfig(collection_name="c", target_field="label", kind="ridge")
        out = run_probe_core(X, y, _ids(n), config, tmp_path)
        assert out.metrics["val_spearman"] > 0.5


# ------------------------------------------------------------------
# Pure core: SVR (rbf) — nonlinear regression
# ------------------------------------------------------------------


class TestRunProbeCoreSVR:
    def test_recovers_signal_and_scores_all(self, tmp_path):
        X, y = _linear_data(n=300, d=16)
        config = ProbeConfig(collection_name="c", target_field="rating", kind="svr")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert out.metrics["val_r2"] > 0.5
        assert len(out.scores) == len(y)
        # Regression → residuals present (predictive kind).
        assert out.residuals is not None
        assert out.direction is None  # rbf SVR has no linear direction

    def test_persists_and_reloads_estimator(self, tmp_path):
        X, y = _linear_data(n=200, d=8)
        config = ProbeConfig(collection_name="c", target_field="rating", kind="svr")
        run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert (tmp_path / "models" / "L0_embedding_svr.joblib").exists()

    def test_scores_track_targets(self, tmp_path):
        X, y = _linear_data(n=300, d=16)
        config = ProbeConfig(collection_name="c", target_field="rating", kind="svr")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        corr = np.corrcoef(np.array(out.scores), y)[0, 1]
        assert corr > 0.6


# ------------------------------------------------------------------
# Pure core: logistic regression — binary only, probability scores
# ------------------------------------------------------------------


def _binary_data(n=300, d=16, seed=2):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X.astype(np.float64) @ w > 0).astype(np.float64)
    return X, y


class TestRunProbeCoreLogreg:
    def test_probability_scores_and_auc(self, tmp_path):
        X, y = _binary_data()
        config = ProbeConfig(collection_name="c", target_field="label", kind="logreg")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        # Classification metrics, not regression.
        assert out.metrics["val_auc"] > 0.7
        assert "val_accuracy" in out.metrics
        assert "val_r2" not in out.metrics
        # Scores are probabilities in [0, 1].
        assert all(0.0 <= s <= 1.0 for s in out.scores)
        # Probability score is not a target-unit prediction → no residual.
        assert out.residuals is None
        assert out.direction is not None  # separating hyperplane normal

    def test_rejects_continuous_target(self, tmp_path):
        X, y = _linear_data(n=200, d=8)  # continuous
        config = ProbeConfig(collection_name="c", target_field="rating", kind="logreg")
        with pytest.raises(ProbeTrainingError, match="binary"):
            run_probe_core(X, y, _ids(len(y)), config, tmp_path)

    def test_rejects_three_classes(self, tmp_path):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 8)).astype(np.float32)
        y = rng.integers(0, 3, size=200).astype(np.float64)  # 3 classes
        config = ProbeConfig(collection_name="c", target_field="label", kind="logreg")
        with pytest.raises(ProbeTrainingError, match="binary"):
            run_probe_core(X, y, _ids(200), config, tmp_path)

    def test_arbitrary_two_values_remapped(self, tmp_path):
        """Two distinct non-0/1 numeric values still train (remapped to 0/1)."""
        X, y = _binary_data()
        y = np.where(y > 0.5, 7.0, 3.0)  # classes {3, 7}
        config = ProbeConfig(collection_name="c", target_field="label", kind="logreg")
        out = run_probe_core(X, y, _ids(len(y)), config, tmp_path)
        assert out.metrics["val_auc"] > 0.7
        assert all(0.0 <= s <= 1.0 for s in out.scores)

    def test_c_reaches_the_estimator(self, tmp_path):
        """Regression: C must actually regularize (it was silently dropped once).

        Extreme L2 (C→0) shrinks the logreg coefficients toward zero, so the
        stored direction's norm must be much smaller than under weak
        regularization. Guards the spec.C -> LogisticRegression(C=...) plumbing.
        """
        X, y = _binary_data()
        weak = run_probe_core(
            X, y, _ids(len(y)),
            ProbeConfig(collection_name="c", target_field="label", kind="logreg", c=100.0),
            tmp_path / "weak",
        )
        strong = run_probe_core(
            X, y, _ids(len(y)),
            ProbeConfig(collection_name="c", target_field="label", kind="logreg", c=1e-4),
            tmp_path / "strong",
        )
        norm = lambda d: float(np.linalg.norm(np.array(d)))  # noqa: E731
        assert norm(strong.direction) < norm(weak.direction) * 0.1
