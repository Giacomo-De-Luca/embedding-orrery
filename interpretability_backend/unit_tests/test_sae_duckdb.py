"""Unit tests for SAE feature/activation storage in DuckDBClient."""

import json

import pandas as pd

from backend.clients.duckdb_client import DuckDBClient

MODEL_ID = "gemma-3-4b-it"
SAE_ID = "9-gemmascope-2-res-16k"


def _make_features_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_index": list(range(n)),
        "density": [0.01 * (i + 1) for i in range(n)],
        "label": [f"feature {i} explanation" for i in range(n)],
        "top_logits": [
            json.dumps([{"token": f"tok_{i}_{j}", "score": 0.5 + j * 0.1} for j in range(3)])
            for i in range(n)
        ],
        "bottom_logits": [
            json.dumps([{"token": f"bot_{i}_{j}", "score": 0.3 + j * 0.05} for j in range(3)])
            for i in range(n)
        ],
    })


def _make_activations_df(feature_indices: list, samples_per: int = 3) -> pd.DataFrame:
    rows = []
    for fi in feature_indices:
        for s in range(samples_per):
            rows.append({
                "id": f"act_{fi}_{s}",
                "model_id": MODEL_ID,
                "sae_id": SAE_ID,
                "feature_index": fi,
                "tokens": json.dumps([f"tok{t}" for t in range(10)]),
                "act_values": json.dumps([float(t) * (s + 1) for t in range(10)]),
                "max_value": float(9 * (s + 1)),
                "max_value_token_idx": 9,
                "min_value": 0.0,
                "qualifying_token_idx": 5,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class TestSaeSchema:
    def test_sae_tables_exist(self, db: DuckDBClient):
        tables = db._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "sae_features" in table_names
        assert "sae_activations" in table_names


# ------------------------------------------------------------------
# Feature CRUD
# ------------------------------------------------------------------

class TestSaeFeatures:
    def test_insert_and_get(self, db: DuckDBClient):
        df = _make_features_df(3)
        count = db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)
        assert count == 3

        feat = db.get_sae_feature(MODEL_ID, SAE_ID, 0)
        assert feat is not None
        assert feat["feature_index"] == 0
        assert feat["label"] == "feature 0 explanation"
        assert abs(feat["density"] - 0.01) < 1e-6
        assert len(feat["top_logits"]) == 3
        assert feat["top_logits"][0]["token"] == "tok_0_0"

    def test_get_nonexistent(self, db: DuckDBClient):
        result = db.get_sae_feature(MODEL_ID, SAE_ID, 999)
        assert result is None

    def test_upsert_replaces(self, db: DuckDBClient):
        df1 = _make_features_df(2)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df1)

        # Update label for feature 0
        df2 = pd.DataFrame({
            "feature_index": [0],
            "density": [0.99],
            "label": ["updated label"],
            "top_logits": [json.dumps([])],
            "bottom_logits": [json.dumps([])],
        })
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df2)

        feat = db.get_sae_feature(MODEL_ID, SAE_ID, 0)
        assert feat["label"] == "updated label"
        assert abs(feat["density"] - 0.99) < 1e-6

    def test_search_by_label(self, db: DuckDBClient):
        df = _make_features_df(10)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)

        results = db.search_sae_features(MODEL_ID, SAE_ID, query="feature 3")
        assert len(results) >= 1
        assert any(r["feature_index"] == 3 for r in results)

    def test_search_by_density(self, db: DuckDBClient):
        df = _make_features_df(10)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)

        # Only features with density >= 0.05
        results = db.search_sae_features(MODEL_ID, SAE_ID, min_density=0.05)
        assert all(r["density"] >= 0.05 for r in results)
        assert len(results) > 0

    def test_search_with_limit(self, db: DuckDBClient):
        df = _make_features_df(20)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)

        results = db.search_sae_features(MODEL_ID, SAE_ID, limit=5)
        assert len(results) == 5


# ------------------------------------------------------------------
# Activation CRUD
# ------------------------------------------------------------------

class TestSaeActivations:
    def test_insert_and_get(self, db: DuckDBClient):
        df = _make_activations_df([0, 1], samples_per=3)
        count = db.insert_sae_activations_batch(df)
        assert count == 6

        acts = db.get_sae_activations(MODEL_ID, SAE_ID, 0)
        assert len(acts) == 3
        # Ordered by max_value DESC
        assert acts[0]["max_value"] >= acts[1]["max_value"]

    def test_get_with_limit(self, db: DuckDBClient):
        df = _make_activations_df([0], samples_per=10)
        db.insert_sae_activations_batch(df)

        acts = db.get_sae_activations(MODEL_ID, SAE_ID, 0, limit=3)
        assert len(acts) == 3

    def test_get_nonexistent_feature(self, db: DuckDBClient):
        acts = db.get_sae_activations(MODEL_ID, SAE_ID, 999)
        assert acts == []

    def test_tokens_and_values_deserialized(self, db: DuckDBClient):
        df = _make_activations_df([0], samples_per=1)
        db.insert_sae_activations_batch(df)

        acts = db.get_sae_activations(MODEL_ID, SAE_ID, 0)
        assert len(acts) == 1
        assert isinstance(acts[0]["tokens"], list)
        assert isinstance(acts[0]["values"], list)
        assert len(acts[0]["tokens"]) == 10
        assert len(acts[0]["values"]) == 10


# ------------------------------------------------------------------
# Model listing
# ------------------------------------------------------------------

class TestSaeModels:
    def test_list_empty(self, db: DuckDBClient):
        models = db.list_sae_models()
        assert models == []

    def test_list_after_insert(self, db: DuckDBClient):
        df = _make_features_df(5)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)

        models = db.list_sae_models()
        assert len(models) == 1
        assert models[0]["model_id"] == MODEL_ID
        assert models[0]["sae_id"] == SAE_ID
        assert models[0]["feature_count"] == 5
        assert models[0]["activation_count"] == 0

    def test_list_with_activations(self, db: DuckDBClient):
        df = _make_features_df(3)
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)

        act_df = _make_activations_df([0, 1], samples_per=2)
        db.insert_sae_activations_batch(act_df)

        models = db.list_sae_models()
        assert len(models) == 1
        assert models[0]["feature_count"] == 3
        assert models[0]["activation_count"] == 4


# ------------------------------------------------------------------
# Cross-SAE search
# ------------------------------------------------------------------

SAE_ID_2 = "9-gemmascope-2-res-65k"
MODEL_ID_2 = "other-model"


def _make_features_df_labeled(n: int, prefix: str = "feature") -> pd.DataFrame:
    """Like _make_features_df but with a configurable label prefix."""
    return pd.DataFrame({
        "feature_index": list(range(n)),
        "density": [0.01 * (i + 1) for i in range(n)],
        "label": [f"{prefix} {i} explanation" for i in range(n)],
        "top_logits": [json.dumps([]) for _ in range(n)],
        "bottom_logits": [json.dumps([]) for _ in range(n)],
    })


class TestCrossSaeSearch:
    def test_search_all_saes_for_model(self, db: DuckDBClient):
        """model_id set, sae_id=None → results from all SAEs for that model."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(5, "alpha"))
        db.insert_sae_features_batch(MODEL_ID, SAE_ID_2, _make_features_df_labeled(5, "alpha"))

        results = db.search_sae_features(model_id=MODEL_ID, query="alpha")
        assert len(results) == 10
        sae_ids_found = {r["sae_id"] for r in results}
        assert SAE_ID in sae_ids_found
        assert SAE_ID_2 in sae_ids_found

    def test_search_with_sae_ids_list(self, db: DuckDBClient):
        """sae_ids list filters to specific SAEs."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(5, "beta"))
        db.insert_sae_features_batch(MODEL_ID, SAE_ID_2, _make_features_df_labeled(5, "beta"))

        results = db.search_sae_features(sae_ids=[SAE_ID], query="beta")
        assert len(results) == 5
        assert all(r["sae_id"] == SAE_ID for r in results)

    def test_search_all_models(self, db: DuckDBClient):
        """model_id=None, sae_id=None → results from all models."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(3, "gamma"))
        db.insert_sae_features_batch(MODEL_ID_2, SAE_ID, _make_features_df_labeled(3, "gamma"))

        results = db.search_sae_features(query="gamma")
        assert len(results) == 6
        model_ids_found = {r["model_id"] for r in results}
        assert MODEL_ID in model_ids_found
        assert MODEL_ID_2 in model_ids_found

    def test_sae_ids_overrides_sae_id(self, db: DuckDBClient):
        """sae_ids takes precedence over sae_id."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(3, "delta"))
        db.insert_sae_features_batch(MODEL_ID, SAE_ID_2, _make_features_df_labeled(3, "delta"))

        results = db.search_sae_features(
            model_id=MODEL_ID, sae_id=SAE_ID, sae_ids=[SAE_ID_2], query="delta",
        )
        assert all(r["sae_id"] == SAE_ID_2 for r in results)

    def test_results_include_model_and_sae(self, db: DuckDBClient):
        """Cross-SAE results include correct model_id and sae_id per row."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(2, "epsilon"))

        results = db.search_sae_features(query="epsilon")
        assert len(results) == 2
        for r in results:
            assert r["model_id"] == MODEL_ID
            assert r["sae_id"] == SAE_ID

    def test_no_filters_returns_all(self, db: DuckDBClient):
        """No filters at all (WHERE TRUE) returns all features."""
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df_labeled(3, "zeta"))
        db.insert_sae_features_batch(MODEL_ID_2, SAE_ID_2, _make_features_df_labeled(4, "eta"))

        results = db.search_sae_features(limit=100)
        assert len(results) == 7


# ------------------------------------------------------------------
# Deletion
# ------------------------------------------------------------------

class TestSaeDeletion:
    def test_delete(self, db: DuckDBClient):
        db.insert_sae_features_batch(MODEL_ID, SAE_ID, _make_features_df(3))
        db.insert_sae_activations_batch(_make_activations_df([0, 1]))

        ok = db.delete_sae_data(MODEL_ID, SAE_ID)
        assert ok is True

        assert db.get_sae_feature(MODEL_ID, SAE_ID, 0) is None
        assert db.get_sae_activations(MODEL_ID, SAE_ID, 0) == []
        assert db.list_sae_models() == []
