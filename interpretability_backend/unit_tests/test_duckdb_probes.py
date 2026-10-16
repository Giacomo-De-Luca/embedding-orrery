"""Unit tests for probe persistence (probes + probe_scores tables).

Covers upsert replace semantics, score round-trips (incl. NULL residuals and
retrain leaving no stray rows), multiclass predicted/probability columns and
their migration onto pre-existing databases, delete paths (probe-level and
dataset-level), and numeric/categorical metadata extraction including dotted
field names like "Conc.M".
"""

import duckdb
import numpy as np
import pandas as pd

from backend.clients.duckdb_client import DuckDBClient

COLLECTION = "probe_test_collection"
DATASET = "probe_test_dataset"


def _seed_dataset_and_collection(db: DuckDBClient, metadatas: list[dict] | None = None):
    """Create a dataset, items table, vector collection, and items.

    A "row_index" key is added to each metadata dict — insert_items_batch pops
    it into the row_index column (production embedding pipelines always set it,
    and get_numeric_metadata_field orders by it).
    """
    if metadatas is None:
        metadatas = [{"idx": i} for i in range(5)]
    metadatas = [{**meta, "row_index": i} for i, meta in enumerate(metadatas)]
    n_items = len(metadatas)
    db._conn.execute(
        "INSERT OR REPLACE INTO datasets (name, item_count) VALUES (?, ?)",
        [DATASET, n_items],
    )
    db._ensure_items_table(DATASET)
    db.insert_items_batch(
        DATASET,
        ids=[f"item_{i}" for i in range(n_items)],
        documents=[f"Document number {i}" for i in range(n_items)],
        metadatas=metadatas,
    )
    db._conn.execute(
        "INSERT OR REPLACE INTO vector_collections "
        "(collection_name, dataset_name, backend, vector_type) VALUES (?, ?, ?, ?)",
        [COLLECTION, DATASET, "chromadb", "dense"],
    )


def _insert_probe(db: DuckDBClient, kind: str = "ridge", metrics: dict | None = None):
    db.upsert_probe(
        COLLECTION,
        "rating",
        kind,
        config={"alpha": 1.0, "seed": 42},
        metrics=metrics if metrics is not None else {"val_r2": 0.8, "val_spearman": 0.9},
        direction=[0.1, 0.2, 0.3],
        scaler_mean=[0.0, 0.0, 0.0],
        scaler_scale=[1.0, 1.0, 1.0],
        intercept=0.5,
        artifact_path="/tmp/probes/rating/ridge",
        n_train=4,
        n_val=1,
    )


def _scores_df(item_ids: list[str], scores: list[float], residuals: list[float | None]):
    return pd.DataFrame({"item_id": item_ids, "score": scores, "residual": residuals})


class TestProbeSchema:
    def test_tables_exist(self, db: DuckDBClient):
        tables = db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "probes" in table_names
        assert "probe_scores" in table_names

    def test_probe_scores_has_multiclass_columns_last(self, db: DuckDBClient):
        columns = db._conn.execute(
            "SELECT name, type FROM pragma_table_info('probe_scores') ORDER BY cid"
        ).fetchall()
        assert columns[-2:] == [("predicted", "INTEGER"), ("probs", "FLOAT[]")]


_LEGACY_PROBE_SCORES_DDL = """
    CREATE TABLE probe_scores (
        collection_name VARCHAR NOT NULL,
        target_field    VARCHAR NOT NULL,
        kind            VARCHAR NOT NULL,
        item_id         VARCHAR NOT NULL,
        score           FLOAT NOT NULL,
        residual        FLOAT,
        PRIMARY KEY (collection_name, target_field, kind, item_id)
    )
"""


class TestProbeScoresMigration:
    """Databases created before multiclass support gain the new columns."""

    def _legacy_db(self, tmp_path):
        path = tmp_path / "legacy.duckdb"
        conn = duckdb.connect(str(path))
        conn.execute(_LEGACY_PROBE_SCORES_DDL)
        conn.execute("INSERT INTO probe_scores VALUES ('c', 'rating', 'ridge', 'item_0', 1.5, 0.1)")
        conn.close()
        return path

    def test_columns_added_and_legacy_rows_null(self, tmp_path):
        client = DuckDBClient(db_path=str(self._legacy_db(tmp_path)))
        try:
            names = [
                r[0]
                for r in client._conn.execute(
                    "SELECT name FROM pragma_table_info('probe_scores') ORDER BY cid"
                ).fetchall()
            ]
            assert names[-2:] == ["predicted", "probs"]
            row = client._conn.execute(
                "SELECT score, predicted, probs FROM probe_scores"
            ).fetchone()
            assert row == (1.5, None, None)
        finally:
            client.close()

    def test_reopening_is_idempotent(self, tmp_path):
        path = self._legacy_db(tmp_path)
        DuckDBClient(db_path=str(path)).close()
        client = DuckDBClient(db_path=str(path))
        try:
            count = client._conn.execute(
                "SELECT count(*) FROM pragma_table_info('probe_scores') WHERE name = 'probs'"
            ).fetchone()[0]
            assert count == 1
        finally:
            client.close()


class TestUpsertProbe:
    def test_insert_and_list(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        probes = db.list_probes(COLLECTION)
        assert len(probes) == 1
        p = probes[0]
        assert p["target_field"] == "rating"
        assert p["kind"] == "ridge"
        assert p["metrics"]["val_r2"] == 0.8
        assert p["config"]["alpha"] == 1.0
        assert p["n_train"] == 4
        assert p["n_val"] == 1
        assert isinstance(p["created_at"], str)

    def test_upsert_replaces_same_key(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db, metrics={"val_r2": 0.5})
        _insert_probe(db, metrics={"val_r2": 0.9})
        probes = db.list_probes(COLLECTION)
        assert len(probes) == 1
        assert probes[0]["metrics"]["val_r2"] == 0.9

    def test_different_kinds_coexist(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db, kind="ridge")
        _insert_probe(db, kind="massmean")
        probes = db.list_probes(COLLECTION)
        assert len(probes) == 2
        assert {p["kind"] for p in probes} == {"ridge", "massmean"}

    def test_nullable_fields(self, db: DuckDBClient):
        """MLP probes have no direction/scaler/intercept."""
        _seed_dataset_and_collection(db)
        db.upsert_probe(
            COLLECTION,
            "rating",
            "mlp",
            config={"epochs": 3},
            metrics={"val_r2": 0.7, "val_pearson": None},
            direction=None,
            scaler_mean=None,
            scaler_scale=None,
            intercept=None,
            artifact_path=None,
            n_train=4,
            n_val=1,
        )
        p = db.list_probes(COLLECTION)[0]
        assert p["direction"] is None
        assert p["intercept"] is None
        assert p["metrics"]["val_pearson"] is None

    def test_direction_round_trip(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        p = db.list_probes(COLLECTION)[0]
        assert [round(v, 4) for v in p["direction"]] == [0.1, 0.2, 0.3]
        assert p["intercept"] == 0.5

    def test_get_probe_matches_list_row(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        assert db.get_probe(COLLECTION, "rating", "ridge") == db.list_probes(COLLECTION)[0]

    def test_get_probe_missing_returns_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        assert db.get_probe(COLLECTION, "rating", "ridge") is None


class TestProbeScores:
    def test_round_trip_with_null_residuals(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db, kind="massmean")
        df = _scores_df(["item_0", "item_1", "item_2"], [0.1, 0.2, 0.3], [None, None, None])
        n = db.insert_probe_scores_bulk(COLLECTION, "rating", "massmean", df)
        assert n == 3
        result = db.get_probe_scores(COLLECTION, "rating", "massmean")
        assert result is not None
        assert result["item_ids"] == ["item_0", "item_1", "item_2"]
        assert [round(s, 4) for s in result["scores"]] == [0.1, 0.2, 0.3]
        assert result["residuals"] == [None, None, None]

    def test_mixed_residuals(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        df = _scores_df(["item_0", "item_1"], [1.5, 2.5], [0.1, None])
        db.insert_probe_scores_bulk(COLLECTION, "rating", "ridge", df)
        result = db.get_probe_scores(COLLECTION, "rating", "ridge")
        assert round(result["residuals"][0], 4) == 0.1
        assert result["residuals"][1] is None

    def test_retrain_leaves_no_strays(self, db: DuckDBClient):
        """Re-inserting with fewer items must delete the old key's rows first."""
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        db.insert_probe_scores_bulk(
            COLLECTION,
            "rating",
            "ridge",
            _scores_df(["item_0", "item_1", "item_2"], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        )
        db.insert_probe_scores_bulk(
            COLLECTION, "rating", "ridge", _scores_df(["item_0"], [9.0], [0.5])
        )
        result = db.get_probe_scores(COLLECTION, "rating", "ridge")
        assert result["item_ids"] == ["item_0"]
        assert result["scores"] == [9.0]

    def test_missing_probe_returns_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        assert db.get_probe_scores(COLLECTION, "nope", "ridge") is None

    def test_legacy_rows_have_no_predictions(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        db.insert_probe_scores_bulk(
            COLLECTION, "rating", "ridge", _scores_df(["item_0"], [1.0], [0.0])
        )
        result = db.get_probe_scores(COLLECTION, "rating", "ridge")
        assert result["predicted"] == [None]


PROBS = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])


def _insert_multiclass(db: DuckDBClient, probs: np.ndarray = PROBS):
    db.upsert_probe(COLLECTION, "category", "logreg", config={"task": "multiclass"})
    df = pd.DataFrame(
        {
            "item_id": ["item_0", "item_1", "item_2"],
            "score": probs.max(axis=1),
            "predicted": probs.argmax(axis=1),
        }
    )
    return db.insert_probe_scores_bulk(COLLECTION, "category", "logreg", df, probs=probs)


class TestMulticlassScores:
    def test_round_trip_predicted_and_confidence(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        assert _insert_multiclass(db) == 3
        result = db.get_probe_scores(COLLECTION, "category", "logreg")
        assert result["predicted"] == [0, 2, 1]
        assert [round(s, 4) for s in result["scores"]] == [0.7, 0.6, 0.5]
        assert result["residuals"] == [None, None, None]

    def test_class_probabilities_use_zero_based_index(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_multiclass(db)
        for k in range(3):
            result = db.get_probe_class_probabilities(COLLECTION, "category", "logreg", k)
            assert result["item_ids"] == ["item_0", "item_1", "item_2"]
            np.testing.assert_allclose(result["probabilities"], PROBS[:, k], atol=1e-6)

    def test_out_of_range_class_returns_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_multiclass(db)
        assert db.get_probe_class_probabilities(COLLECTION, "category", "logreg", 3) is None
        assert db.get_probe_class_probabilities(COLLECTION, "category", "logreg", -1) is None

    def test_probe_without_probabilities_returns_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        db.insert_probe_scores_bulk(
            COLLECTION, "rating", "ridge", _scores_df(["item_0"], [1.0], [0.0])
        )
        assert db.get_probe_class_probabilities(COLLECTION, "rating", "ridge", 0) is None
        assert db.get_probe_class_probabilities(COLLECTION, "missing", "ridge", 0) is None

    def test_retrain_replaces_probabilities(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_multiclass(db)
        _insert_multiclass(db, probs=PROBS[:, ::-1].copy())
        result = db.get_probe_class_probabilities(COLLECTION, "category", "logreg", 0)
        np.testing.assert_allclose(result["probabilities"], PROBS[:, 2], atol=1e-6)


class TestProbeDeletion:
    def test_delete_probe_removes_both_tables(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        db.insert_probe_scores_bulk(
            COLLECTION, "rating", "ridge", _scores_df(["item_0"], [1.0], [0.0])
        )
        assert db.delete_probe(COLLECTION, "rating", "ridge") is True
        assert db.list_probes(COLLECTION) == []
        assert db.get_probe_scores(COLLECTION, "rating", "ridge") is None

    def test_delete_probe_missing_returns_false(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        assert db.delete_probe(COLLECTION, "rating", "ridge") is False

    def test_delete_dataset_removes_probe_rows(self, db: DuckDBClient):
        _seed_dataset_and_collection(db)
        _insert_probe(db)
        db.insert_probe_scores_bulk(
            COLLECTION, "rating", "ridge", _scores_df(["item_0"], [1.0], [0.0])
        )
        assert db.delete_dataset(DATASET) is True
        n_probes = db._conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
        n_scores = db._conn.execute("SELECT COUNT(*) FROM probe_scores").fetchone()[0]
        assert n_probes == 0
        assert n_scores == 0


class TestNumericMetadataField:
    def test_numeric_values_ordered_by_row_index(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, metadatas=[{"rating": float(i) * 1.5} for i in range(4)])
        rows = db.get_numeric_metadata_field(DATASET, "rating")
        assert [r[0] for r in rows] == ["item_0", "item_1", "item_2", "item_3"]
        assert [r[1] for r in rows] == [0.0, 1.5, 3.0, 4.5]

    def test_numeric_strings_coerced(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, metadatas=[{"rating": "2.5"}, {"rating": "7"}])
        rows = db.get_numeric_metadata_field(DATASET, "rating")
        assert [r[1] for r in rows] == [2.5, 7.0]

    def test_nulls_and_junk_return_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(
            db,
            metadatas=[
                {"rating": 1.0},
                {"rating": None},
                {"rating": "not a number"},
                {"other": 5},
            ],
        )
        rows = db.get_numeric_metadata_field(DATASET, "rating")
        assert rows[0][1] == 1.0
        assert rows[1][1] is None
        assert rows[2][1] is None
        assert rows[3][1] is None

    def test_dotted_field_name(self, db: DuckDBClient):
        """Fields like "Conc.M" are literal keys, not nested JSON paths."""
        _seed_dataset_and_collection(db, metadatas=[{"Conc.M": 4.2}, {"Conc.M": 1.1}])
        rows = db.get_numeric_metadata_field(DATASET, "Conc.M")
        assert [r[1] for r in rows] == [4.2, 1.1]

    def test_missing_dataset_returns_empty(self, db: DuckDBClient):
        assert db.get_numeric_metadata_field("no_such_dataset", "rating") == []


class TestCategoricalMetadataField:
    def test_strings_ordered_by_row_index(self, db: DuckDBClient):
        _seed_dataset_and_collection(
            db, metadatas=[{"label": "unsafe"}, {"label": "safe"}, {"label": "safe"}]
        )
        rows = db.get_categorical_metadata_field(DATASET, "label")
        assert [r[0] for r in rows] == ["item_0", "item_1", "item_2"]
        assert [r[1] for r in rows] == ["unsafe", "safe", "safe"]
        assert {r[2] for r in rows} == {"VARCHAR"}

    def test_json_types_are_reported(self, db: DuckDBClient):
        _seed_dataset_and_collection(
            db,
            metadatas=[
                {"label": 3},
                {"label": -2},
                {"label": 3.0},
                {"label": True},
                {"label": [1, 2]},
                {"label": None},
                {"other": 1},
            ],
        )
        rows = db.get_categorical_metadata_field(DATASET, "label")
        assert [(r[1], r[2]) for r in rows] == [
            ("3", "UBIGINT"),
            ("-2", "BIGINT"),
            ("3.0", "DOUBLE"),
            ("true", "BOOLEAN"),
            ("[1,2]", "ARRAY"),
            (None, "NULL"),
            (None, None),
        ]

    def test_dotted_field_name(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, metadatas=[{"a.b": "x"}, {"a.b": "y"}])
        rows = db.get_categorical_metadata_field(DATASET, "a.b")
        assert [r[1] for r in rows] == ["x", "y"]

    def test_missing_dataset_returns_empty(self, db: DuckDBClient):
        assert db.get_categorical_metadata_field("no_such_dataset", "label") == []
