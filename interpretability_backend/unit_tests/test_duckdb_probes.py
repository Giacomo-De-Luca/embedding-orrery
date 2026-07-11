"""Unit tests for probe persistence (probes + probe_scores tables).

Covers upsert replace semantics, score round-trips (incl. NULL residuals and
retrain leaving no stray rows), delete paths (probe-level and dataset-level),
and numeric metadata extraction including dotted field names like "Conc.M".
"""

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


class TestTextMetadataField:
    def test_strings_ordered_by_row_index(self, db: DuckDBClient):
        _seed_dataset_and_collection(
            db, metadatas=[{"label": "unsafe"}, {"label": "safe"}, {"label": "safe"}]
        )
        rows = db.get_text_metadata_field(DATASET, "label")
        assert [r[0] for r in rows] == ["item_0", "item_1", "item_2"]
        assert [r[1] for r in rows] == ["unsafe", "safe", "safe"]

    def test_nulls_and_missing_return_none(self, db: DuckDBClient):
        _seed_dataset_and_collection(
            db, metadatas=[{"label": "safe"}, {"label": None}, {"other": 1}]
        )
        rows = db.get_text_metadata_field(DATASET, "label")
        assert [r[1] for r in rows] == ["safe", None, None]

    def test_numbers_come_back_as_strings(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, metadatas=[{"label": 1}, {"label": 0}])
        rows = db.get_text_metadata_field(DATASET, "label")
        assert [r[1] for r in rows] == ["1", "0"]

    def test_dotted_field_name(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, metadatas=[{"a.b": "x"}, {"a.b": "y"}])
        rows = db.get_text_metadata_field(DATASET, "a.b")
        assert [r[1] for r in rows] == ["x", "y"]

    def test_missing_dataset_returns_empty(self, db: DuckDBClient):
        assert db.get_text_metadata_field("no_such_dataset", "label") == []
