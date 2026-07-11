"""Unit tests for SAE document activation storage and search."""

import json

import pandas as pd
import pytest

from backend.clients.duckdb_client import DuckDBClient

MODEL_ID = "gemma-3-4b-it"
SAE_ID = "9-gemmascope-2-res-16k"
COLLECTION = "test_collection"
DATASET = "test_dataset"


def _seed_dataset_and_collection(db: DuckDBClient, n_items: int = 5):
    """Create a dataset, items table, vector collection, and items."""
    db._conn.execute(
        "INSERT OR REPLACE INTO datasets (name, item_count) VALUES (?, ?)",
        [DATASET, n_items],
    )
    db._ensure_items_table(DATASET)
    db.insert_items_batch(
        DATASET,
        ids=[f"item_{i}" for i in range(n_items)],
        documents=[f"Document number {i} about topic {i}" for i in range(n_items)],
        metadatas=[{"idx": i} for i in range(n_items)],
    )
    db._conn.execute(
        "INSERT OR REPLACE INTO vector_collections "
        "(collection_name, dataset_name, backend, vector_type) VALUES (?, ?, ?, ?)",
        [COLLECTION, DATASET, "chromadb", "dense"],
    )


def _seed_sae_features(db: DuckDBClient, n: int = 10):
    """Insert SAE features with known labels for search testing."""
    df = pd.DataFrame(
        {
            "feature_index": list(range(n)),
            "density": [0.01 * (i + 1) for i in range(n)],
            "label": [
                "poetry and verse",
                "religion and faith",
                "yellow color",
                "mathematics equations",
                "poetry rhyme",
                "ocean and water",
                "religion prayer",
                "sports basketball",
                "yellow sunflower",
                "music rhythm",
            ][:n],
            "top_logits": [json.dumps([]) for _ in range(n)],
            "bottom_logits": [json.dumps([]) for _ in range(n)],
        }
    )
    db.insert_sae_features_batch(MODEL_ID, SAE_ID, df)


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------


class TestDocActivationsSchema:
    def test_table_exists(self, db: DuckDBClient):
        tables = db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "sae_document_activations" in table_names


# ------------------------------------------------------------------
# Insert & Retrieve
# ------------------------------------------------------------------


class TestDocActivationsInsert:
    def test_insert_single_document(self, db: DuckDBClient):
        activations = [(0, 1.5), (3, 0.8), (7, 2.1)]
        count = db.insert_document_activations_batch(COLLECTION, "item_0", activations)
        assert count == 3

        rows = db._conn.execute(
            "SELECT feature_index, activation FROM sae_document_activations "
            "WHERE collection_name = ? AND item_id = ? ORDER BY feature_index",
            [COLLECTION, "item_0"],
        ).fetchall()
        assert len(rows) == 3
        assert rows[0] == (0, pytest.approx(1.5))
        assert rows[1] == (3, pytest.approx(0.8))
        assert rows[2] == (7, pytest.approx(2.1))

    def test_insert_empty_activations(self, db: DuckDBClient):
        count = db.insert_document_activations_batch(COLLECTION, "item_0", [])
        assert count == 0

    def test_insert_bulk(self, db: DuckDBClient):
        df = pd.DataFrame(
            {
                "collection_name": [COLLECTION] * 4,
                "item_id": ["item_0", "item_0", "item_1", "item_1"],
                "feature_index": [0, 1, 0, 2],
                "activation": [1.0, 2.0, 0.5, 3.0],
            }
        )
        count = db.insert_document_activations_bulk(df)
        assert count == 4

    def test_insert_replace_on_conflict(self, db: DuckDBClient):
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 1.0)])
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 9.9)])

        row = db._conn.execute(
            "SELECT activation FROM sae_document_activations "
            "WHERE collection_name = ? AND item_id = ? AND feature_index = 0",
            [COLLECTION, "item_0"],
        ).fetchone()
        assert row[0] == pytest.approx(9.9)


# ------------------------------------------------------------------
# Resume (existing IDs)
# ------------------------------------------------------------------


class TestDocActivationsResume:
    def test_get_existing_item_ids(self, db: DuckDBClient):
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 1.0)])
        db.insert_document_activations_batch(COLLECTION, "item_1", [(1, 2.0)])

        existing = db.get_document_activation_item_ids(COLLECTION)
        assert existing == {"item_0", "item_1"}

    def test_empty_collection_returns_empty_set(self, db: DuckDBClient):
        existing = db.get_document_activation_item_ids("nonexistent")
        assert existing == set()

    def test_has_document_activations(self, db: DuckDBClient):
        assert not db.has_document_activations(COLLECTION)
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 1.0)])
        assert db.has_document_activations(COLLECTION)


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------


class TestDocActivationsDelete:
    def test_delete_collection(self, db: DuckDBClient):
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 1.0)])
        db.insert_document_activations_batch(COLLECTION, "item_1", [(1, 2.0)])
        deleted = db.delete_document_activations(COLLECTION)
        assert deleted == 2
        assert not db.has_document_activations(COLLECTION)

    def test_cascade_on_dataset_delete(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, n_items=3)
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 1.0)])
        db.insert_document_activations_batch(COLLECTION, "item_1", [(1, 2.0)])

        db.delete_dataset(DATASET)
        assert not db.has_document_activations(COLLECTION)


# ------------------------------------------------------------------
# Two-hop search
# ------------------------------------------------------------------


class TestDocActivationsSearch:
    @pytest.fixture(autouse=True)
    def _setup(self, db: DuckDBClient):
        """Seed features and document activations for search tests."""
        _seed_dataset_and_collection(db, n_items=5)
        _seed_sae_features(db, n=10)

        # item_0 activates "poetry" features (0, 4) strongly
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 5.0), (4, 3.0), (5, 0.1)])
        # item_1 activates "poetry" feature (0) weakly
        db.insert_document_activations_batch(COLLECTION, "item_1", [(0, 1.0), (7, 8.0)])
        # item_2 activates "religion" features (1, 6)
        db.insert_document_activations_batch(COLLECTION, "item_2", [(1, 4.0), (6, 2.0)])
        # item_3 activates "yellow" features (2, 8)
        db.insert_document_activations_batch(COLLECTION, "item_3", [(2, 3.0), (8, 6.0)])
        # item_4 activates nothing matching any label well
        db.insert_document_activations_batch(COLLECTION, "item_4", [(5, 0.5)])

    def test_search_poetry(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(COLLECTION, "poetry", MODEL_ID, SAE_ID)
        assert result["matched_feature_count"] == 2  # features 0 and 4
        assert len(result["results"]) == 2  # item_0 and item_1

        # item_0 has MAX(5.0, 3.0) = 5.0, item_1 has MAX(1.0) = 1.0
        assert result["results"][0]["item_id"] == "item_0"
        assert result["results"][0]["score"] == pytest.approx(5.0)
        assert result["results"][0]["matching_features"] == 2

        assert result["results"][1]["item_id"] == "item_1"
        assert result["results"][1]["score"] == pytest.approx(1.0)
        assert result["results"][1]["matching_features"] == 1

    def test_search_religion(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(COLLECTION, "religion", MODEL_ID, SAE_ID)
        assert result["matched_feature_count"] == 2  # features 1 and 6
        assert len(result["results"]) == 1  # only item_2
        assert result["results"][0]["item_id"] == "item_2"
        assert result["results"][0]["score"] == pytest.approx(4.0)

    def test_search_yellow(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(COLLECTION, "yellow", MODEL_ID, SAE_ID)
        assert result["matched_feature_count"] == 2  # features 2 and 8
        assert result["results"][0]["item_id"] == "item_3"
        # MAX(3.0, 6.0) = 6.0
        assert result["results"][0]["score"] == pytest.approx(6.0)

    def test_search_no_match(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(
            COLLECTION, "zzz_nonexistent", MODEL_ID, SAE_ID
        )
        assert result["matched_feature_count"] == 0
        assert result["results"] == []

    def test_search_enriches_with_document(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(COLLECTION, "poetry", MODEL_ID, SAE_ID)
        first = result["results"][0]
        assert first["document"] is not None
        assert "Document number 0" in first["document"]
        assert "metadata" in first

    def test_search_limit(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(
            COLLECTION, "poetry", MODEL_ID, SAE_ID, limit=1
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["item_id"] == "item_0"

    def test_matched_features_returned(self, db: DuckDBClient):
        result = db.search_documents_by_feature_labels(COLLECTION, "poetry", MODEL_ID, SAE_ID)
        assert result["matched_features"] is not None
        labels = {f["label"] for f in result["matched_features"]}
        assert "poetry and verse" in labels
        assert "poetry rhyme" in labels


# ------------------------------------------------------------------
# Search by explicit feature indices (combobox selection)
# ------------------------------------------------------------------


class TestDocActivationsFeatureIndicesSearch:
    @pytest.fixture(autouse=True)
    def _setup(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, n_items=3)
        db.insert_document_activations_batch(
            COLLECTION, "item_0", [(0, 5.0), (1, 3.0)]
        )
        db.insert_document_activations_batch(
            COLLECTION, "item_1", [(0, 1.0), (2, 8.0)]
        )
        db.insert_document_activations_batch(COLLECTION, "item_2", [(2, 4.0)])

    def test_single_feature(self, db: DuckDBClient):
        res = db.search_documents_by_feature_indices(COLLECTION, [0], ranking="max")
        results = res["results"]
        assert len(results) == 2
        assert res["total_matches"] == 2
        assert results[0]["item_id"] == "item_0"
        assert results[0]["score"] == pytest.approx(5.0)

    def test_multiple_features(self, db: DuckDBClient):
        # MAX ranking: item_1 has MAX(1.0, 8.0)=8.0, item_0 has MAX(5.0)=5.0
        results = db.search_documents_by_feature_indices(
            COLLECTION, [0, 2], ranking="max"
        )["results"]
        assert results[0]["item_id"] == "item_1"
        assert results[0]["score"] == pytest.approx(8.0)
        assert results[0]["matching_features"] == 2

    def test_empty_indices(self, db: DuckDBClient):
        assert db.search_documents_by_feature_indices(COLLECTION, []) == {
            "results": [],
            "total_matches": 0,
        }

    def test_no_match(self, db: DuckDBClient):
        assert db.search_documents_by_feature_indices(COLLECTION, [999]) == {
            "results": [],
            "total_matches": 0,
        }

    def test_limit_preserves_total_matches(self, db: DuckDBClient):
        res = db.search_documents_by_feature_indices(COLLECTION, [0, 2], limit=1)
        assert len(res["results"]) == 1
        assert res["total_matches"] == 3

    def test_enriches_document(self, db: DuckDBClient):
        results = db.search_documents_by_feature_indices(COLLECTION, [0], limit=1)[
            "results"
        ]
        assert results[0]["document"] is not None
        assert "Document number 0" in results[0]["document"]


class TestDocActivationsRankingModes:
    """Ranking modes over features with ~100x magnitude difference.

    Feature 0 ("weak") maxes at 10.0 over the collection; feature 1
    ("strong") maxes at 1000.0. item_0 matches both, item_1 only the
    strong one (harder), item_2 only the weak one.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, db: DuckDBClient):
        _seed_dataset_and_collection(db, n_items=3)
        db.insert_document_activations_batch(
            COLLECTION, "item_0", [(0, 10.0), (1, 500.0)]
        )
        db.insert_document_activations_batch(COLLECTION, "item_1", [(1, 1000.0)])
        db.insert_document_activations_batch(COLLECTION, "item_2", [(0, 8.0)])

    def test_max_lets_strong_feature_dominate(self, db: DuckDBClient):
        results = db.search_documents_by_feature_indices(
            COLLECTION, [0, 1], ranking="max"
        )["results"]
        assert [r["item_id"] for r in results] == ["item_1", "item_0", "item_2"]
        assert results[0]["score"] == pytest.approx(1000.0)

    def test_sum_is_raw_sum(self, db: DuckDBClient):
        results = db.search_documents_by_feature_indices(
            COLLECTION, [0, 1], ranking="sum"
        )["results"]
        assert [r["item_id"] for r in results] == ["item_1", "item_0", "item_2"]
        assert results[1]["score"] == pytest.approx(510.0)

    def test_scaled_sum_rewards_multi_feature_match(self, db: DuckDBClient):
        # item_0: 10/10 + 500/1000 = 1.5 beats item_1: 1000/1000 = 1.0
        results = db.search_documents_by_feature_indices(
            COLLECTION, [0, 1], ranking="scaled_sum"
        )["results"]
        assert [r["item_id"] for r in results] == ["item_0", "item_1", "item_2"]
        assert results[0]["score"] == pytest.approx(1.5)
        assert results[1]["score"] == pytest.approx(1.0)
        assert results[2]["score"] == pytest.approx(0.8)

    def test_scaled_sum_is_default(self, db: DuckDBClient):
        explicit = db.search_documents_by_feature_indices(
            COLLECTION, [0, 1], ranking="scaled_sum"
        )["results"]
        default = db.search_documents_by_feature_indices(COLLECTION, [0, 1])["results"]
        assert [r["item_id"] for r in default] == [r["item_id"] for r in explicit]
        assert default[0]["score"] == pytest.approx(explicit[0]["score"])

    def test_matching_features_count_first(self, db: DuckDBClient):
        # item_0 matches 2 features → first despite lower per-feature peaks;
        # item_1 vs item_2 (both 1 match) tie-break by scaled sum (1.0 > 0.8).
        results = db.search_documents_by_feature_indices(
            COLLECTION, [0, 1], ranking="matching_features"
        )["results"]
        assert [r["item_id"] for r in results] == ["item_0", "item_1", "item_2"]
        assert results[0]["matching_features"] == 2
        assert results[1]["matching_features"] == 1

    def test_invalid_mode_raises(self, db: DuckDBClient):
        with pytest.raises(ValueError, match="ranking mode"):
            db.search_documents_by_feature_indices(
                COLLECTION, [0, 1], ranking="bogus"
            )


# ------------------------------------------------------------------
# Sparse dot-product search (prompt → documents)
# ------------------------------------------------------------------


class TestDocActivationsDotProductSearch:
    @pytest.fixture(autouse=True)
    def _setup(self, db: DuckDBClient):
        """Seed document activations for dot-product search tests."""
        _seed_dataset_and_collection(db, n_items=4)

        # item_0: features 0, 1, 2 active
        db.insert_document_activations_batch(COLLECTION, "item_0", [(0, 5.0), (1, 3.0), (2, 1.0)])
        # item_1: features 0, 3 active
        db.insert_document_activations_batch(COLLECTION, "item_1", [(0, 2.0), (3, 4.0)])
        # item_2: features 1, 2, 3 active
        db.insert_document_activations_batch(COLLECTION, "item_2", [(1, 1.0), (2, 6.0), (3, 2.0)])
        # item_3: feature 4 only (no overlap with typical query)
        db.insert_document_activations_batch(COLLECTION, "item_3", [(4, 10.0)])

    def test_basic_dot_product(self, db: DuckDBClient):
        # Query vector: feature 0=1.0, feature 1=2.0
        # item_0: 5*1 + 3*2 = 11.0
        # item_1: 2*1 + 0 = 2.0
        # item_2: 0 + 1*2 = 2.0
        # item_3: 0 (no overlap)
        results = db.search_documents_by_activations(COLLECTION, [(0, 1.0), (1, 2.0)])
        assert len(results) == 3  # item_3 has no overlap
        assert results[0]["item_id"] == "item_0"
        assert results[0]["score"] == pytest.approx(11.0)
        assert results[0]["shared_features"] == 2

    def test_top_k_limits_features(self, db: DuckDBClient):
        # Query: features 0=1.0, 1=0.5, 2=10.0 — with top_k=1, only feature 2 used
        # item_0: 1*10 = 10.0 (feature 2)
        # item_2: 6*10 = 60.0 (feature 2)
        results = db.search_documents_by_activations(
            COLLECTION, [(0, 1.0), (1, 0.5), (2, 10.0)], top_k=1
        )
        assert results[0]["item_id"] == "item_2"
        assert results[0]["score"] == pytest.approx(60.0)

    def test_no_overlap_returns_empty(self, db: DuckDBClient):
        # Query: feature 99 — no document has it
        results = db.search_documents_by_activations(COLLECTION, [(99, 5.0)])
        assert results == []

    def test_empty_activations_returns_empty(self, db: DuckDBClient):
        results = db.search_documents_by_activations(COLLECTION, [])
        assert results == []

    def test_limit_respected(self, db: DuckDBClient):
        results = db.search_documents_by_activations(COLLECTION, [(0, 1.0), (1, 2.0)], limit=1)
        assert len(results) == 1
        assert results[0]["item_id"] == "item_0"

    def test_enriches_document(self, db: DuckDBClient):
        results = db.search_documents_by_activations(COLLECTION, [(0, 1.0), (1, 2.0)], limit=1)
        assert results[0]["document"] is not None
        assert "Document number 0" in results[0]["document"]
