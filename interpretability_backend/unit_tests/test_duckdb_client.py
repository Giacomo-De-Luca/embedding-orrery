"""Unit tests for DuckDBClient — all tables, CRUD, and search."""


import duckdb
import numpy as np
import pytest

from backend.clients.duckdb_client import DuckDBClient

# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class TestSchema:
    def test_tables_created(self, db: DuckDBClient):
        tables = db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        expected = {
            "datasets", "vector_collections", "projections",
            "projection_metadata", "topic_extractions", "topic_info", "topic_assignments",
            "topic_centroids",
        }
        assert expected.issubset(table_names)
        # No global "items" table — items are per-dataset
        assert "items" not in table_names

    def test_schema_idempotent(self, db: DuckDBClient):
        """Calling _ensure_schema again should not fail."""
        db._ensure_schema()


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------

class TestDatasets:
    def test_create_and_get(self, db: DuckDBClient):
        name = db.create_dataset("test_ds", description="A test", source_type="local_file")
        assert name == "test_ds"

        ds = db.get_dataset("test_ds")
        assert ds is not None
        assert ds["name"] == "test_ds"
        assert ds["description"] == "A test"
        assert ds["source_type"] == "local_file"
        assert ds["count"] == 0

    def test_creates_items_table(self, db: DuckDBClient):
        db.create_dataset("my_ds")
        # Per-dataset items table should exist
        tables = db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "items_my_ds" in table_names

    def test_list_datasets(self, db: DuckDBClient):
        db.create_dataset("ds1", source_type="huggingface")
        db.create_dataset("ds2", source_type="local_file")
        datasets = db.list_datasets()
        assert len(datasets) == 2
        names = {d["name"] for d in datasets}
        assert names == {"ds1", "ds2"}

    def test_update_dataset(self, db: DuckDBClient):
        db.create_dataset("ds", description="old")
        db.update_dataset("ds", description="new", source_type="vector")
        ds = db.get_dataset("ds")
        assert ds["description"] == "new"
        assert ds["source_type"] == "vector"

    def test_delete_dataset_cascades(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["i1", "i2"], ["doc1", "doc2"], [{"a": 1}, {"a": 2}])
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
        db.insert_projections_batch("ds_dense", ["i1", "i2"], "pca_2d", [[1.0, 2.0], [3.0, 4.0]])

        assert db.delete_dataset("ds")
        assert db.get_dataset("ds") is None
        assert db._conn.execute("SELECT COUNT(*) FROM projections").fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM vector_collections").fetchone()[0] == 0
        # Items table should be dropped
        tables = {t[0] for t in db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()}
        assert "items_ds" not in tables

    def test_get_nonexistent(self, db: DuckDBClient):
        assert db.get_dataset("nope") is None

    def test_delete_nonexistent(self, db: DuckDBClient):
        assert db.delete_dataset("nope") is False


# ------------------------------------------------------------------
# Items (per-dataset tables)
# ------------------------------------------------------------------

class TestItems:
    def test_insert_and_get_ids(self, db: DuckDBClient):
        db.create_dataset("ds")
        count = db.insert_items_batch("ds", ["a", "b", "c"],
                                      ["doc a", "doc b", "doc c"],
                                      [{"x": 1}, {"x": 2}, {"x": 3}])
        assert count == 3
        ids = db.get_item_ids("ds")
        assert ids == {"a", "b", "c"}

    def test_get_items_by_ids(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a", "b"], ["doc a", "doc b"], [{"k": "v1"}, {"k": "v2"}])
        items = db.get_items_by_ids("ds", ["b"])
        assert len(items) == 1
        assert items[0]["id"] == "b"
        assert items[0]["document"] == "doc b"
        assert items[0]["metadata"]["k"] == "v2"

    def test_strips_projection_and_topic_keys(self, db: DuckDBClient):
        db.create_dataset("ds")
        meta = {
            "custom_field": "keep",
            "pca_2d": "[0.1, 0.2]",
            "umap_3d": "[1, 2, 3]",
            "topic_id": "5",
            "topic_label": "hello",
            "row_index": 42,
        }
        db.insert_items_batch("ds", ["a"], ["doc"], [meta])
        items = db.get_items_by_ids("ds", ["a"])
        stored_meta = items[0]["metadata"]
        assert "custom_field" in stored_meta
        assert "pca_2d" not in stored_meta
        assert "umap_3d" not in stored_meta
        assert "topic_id" not in stored_meta
        assert "row_index" not in stored_meta
        assert items[0]["row_index"] == 42

    def test_dataset_item_count(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a", "b"], ["x", "y"], [{}, {}])
        ds = db.get_dataset("ds")
        assert ds["count"] == 2

    def test_bulk_insert(self, db: DuckDBClient):
        db.create_dataset("ds")
        ids = [f"item_{i}" for i in range(250)]
        docs = [f"document {i}" for i in range(250)]
        metas = [{"idx": i} for i in range(250)]
        count = db.insert_items_batch("ds", ids, docs, metas)
        assert count == 250
        assert len(db.get_item_ids("ds")) == 250


# ------------------------------------------------------------------
# Vector Collections
# ------------------------------------------------------------------

class TestVectorCollections:
    def test_register_and_get(self, db: DuckDBClient):
        db.create_dataset("ds")
        vc_name = db.register_vector_collection(
            "ds", "chromadb", "ds_dense", "dense",
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        assert vc_name == "ds_dense"
        vcs = db.get_vector_collections("ds")
        assert len(vcs) == 1
        assert vcs[0]["collection_name"] == "ds_dense"
        assert vcs[0]["embedding_dim"] == 384

    def test_get_by_name(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.register_vector_collection("ds", "chromadb", "my_coll", "dense")
        vc = db.get_vector_collection("my_coll")
        assert vc is not None
        assert vc["backend"] == "chromadb"

    def test_multiple_collections_per_dataset(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.register_vector_collection("ds", "chromadb", "ds_bge", "dense",
                                      embedding_model="bge-m3")
        db.register_vector_collection("ds", "chromadb", "ds_gemini", "dense",
                                      embedding_model="gemini-embedding")
        vcs = db.get_vector_collections("ds")
        assert len(vcs) == 2
        models = {vc["embedding_model"] for vc in vcs}
        assert models == {"bge-m3", "gemini-embedding"}


# ------------------------------------------------------------------
# Projections
# ------------------------------------------------------------------

class TestProjections:
    def test_insert_and_read(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a", "b"], ["doc a", "doc b"], [{}, {}])
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")

        coords = [[1.0, 2.0], [3.0, 4.0]]
        count = db.insert_projections_batch("ds_dense", ["a", "b"], "pca_2d", coords)
        assert count == 2

        data = db.get_projection_data("ds_dense", "pca_2d")
        assert data is not None
        assert len(data["ids"]) == 2
        assert data["coordinates"][0] == pytest.approx([1.0, 2.0])
        assert data["coordinates"][1] == pytest.approx([3.0, 4.0])

    def test_separate_projections_per_vc(self, db: DuckDBClient):
        """Different vector collections should have independent projections."""
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a"], ["doc"], [{}])
        db.register_vector_collection("ds", "chromadb", "ds_bge", "dense")
        db.register_vector_collection("ds", "chromadb", "ds_gemini", "dense")

        db.insert_projections_batch("ds_bge", ["a"], "pca_2d", [[1.0, 2.0]])
        db.insert_projections_batch("ds_gemini", ["a"], "pca_2d", [[9.0, 8.0]])

        data1 = db.get_projection_data("ds_bge", "pca_2d")
        data2 = db.get_projection_data("ds_gemini", "pca_2d")
        assert data1["coordinates"][0] == pytest.approx([1.0, 2.0])
        assert data2["coordinates"][0] == pytest.approx([9.0, 8.0])

    def test_projection_metadata(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
        db.upsert_projection_metadata("ds_dense", "pca_2d", variance=[0.45, 0.32])

        row = db._conn.execute(
            "SELECT variance FROM projection_metadata WHERE collection_name = ? AND projection_type = ?",
            ["ds_dense", "pca_2d"]
        ).fetchone()
        assert row is not None
        assert list(row[0]) == pytest.approx([0.45, 0.32])

    def test_nonexistent_projection_returns_none(self, db: DuckDBClient):
        assert db.get_projection_data("nonexistent", "pca_2d") is None


# ------------------------------------------------------------------
# Text Search
# ------------------------------------------------------------------

class TestTextSearch:
    def _setup_docs(self, db: DuckDBClient):
        db.create_dataset("docs")
        db.insert_items_batch("docs",
            ["d1", "d2", "d3"],
            [
                "The mallard is a dabbling duck that breeds in temperate regions",
                "The domestic cat is a small carnivorous mammal",
                "Ducks and geese are both waterfowl but differ in many ways",
            ],
            [
                {"category": "bird", "region": "temperate"},
                {"category": "mammal", "region": "worldwide"},
                {"category": "bird", "region": "various"},
            ],
        )

    def test_document_contains_search(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "duck")
        assert results["total_matches"] == 2
        matched_ids = {m["id"] for m in results["matches"]}
        assert matched_ids == {"d1", "d3"}
        assert all(m["matched_field"] == "__document__" for m in results["matches"])
        assert any(m["snippet"] is not None for m in results["matches"])

    def test_document_case_insensitive(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "DUCK")
        assert results["total_matches"] == 2

    def test_metadata_field_search(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "bird", fields=["category"])
        assert results["total_matches"] == 2

    def test_combined_document_and_metadata(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "temperate", fields=["__document__", "region"])
        assert results["total_matches"] >= 1

    def test_exact_mode(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "bird", fields=["category"], mode="exact")
        assert results["total_matches"] == 2
        results = db.text_search("docs", "bir", fields=["category"], mode="exact")
        assert results["total_matches"] == 0

    def test_bm25_search(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search_bm25("docs", "duck waterfowl")
        assert len(results) >= 1
        assert all("score" in r for r in results)
        ids = [r["id"] for r in results]
        assert "d3" in ids

    def test_empty_search(self, db: DuckDBClient):
        self._setup_docs(db)
        results = db.text_search("docs", "xyznonexistent")
        assert results["total_matches"] == 0

    def test_limit_truncates_matches_but_total_is_full(self, db: DuckDBClient):
        db.create_dataset("many")
        n = 10
        db.insert_items_batch(
            "many",
            [f"i{k}" for k in range(n)],
            [f"the quick brown fox number {k}" for k in range(n)],
            [{"category": "fox"} for _ in range(n)],
        )
        results = db.text_search("many", "fox", limit=3)
        assert len(results["matches"]) == 3
        assert results["total_matches"] == n

    def test_limit_counts_deduplicated_rows_across_fields(self, db: DuckDBClient):
        # d1 matches "temperate" in both document and region metadata —
        # it must count once, and the capped matches must respect that.
        self._setup_docs(db)
        results = db.text_search("docs", "temperate", fields=["__document__", "region"], limit=1)
        assert len(results["matches"]) == 1
        assert results["total_matches"] == 1

    def test_limit_respects_metadata_filters(self, db: DuckDBClient):
        self._setup_docs(db)
        filters = [{"field": "category", "operator": "$eq", "value": "bird"}]
        # "a" matches all three documents, but only d1/d3 are birds.
        results = db.text_search("docs", "a", filters=filters, limit=1)
        assert results["total_matches"] == 2
        assert len(results["matches"]) == 1
        assert results["matches"][0]["id"] in {"d1", "d3"}

    def test_filters_without_limit_unchanged(self, db: DuckDBClient):
        self._setup_docs(db)
        filters = [{"field": "category", "operator": "$eq", "value": "mammal"}]
        results = db.text_search("docs", "a", filters=filters)
        assert results["total_matches"] == 1
        assert {m["id"] for m in results["matches"]} == {"d2"}


# ------------------------------------------------------------------
# Topics
# ------------------------------------------------------------------

class TestTopics:
    def _setup(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a", "b", "c"], ["x", "y", "z"], [{}, {}, {}])
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")

    def test_create_extraction_and_topics(self, db: DuckDBClient):
        self._setup(db)
        ext_id = db.create_topic_extraction("ds_dense", "ds", config={"min_topic_size": 10})

        db.insert_topic_info_batch(ext_id, [
            {"topic_id": -1, "label": "Unclustered", "count": 1,
             "keywords": [{"word": "noise", "score": 0.5}]},
            {"topic_id": 0, "label": "Topic A", "count": 2,
             "keywords": [{"word": "alpha", "score": 0.9}]},
        ])

        db.insert_topic_assignments_batch(ext_id, [
            {"item_id": "a", "topic_id": -1, "topic_label": "Unclustered"},
            {"item_id": "b", "topic_id": 0, "topic_label": "Topic A"},
            {"item_id": "c", "topic_id": 0, "topic_label": "Topic A"},
        ])

        topics = db.get_active_topics("ds_dense")
        assert topics is not None
        assert len(topics["topics"]) == 2
        assert topics["topics"][1]["label"] == "Topic A"

    def test_update_topic_label(self, db: DuckDBClient):
        self._setup(db)
        ext_id = db.create_topic_extraction("ds_dense", "ds")
        db.insert_topic_info_batch(ext_id, [
            {"topic_id": 0, "label": "old_label", "count": 2}
        ])
        db.insert_topic_assignments_batch(ext_id, [
            {"item_id": "a", "topic_id": 0, "topic_label": "old_label"},
            {"item_id": "b", "topic_id": 0, "topic_label": "old_label"},
        ])

        db.update_topic_label(ext_id, 0, "new_label")

        topics = db.get_active_topics("ds_dense")
        assert topics["topics"][0]["label"] == "new_label"

        row = db._conn.execute(
            "SELECT topic_label FROM topic_assignments WHERE extraction_id = ? AND item_id = 'a'",
            [ext_id]
        ).fetchone()
        assert row[0] == "new_label"

    def test_get_items_for_topic(self, db: DuckDBClient):
        self._setup(db)
        ext_id = db.create_topic_extraction("ds_dense", "ds")
        db.insert_topic_assignments_batch(ext_id, [
            {"item_id": "a", "topic_id": 0},
            {"item_id": "b", "topic_id": 1},
            {"item_id": "c", "topic_id": 0},
        ])
        items = db.get_items_for_topic(ext_id, 0)
        assert set(items) == {"a", "c"}

    def test_new_extraction_deactivates_previous(self, db: DuckDBClient):
        self._setup(db)
        ext1 = db.create_topic_extraction("ds_dense", "ds")
        ext2 = db.create_topic_extraction("ds_dense", "ds")

        row = db._conn.execute(
            "SELECT is_active FROM topic_extractions WHERE id = ?", [ext1]
        ).fetchone()
        assert row[0] is False

        row = db._conn.execute(
            "SELECT is_active FROM topic_extractions WHERE id = ?", [ext2]
        ).fetchone()
        assert row[0] is True

    def test_get_active_extraction_ids(self, db: DuckDBClient):
        self._setup(db)
        db.create_topic_extraction("ds_dense", "ds")
        active = db.create_topic_extraction("ds_dense", "ds")

        assert db.get_active_extraction_id("ds_dense") == active
        assert db.get_active_extraction_id("unknown") is None
        assert db.get_active_extractions() == [("ds_dense", active)]


class TestTopicCentroids:
    def _setup(self, db: DuckDBClient) -> str:
        db.create_dataset("ds")
        db.insert_items_batch("ds", ["a", "b"], ["x", "y"], [{}, {}])
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
        return db.create_topic_extraction("ds_dense", "ds")

    ROWS = [
        {"level": "topic", "topic_id": 0, "member_count": 2, "centroid": [0.5, 1.5]},
        {"level": "topic", "topic_id": -1, "member_count": 1, "centroid": [0.0, 1.0]},
        {
            "level": "subtopic",
            "topic_id": 3,
            "parent_topic_id": 0,
            "member_count": 2,
            "centroid": [0.25, 0.75],
        },
    ]

    def test_roundtrip_per_level(self, db: DuckDBClient):
        ext = self._setup(db)
        assert db.replace_topic_centroids(ext, self.ROWS) == 3

        topics = db.get_topic_centroids(ext, "topic")
        assert topics["ids"].tolist() == [-1, 0]
        assert topics["counts"].tolist() == [1, 2]
        np.testing.assert_allclose(topics["means"], [[0.0, 1.0], [0.5, 1.5]])

        subtopics = db.get_topic_centroids(ext, "subtopic")
        assert subtopics["ids"].tolist() == [3]
        assert subtopics["parents"].tolist() == [0]

        assert db.get_topic_centroids(ext, "nonexistent") is None

    def test_replace_drops_previous_rows(self, db: DuckDBClient):
        ext = self._setup(db)
        db.replace_topic_centroids(ext, self.ROWS)
        db.replace_topic_centroids(
            ext, [{"level": "topic", "topic_id": 5, "member_count": 2, "centroid": [1.0, 1.0]}]
        )

        assert db.get_topic_centroids(ext, "topic")["ids"].tolist() == [5]
        assert db.get_topic_centroids(ext, "subtopic") is None

    def test_failed_replace_keeps_previous_rows(self, db: DuckDBClient):
        ext = self._setup(db)
        db.replace_topic_centroids(ext, self.ROWS)

        bad = [{"level": None, "topic_id": 9, "member_count": 1, "centroid": [1.0, 1.0]}]
        with pytest.raises(duckdb.ConstraintException):
            db.replace_topic_centroids(ext, bad)

        assert db.get_topic_centroids(ext, "topic")["ids"].tolist() == [-1, 0]

    def test_delete_dataset_removes_centroids(self, db: DuckDBClient):
        ext = self._setup(db)
        db.replace_topic_centroids(ext, self.ROWS)

        assert db.delete_dataset("ds") is True
        count = db._conn.execute("SELECT count(*) FROM topic_centroids").fetchone()[0]
        assert count == 0


class TestTopicFilters:
    """Topic fields live in topic_assignments, not item JSON: filters must join."""

    def _setup(self, db: DuckDBClient) -> str:
        db.create_dataset("ds")
        db.insert_items_batch(
            "ds",
            ["a", "b", "c", "d"],
            ["alpha doc", "beta doc", "gamma doc", "delta doc"],
            [{"year": 2020}, {"year": 2021}, {"year": 2020}, {"year": 2021}],
        )
        db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
        db.register_vector_collection("ds", "chromadb", "ds_no_topics", "dense")

        # An older, now-inactive extraction that put everything in topic 0.
        stale = db.create_topic_extraction("ds_dense", "ds")
        db.insert_topic_assignments_batch(
            stale, [{"item_id": i, "topic_id": 0, "topic_label": "Old"} for i in "abcd"]
        )

        active = db.create_topic_extraction("ds_dense", "ds")
        db.insert_topic_assignments_batch(active, [
            {"item_id": "a", "topic_id": 0, "topic_label": "Cats",
             "subtopic_id": 10, "subtopic_label": "Lions"},
            {"item_id": "b", "topic_id": 0, "topic_label": "Cats",
             "subtopic_id": 11, "subtopic_label": "Tabbies"},
            {"item_id": "c", "topic_id": 1, "topic_label": "Dogs",
             "subtopic_id": 12, "subtopic_label": "Pugs"},
            {"item_id": "d", "topic_id": -1, "topic_label": "Unclustered",
             "subtopic_id": -1, "subtopic_label": "Unclustered"},
        ])
        return active

    def _ids(self, db: DuckDBClient, filters: list[dict], collection: str = "ds_dense"):
        items = db.get_filtered_items("ds", filters, collection_name=collection)
        return sorted(item["id"] for item in items)

    @pytest.mark.parametrize(
        ("field", "operator", "value", "expected"),
        [
            ("topic_id", "$eq", 0, ["a", "b"]),
            ("topic_id", "$eq", "1", ["c"]),
            ("topic_id", "$in", [0, 1], ["a", "b", "c"]),
            ("topic_id", "$ne", -1, ["a", "b", "c"]),
            ("topic_id", "$nin", [0], ["c", "d"]),
            ("topic_label", "$eq", "Dogs", ["c"]),
            ("subtopic_id", "$eq", 11, ["b"]),
            ("subtopic_label", "$in", ["Lions", "Pugs"], ["a", "c"]),
            ("topic_id", "$gt", 0.5, ["c"]),
            ("topic_label", "$gt", "Cats", ["c", "d"]),  # text compares as text
        ],
    )
    def test_topic_field_filters_use_active_assignments(
        self, db: DuckDBClient, field, operator, value, expected
    ):
        self._setup(db)
        assert self._ids(db, [{"field": field, "operator": operator, "value": value}]) == expected

    def test_topic_and_metadata_filters_combine_with_and(self, db: DuckDBClient):
        self._setup(db)
        filters = [
            {"field": "topic_id", "operator": "$eq", "value": 0},
            {"field": "year", "operator": "$eq", "value": 2020},
        ]
        assert self._ids(db, filters) == ["a"]

    def test_empty_in_list_matches_nothing(self, db: DuckDBClient):
        self._setup(db)
        assert self._ids(db, [{"field": "topic_id", "operator": "$in", "value": []}]) == []

    def test_collection_without_topics_matches_nothing(self, db: DuckDBClient):
        self._setup(db)
        filters = [{"field": "topic_id", "operator": "$eq", "value": 0}]
        assert self._ids(db, filters, collection="ds_no_topics") == []

    def test_topic_filter_without_collection_name_raises(self, db: DuckDBClient):
        self._setup(db)
        with pytest.raises(ValueError, match="collection"):
            db.get_filtered_items("ds", [{"field": "topic_id", "operator": "$eq", "value": 0}])

    def test_filtered_item_ids(self, db: DuckDBClient):
        self._setup(db)
        filters = [{"field": "topic_id", "operator": "$in", "value": [0, 1]}]
        ids = db.get_filtered_item_ids("ds", filters, collection_name="ds_dense")
        assert sorted(ids) == ["a", "b", "c"]

        year_ids = db.get_filtered_item_ids(
            "ds", [{"field": "year", "operator": "$gte", "value": 2021}]
        )
        assert sorted(year_ids) == ["b", "d"]

    def test_text_search_honours_topic_filters(self, db: DuckDBClient):
        self._setup(db)
        result = db.text_search(
            "ds",
            "doc",
            filters=[{"field": "topic_id", "operator": "$eq", "value": 1}],
            collection_name="ds_dense",
        )
        assert result["total_matches"] == 1
        assert [m["id"] for m in result["matches"]] == ["c"]

    def test_subtopic_labels_lookup(self, db: DuckDBClient):
        active = self._setup(db)
        assert db.get_subtopic_labels(active, [10, 12]) == {10: "Lions", 12: "Pugs"}
        assert db.get_subtopic_labels(active, []) == {}


# ------------------------------------------------------------------
# Field Analysis
# ------------------------------------------------------------------

class TestFieldAnalysis:
    def test_compute_analysis(self, db: DuckDBClient):
        db.create_dataset("ds")
        db.insert_items_batch("ds",
            ["a", "b", "c"],
            ["x", "y", "z"],
            [
                {"color": "red", "size": "large"},
                {"color": "blue", "size": "small"},
                {"color": "red", "size": "medium"},
            ],
        )
        analysis = db.compute_field_analysis("ds")
        assert "color" in analysis
        assert analysis["color"]["distinct_count"] == 2
        assert analysis["color"]["total"] == 3
        assert analysis["size"]["distinct_count"] == 3
