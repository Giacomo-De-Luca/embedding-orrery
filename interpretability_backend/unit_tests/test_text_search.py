"""Tests for DuckDBClient text_search — document + metadata searching."""

import pytest
from backend.clients.duckdb_client import DuckDBClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """Fresh in-memory DuckDB client with test data."""
    client = DuckDBClient(db_path=":memory:")

    ds_id = client.create_dataset("test_col")
    client.insert_items_batch(
        ds_id,
        ["id_1", "id_2", "id_3", "id_4"],
        [
            "The quick brown fox jumps over the lazy dog",
            "Hello world, this is a test document",
            "Python programming is fun and productive",
            "chromadb is a vector database",
        ],
        [
            {"word": "fox", "category": "animal"},
            {"word": "hello", "category": "greeting"},
            {"word": "python", "category": "programming"},
            {"word": "chromadb", "category": "database"},
        ],
    )
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Tests: document search
# ---------------------------------------------------------------------------

class TestDocumentSearch:
    def test_document_search_case_insensitive(self, db: DuckDBClient):
        result = db.text_search("test_col", "fox")
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" in ids
        assert result["total_matches"] >= 1

    def test_document_search_returns_snippets(self, db: DuckDBClient):
        result = db.text_search("test_col", "fox")
        doc_matches = [m for m in result["matches"] if m["matched_field"] == "__document__"]
        assert len(doc_matches) > 0
        assert doc_matches[0]["snippet"] is not None
        assert "fox" in doc_matches[0]["snippet"].lower()

    def test_case_sensitive_document_search(self, db: DuckDBClient):
        result = db.text_search("test_col", "The quick", mode="contains", case_sensitive=True)
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" in ids

    def test_case_sensitive_no_match(self, db: DuckDBClient):
        result = db.text_search("test_col", "the quick", mode="contains", case_sensitive=True)
        # "the quick" (lowercase) should NOT match "The quick" (uppercase)
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" not in ids

    def test_default_fields_searches_document_only(self, db: DuckDBClient):
        """fields=None should search documents only."""
        result = db.text_search("test_col", "fox", fields=None)
        for m in result["matches"]:
            assert m["matched_field"] == "__document__"

    def test_no_matches(self, db: DuckDBClient):
        result = db.text_search("test_col", "zzzzzzzznotfound")
        assert result["matches"] == []
        assert result["total_matches"] == 0

    def test_snippet_long_document(self, db: DuckDBClient):
        """Snippet should be truncated with ellipsis for long documents."""
        ds_id = db.create_dataset("long_docs")
        long_doc = "A" * 200 + "MATCH" + "B" * 200
        db.insert_items_batch(ds_id, ["l1"], [long_doc], [{}])

        result = db.text_search("long_docs", "MATCH", case_sensitive=True)
        assert result["total_matches"] == 1
        snippet = result["matches"][0]["snippet"]
        assert "MATCH" in snippet
        assert snippet.startswith("...")
        assert snippet.endswith("...")


# ---------------------------------------------------------------------------
# Tests: metadata field search
# ---------------------------------------------------------------------------

class TestMetadataSearch:
    def test_metadata_field_search_contains(self, db: DuckDBClient):
        result = db.text_search("test_col", "fox", fields=["word"], mode="contains")
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" in ids
        assert result["matches"][0]["matched_field"] == "word"

    def test_metadata_field_search_exact(self, db: DuckDBClient):
        result = db.text_search("test_col", "fox", fields=["word"], mode="exact")
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" in ids
        assert result["total_matches"] == 1

    def test_metadata_snippet_is_none(self, db: DuckDBClient):
        """Metadata matches should not have snippets."""
        result = db.text_search("test_col", "fox", fields=["word"], mode="contains")
        for m in result["matches"]:
            if m["matched_field"] != "__document__":
                assert m["snippet"] is None


# ---------------------------------------------------------------------------
# Tests: multi-field and deduplication
# ---------------------------------------------------------------------------

class TestMultiField:
    def test_multi_field_union(self, db: DuckDBClient):
        """Searching document + metadata should union results without duplicates."""
        result = db.text_search(
            "test_col", "python",
            fields=["__document__", "word"],
            mode="contains",
        )
        ids = [m["id"] for m in result["matches"]]
        assert "id_3" in ids
        # Should not be duplicated
        assert ids.count("id_3") == 1

    def test_exact_mode_metadata_partial_no_match(self, db: DuckDBClient):
        """Exact mode should not match partial strings."""
        result = db.text_search("test_col", "fo", fields=["word"], mode="exact")
        assert result["total_matches"] == 0


# ---------------------------------------------------------------------------
# Tests: filtered items (JSON metadata filtering)
# ---------------------------------------------------------------------------

class TestFilteredItems:
    def test_eq_filter(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [
            {"field": "category", "operator": "$eq", "value": "animal"},
        ])
        assert len(items) == 1
        assert items[0]["id"] == "id_1"

    def test_ne_filter(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [
            {"field": "category", "operator": "$ne", "value": "animal"},
        ])
        assert len(items) == 3
        ids = {i["id"] for i in items}
        assert "id_1" not in ids

    def test_in_filter(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [
            {"field": "category", "operator": "$in", "value": ["animal", "database"]},
        ])
        ids = {i["id"] for i in items}
        assert ids == {"id_1", "id_4"}

    def test_nin_filter(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [
            {"field": "category", "operator": "$nin", "value": ["animal", "database"]},
        ])
        ids = {i["id"] for i in items}
        assert ids == {"id_2", "id_3"}

    def test_no_filters_returns_all(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [])
        assert len(items) == 4

    def test_limit_and_offset(self, db: DuckDBClient):
        items = db.get_filtered_items("test_col", [], limit=2, offset=0)
        assert len(items) == 2
        items2 = db.get_filtered_items("test_col", [], limit=2, offset=2)
        assert len(items2) == 2
        # No overlap
        ids1 = {i["id"] for i in items}
        ids2 = {i["id"] for i in items2}
        assert ids1.isdisjoint(ids2)

    def test_numeric_filter(self, db: DuckDBClient):
        """Numeric comparison operators on metadata fields."""
        ds_id = db.create_dataset("nums")
        db.insert_items_batch(ds_id,
            ["n1", "n2", "n3"],
            ["a", "b", "c"],
            [{"score": 10}, {"score": 20}, {"score": 30}],
        )
        items = db.get_filtered_items("nums", [
            {"field": "score", "operator": "$gt", "value": 15},
        ])
        ids = {i["id"] for i in items}
        assert ids == {"n2", "n3"}

        items = db.get_filtered_items("nums", [
            {"field": "score", "operator": "$lte", "value": 20},
        ])
        ids = {i["id"] for i in items}
        assert ids == {"n1", "n2"}

    def test_nonexistent_dataset(self, db: DuckDBClient):
        items = db.get_filtered_items("nonexistent", [])
        assert items == []


# ---------------------------------------------------------------------------
# Tests: text search with metadata filters
# ---------------------------------------------------------------------------

class TestTextSearchWithFilters:
    def test_eq_filter_restricts_results(self, db: DuckDBClient):
        """Text search with category=animal should only match items in that category."""
        result = db.text_search(
            "test_col", "the",
            filters=[{"field": "category", "operator": "$eq", "value": "animal"}],
        )
        ids = [m["id"] for m in result["matches"]]
        # "The quick brown fox..." matches "the" and is category=animal
        assert "id_1" in ids
        # "Hello world, this is a test document" matches "the" but is category=greeting
        assert "id_2" not in ids

    def test_in_filter(self, db: DuckDBClient):
        result = db.text_search(
            "test_col", "is",
            filters=[{"field": "category", "operator": "$in", "value": ["programming", "database"]}],
        )
        ids = [m["id"] for m in result["matches"]]
        # "Python programming is fun..." and "chromadb is a vector database" match
        assert "id_3" in ids
        assert "id_4" in ids
        # "this is a test" matches "is" but category=greeting, excluded
        assert "id_2" not in ids

    def test_filter_no_matches(self, db: DuckDBClient):
        result = db.text_search(
            "test_col", "fox",
            filters=[{"field": "category", "operator": "$eq", "value": "nonexistent"}],
        )
        assert result["matches"] == []
        assert result["total_matches"] == 0

    def test_none_filters_same_as_no_filters(self, db: DuckDBClient):
        """filters=None and filters=[] should behave like unfiltered search."""
        result_none = db.text_search("test_col", "fox", filters=None)
        result_empty = db.text_search("test_col", "fox", filters=[])
        result_default = db.text_search("test_col", "fox")
        assert result_none == result_default
        assert result_empty == result_default

    def test_filter_with_metadata_field_search(self, db: DuckDBClient):
        """Filters should also apply when searching metadata fields."""
        result = db.text_search(
            "test_col", "python",
            fields=["word"],
            filters=[{"field": "category", "operator": "$eq", "value": "programming"}],
        )
        ids = [m["id"] for m in result["matches"]]
        assert "id_3" in ids
        assert len(ids) == 1

    def test_filter_with_all_fields(self, db: DuckDBClient):
        """Filters apply to combined document + metadata search."""
        result = db.text_search(
            "test_col", "fox",
            fields=["__document__", "word"],
            filters=[{"field": "category", "operator": "$eq", "value": "animal"}],
        )
        ids = [m["id"] for m in result["matches"]]
        assert "id_1" in ids
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Tests: _build_metadata_where helper
# ---------------------------------------------------------------------------

class TestBuildMetadataWhere:
    def test_empty_filters(self, db: DuckDBClient):
        where_sql, params = db._build_metadata_where([])
        assert where_sql == "TRUE"
        assert params == []

    def test_single_eq(self, db: DuckDBClient):
        where_sql, params = db._build_metadata_where([
            {"field": "category", "operator": "$eq", "value": "animal"},
        ])
        assert "= ?" in where_sql
        assert "$.category" in params
