"""Projection-only collection loading avoids retransmitting shared item data."""

import pytest

from backend.API import queries as query_module, schema


def _seed_projection_collection(db):
    db.create_dataset("ds")
    db.insert_items_batch(
        "ds",
        ["a", "b"],
        ["document a", "document b"],
        [
            {"group": "one", "row_index": 0},
            {"group": "two", "row_index": 1},
        ],
    )
    db.register_vector_collection("ds", "chromadb", "ds_dense", "dense")
    db.insert_projections_batch(
        "ds_dense",
        ["a", "b"],
        "umap_3d",
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_projection_coordinate_reader_returns_only_ordered_coordinates(db):
    _seed_projection_collection(db)

    projection = db.get_projection_coordinates("ds_dense", "umap_3d")

    assert projection is not None
    assert projection["coordinates"][0] == pytest.approx([1.0, 2.0, 3.0])
    assert projection["coordinates"][1] == pytest.approx([4.0, 5.0, 6.0])
    assert projection["item_signature"]


def test_projection_coordinate_reader_falls_back_to_item_insertion_order(db):
    db.create_dataset("unordered")
    db.insert_items_batch(
        "unordered",
        ["first", "second"],
        ["document first", "document second"],
        [{}, {}],
    )
    db.register_vector_collection(
        "unordered",
        "chromadb",
        "unordered_dense",
        "dense",
    )
    db.insert_projections_batch(
        "unordered_dense",
        ["first", "second"],
        "pca_2d",
        [[1.0, 2.0], [3.0, 4.0]],
    )

    projection = db.get_projection_coordinates("unordered_dense", "pca_2d")

    assert projection is not None
    assert projection["coordinates"][0] == pytest.approx([1.0, 2.0])
    assert projection["coordinates"][1] == pytest.approx([3.0, 4.0])


def test_item_batches_assign_monotonic_row_indices_when_missing(db):
    db.create_dataset("generated_order")
    db.insert_items_batch(
        "generated_order",
        ["first", "second"],
        ["first", "second"],
        [{}, {}],
    )
    db.insert_items_batch(
        "generated_order",
        ["third"],
        ["third"],
        [{}],
    )

    rows = db.get_items_columns("generated_order", ("id", "row_index"))

    assert rows == [("first", 0), ("second", 1), ("third", 2)]


def test_appending_to_legacy_null_indices_preserves_existing_physical_order(db):
    db.create_dataset("legacy_order")
    db._conn.execute(
        """
        INSERT INTO items_legacy_order (id, document, metadata, row_index)
        VALUES ('legacy-a', 'legacy-a', NULL, NULL),
               ('legacy-b', 'legacy-b', NULL, NULL)
        """
    )
    existing_ids = [
        row[0]
        for row in db._conn.execute(
            "SELECT id FROM items_legacy_order ORDER BY rowid"
        ).fetchall()
    ]

    db.insert_items_batch(
        "legacy_order",
        ["new-item"],
        ["new-item"],
        [{}],
    )

    rows = db.get_items_columns("legacy_order", ("id", "row_index"))

    assert [row[0] for row in rows] == [*existing_ids, "new-item"]
    assert [row[1] for row in rows] == list(range(len(rows)))


def test_collection_projection_only_response_omits_core_data(db, monkeypatch):
    _seed_projection_collection(db)
    monkeypatch.setattr(query_module, "get_duckdb_client", lambda: db)

    def fail_if_core_is_loaded(*_args, **_kwargs):
        raise AssertionError("projection-only request loaded shared item data")

    monkeypatch.setattr(db, "get_projection_data", fail_if_core_is_loaded)

    result = query_module.Query().collection(
        name="ds_dense",
        info=None,
        projection_types=["umap_3d"],
        include_core=False,
    )

    assert result is not None
    assert result.ids == []
    assert result.documents == []
    assert result.item_metadata == []
    assert result.available_fields == []
    assert result.umap_3d[0] == pytest.approx([1.0, 2.0, 3.0])
    assert result.umap_3d[1] == pytest.approx([4.0, 5.0, 6.0])
    assert result.projection_signatures["umap_3d"]


def test_collection_core_response_remains_backward_compatible(db, monkeypatch):
    _seed_projection_collection(db)
    monkeypatch.setattr(query_module, "get_duckdb_client", lambda: db)

    result = query_module.Query().collection(
        name="ds_dense",
        info=None,
        projection_types=["umap_3d"],
    )

    assert result is not None
    assert result.ids == ["a", "b"]
    assert result.documents == ["document a", "document b"]
    assert result.item_metadata == [{"group": "one"}, {"group": "two"}]
    assert result.available_fields == ["group"]
    assert result.item_signature == result.projection_signatures["umap_3d"]


def test_partial_projection_has_a_different_membership_signature(db):
    db.create_dataset("partial")
    db.insert_items_batch(
        "partial",
        ["a", "b", "c"],
        ["a", "b", "c"],
        [
            {"row_index": 0},
            {"row_index": 1},
            {"row_index": 2},
        ],
    )
    db.register_vector_collection("partial", "chromadb", "partial_dense", "dense")
    db.insert_projections_batch(
        "partial_dense",
        ["a", "b", "c"],
        "pca_2d",
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
    )
    db.insert_projections_batch(
        "partial_dense",
        ["a", "c"],
        "umap_3d",
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
    )

    complete = db.get_projection_data("partial_dense", "pca_2d")
    partial = db.get_projection_coordinates("partial_dense", "umap_3d")

    assert complete is not None
    assert partial is not None
    assert complete["item_signature"] != partial["item_signature"]


def test_graphql_include_core_argument_exposes_projection_only_contract(db, monkeypatch):
    _seed_projection_collection(db)
    monkeypatch.setattr(query_module, "get_duckdb_client", lambda: db)

    result = schema.execute_sync(
        """
        query ProjectionOnly($includeCore: Boolean = true) {
          collection(
            name: "ds_dense"
            projectionTypes: ["umap_3d"]
            includeCore: $includeCore
          ) {
            ids
            documents
            itemMetadata
            umap3d
            projectionSignatures
          }
        }
        """,
        variable_values={"includeCore": False},
    )

    assert result.errors is None
    payload = result.data["collection"]
    assert payload["ids"] == []
    assert payload["documents"] == []
    assert payload["itemMetadata"] == []
    assert payload["umap3d"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert payload["projectionSignatures"]["umap_3d"]
