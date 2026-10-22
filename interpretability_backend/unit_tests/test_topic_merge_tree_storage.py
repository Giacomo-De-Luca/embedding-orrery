"""DuckDB storage behind topic merge trees and safe extraction activation.

Covers the ``topic_merge_trees`` table (density trees of an extraction), the
activate-last flow used by topic sync (``create_topic_extraction(activate=False)``
then ``activate_topic_extraction``), deleting a half-written extraction, and
the assignments-joined-with-documents read the reduction service loads.
"""

import numpy as np
import pytest

from backend.clients.duckdb_client import DuckDBClient

COLLECTION = "ds_dense"
TREE = {
    "leaf_ids": [0, 1, 2],
    "leaf_sizes": [5, 3, 2],
    "children": [[1, 2], [0, 3]],
    "heights": [0.5, 2.0],
    "noise_item_ids": ["n1", "n2"],
    "noise_merges": [0, 1],
}


def _seed(db: DuckDBClient) -> str:
    db.create_dataset("ds")
    db.insert_items_batch("ds", ["a", "b", "c", "d"], ["doc a", "doc b", "doc c", None], [{}] * 4)
    db.register_vector_collection("ds", "chromadb", COLLECTION, "dense")
    ext = db.create_topic_extraction(COLLECTION, "ds")
    db.insert_topic_assignments_batch(
        ext,
        [
            {"item_id": "c", "topic_id": 1, "topic_label": "C"},
            {"item_id": "a", "topic_id": 0, "topic_label": "A"},
            {"item_id": "d", "topic_id": -1, "topic_label": "Unclustered"},
        ],
    )
    return ext


def _store(db: DuckDBClient, ext: str, linkage: str = "density") -> None:
    db.replace_topic_merge_tree(ext, linkage, **TREE)


class TestMergeTreeRows:
    def test_roundtrip(self, db):
        ext = _seed(db)
        _store(db, ext)

        row = db.get_topic_merge_tree(ext, "density")
        assert row["leaf_ids"].tolist() == [0, 1, 2]
        assert row["leaf_sizes"].tolist() == [5, 3, 2]
        assert row["children"].tolist() == [[1, 2], [0, 3]]
        np.testing.assert_allclose(row["heights"], [0.5, 2.0])
        assert row["noise_item_ids"] == ["n1", "n2"]
        assert row["noise_merges"].tolist() == [0, 1]

    def test_replace_overwrites_the_same_linkage(self, db):
        ext = _seed(db)
        _store(db, ext)
        db.replace_topic_merge_tree(
            ext, "density", leaf_ids=[0, 1], leaf_sizes=[1, 1], children=[[0, 1]], heights=[1.0]
        )

        row = db.get_topic_merge_tree(ext, "density")
        assert row["leaf_ids"].tolist() == [0, 1]
        assert row["noise_item_ids"] == []

    def test_missing_tree_is_none(self, db):
        ext = _seed(db)
        assert db.get_topic_merge_tree(ext, "density") is None

    def test_deleting_the_collection_removes_its_trees(self, db):
        ext = _seed(db)
        _store(db, ext)

        assert db.delete_vector_collection(COLLECTION)
        assert db.get_topic_merge_tree(ext, "density") is None


class TestActivation:
    def test_an_inactive_extraction_leaves_the_active_one_alone(self, db):
        active = _seed(db)
        pending = db.create_topic_extraction(COLLECTION, "ds", activate=False)

        assert pending != active
        assert db.get_active_extraction_id(COLLECTION) == active

    def test_activation_makes_exactly_one_extraction_active(self, db):
        first = _seed(db)
        second = db.create_topic_extraction(COLLECTION, "ds", activate=False)

        db.activate_topic_extraction(COLLECTION, second)

        assert db.get_active_extraction_id(COLLECTION) == second
        active = db._exec(
            "SELECT id FROM topic_extractions WHERE collection_name = ? AND is_active",
            [COLLECTION],
        ).fetchall()
        assert active == [(second,)]
        assert first != second

    def test_deleting_an_extraction_removes_its_rows(self, db):
        active = _seed(db)
        pending = db.create_topic_extraction(COLLECTION, "ds", activate=False)
        db.insert_topic_info_batch(pending, [{"topic_id": 0, "label": "A", "count": 1}])
        db.insert_topic_assignments_batch(pending, [{"item_id": "a", "topic_id": 0}])
        db.replace_topic_centroids(
            pending, [{"level": "topic", "topic_id": 0, "member_count": 1, "centroid": [1.0]}]
        )
        _store(db, pending)

        db.delete_topic_extraction(pending)

        for table in ("topic_info", "topic_assignments", "topic_centroids", "topic_merge_trees"):
            count = db._exec(
                f"SELECT count(*) FROM {table} WHERE extraction_id = ?", [pending]
            ).fetchone()[0]
            assert count == 0, table
        assert db._exec(
            "SELECT count(*) FROM topic_extractions WHERE id = ?", [pending]
        ).fetchone() == (0,)
        assert db.get_active_extraction_id(COLLECTION) == active


class TestAssignmentDocuments:
    def test_joins_documents_in_item_order(self, db):
        ext = _seed(db)

        rows = db.get_assignment_documents(ext, "ds")

        assert [r[0] for r in rows] == ["a", "c", "d"]  # item "b" has no assignment
        assert rows[0] == ("a", "doc a", 0, "A", None, None)
        assert rows[2][1] is None  # a NULL document stays NULL

    def test_unknown_extraction_has_no_rows(self, db):
        _seed(db)
        assert db.get_assignment_documents("nope", "ds") == []


@pytest.mark.parametrize("linkage", ["density", "average"])
def test_trees_are_keyed_by_linkage(db, linkage):
    ext = _seed(db)
    _store(db, ext, linkage)
    other = "average" if linkage == "density" else "density"
    assert db.get_topic_merge_tree(ext, linkage) is not None
    assert db.get_topic_merge_tree(ext, other) is None
