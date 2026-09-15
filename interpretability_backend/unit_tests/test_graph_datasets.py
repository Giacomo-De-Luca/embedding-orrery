"""Dataset contract and counting acceptance tests; no model/network needed."""

import json
from dataclasses import replace

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from backend.utils.graph_datasets.contract import (
    EDGE_SCHEMA,
    NODE_SCHEMA,
    GraphBundleWriter,
    GraphManifest,
    GraphProvenance,
    SaeIdentity,
)
from backend.utils.graph_datasets.counting import EventActivity, SparseGraphBuilder, Sparsification
from backend.utils.graph_datasets.source import DuckDBDocumentSource


IDENTITY = SaeIdentity("model", "sae", 4)


def fixture_tables():
    # Feature 2 is at rendering index 0. Never equate feature identity and row.
    features = [2, 0, 3, 1]
    ids = [f"model::sae::{i}" for i in features]
    nodes = pa.Table.from_pydict({
        "id": ids, "index": range(4), "label": ["C", "A", "D", "B"],
        "modelId": ["model"] * 4, "saeId": ["sae"] * 4,
        "featureIndex": features, "activeEventCount": [2., 3., 0., 2.],
    }, schema=NODE_SCHEMA)
    edges = pa.Table.from_pydict({
        "source": [ids[0]], "target": [ids[1]], "sourceIndex": [0], "targetIndex": [1],
        "weight": [0.5], "cooccurrenceCount": [2.], "score": [0.5],
    }, schema=EDGE_SCHEMA)
    provenance = GraphProvenance("fixture", "sha256:fixture", "document", 5,
        {"threshold": 0}, {"sourceCoverage": {"processedEventCount": 5}},
        {"name": "npmi"}, {"name": "identity"}, {"topK": 2})
    manifest = GraphManifest("fixture", "revision", "Fixture", 4, 1, IDENTITY, provenance)
    return nodes, edges, manifest


def changed(table, name, values, dtype=None):
    field = table.schema.field(name)
    return table.set_column(table.schema.get_field_index(name),
        pa.field(name, dtype or field.type, nullable=False), pa.array(values, dtype or field.type))


def test_identity_and_isolates():
    nodes, edges, manifest = fixture_tables()
    report = GraphBundleWriter.validate(nodes, edges, manifest)
    assert nodes["id"][0].as_py() == "model::sae::2"
    assert report["isolatedNodeCount"] == 2
    assert report["inactiveNodeCount"] == 1


@pytest.mark.parametrize("column,values", [
    ("id", ["model::sae::2"] * 4), ("index", [0, 1, 1, 3]),
    ("featureIndex", [2, 0, 3, 2]), ("modelId", ["wrong"] * 4),
    ("label", [None, "A", "B", "C"]),
    ("activeEventCount", [2., 3., 0., 6.]),
    ("activeEventCount", [2., 3., 0., 1.5]),
    ("activeEventCount", [2., 3., 0., float("nan")]),
    ("activeEventCount", [2., 3., 0., float(2**53)]),
])
def test_bad_nodes(column, values):
    nodes, edges, manifest = fixture_tables()
    with pytest.raises(ValueError):
        GraphBundleWriter.validate(changed(nodes, column, values), edges, manifest)


@pytest.mark.parametrize("column,values", [
    ("source", ["absent"]), ("target", ["model::sae::1"]),
    ("sourceIndex", [1]), ("targetIndex", [4]),
    ("weight", [0.]), ("weight", [1.01]), ("weight", [float("inf")]),
    ("score", [float("nan")]), ("cooccurrenceCount", [0.]),
    ("cooccurrenceCount", [3.]), ("cooccurrenceCount", [1.5]),
])
def test_bad_edges(column, values):
    nodes, edges, manifest = fixture_tables()
    with pytest.raises(ValueError):
        GraphBundleWriter.validate(nodes, changed(edges, column, values), manifest)


def test_duplicate_pairs_and_types():
    nodes, edges, manifest = fixture_tables()
    with pytest.raises(ValueError, match="duplicate"):
        GraphBundleWriter.validate(nodes, pa.concat_tables([edges, edges]),
                                   replace(manifest, edgeCount=2))
    with pytest.raises(ValueError, match="type"):
        GraphBundleWriter.validate(nodes, changed(edges, "weight", [.5], pa.float64()), manifest)


def test_coordinates_and_empty_tables():
    nodes, _, manifest = fixture_tables()
    edges = pa.Table.from_batches([], schema=EDGE_SCHEMA)
    manifest = replace(manifest, edgeCount=0)
    assert GraphBundleWriter.validate(nodes, edges, manifest)["isolatedNodeCount"] == 4
    for axis in "xyz":
        nodes = nodes.append_column(pa.field(axis, pa.float32(), nullable=False),
                                    pa.array([0.] * 4, pa.float32()))
        if axis != "z":
            with pytest.raises(ValueError, match="coordinate"):
                GraphBundleWriter.validate(nodes, edges, replace(manifest, positions="xyz"))
    GraphBundleWriter.validate(nodes, edges, replace(manifest, positions="xyz"))
    with pytest.raises(ValueError):
        GraphBundleWriter.validate(changed(nodes, "z", [float("inf")] * 4), edges,
                                   replace(manifest, positions="xyz"))


def test_document_and_token_semantics_with_empty_events():
    tokens = [("doc1", 0, [0, 0]), ("doc1", 1, [1]), ("doc2", 0, [])]
    documents = EventActivity.from_token_events(4, tokens, "document")
    per_token = EventActivity.from_token_events(4, tokens, "token")
    assert documents.event_count == 2
    assert per_token.event_count == 3
    assert documents.joint_counts(0)[1] == 1
    assert per_token.joint_counts(0)[1] == 0
    assert documents.active_counts().tolist() == [1, 1, 0, 0]


def test_exact_counts_and_sparse_degree_bound():
    rng = np.random.default_rng(42)
    dense = rng.random((350, 17)) < .3
    dense[:, 1] = dense[:, 0]
    dense[:, 16] = False
    events, features = np.nonzero(dense)
    activity = EventActivity(17, 350)
    activity.add(events, features)
    expected = dense.astype(np.int64).T @ dense.astype(np.int64)
    for row in range(17):
        np.testing.assert_array_equal(activity.joint_counts(row), expected[row])
    policy = Sparsification(topK=2, minimumJointCount=1, minimumScore=0.0)
    builder = SparseGraphBuilder(activity)
    ranked = builder.rank(2, 1, 0.0)
    edges = builder.edges(ranked, policy, [str(i) for i in range(17)])
    degrees = np.bincount(np.r_[edges["sourceIndex"].to_numpy(),
                                edges["targetIndex"].to_numpy()].astype(int), minlength=17)
    assert degrees.max() <= 2
    assert degrees[16] == 0
    assert all(s < t for s, t in zip(edges["sourceIndex"].to_pylist(),
                                    edges["targetIndex"].to_pylist()))
    # Independently compute NPMI for every retained edge.
    counts = dense.sum(axis=0)
    for edge in edges.to_pylist():
        i, j = edge["sourceIndex"], edge["targetIndex"]
        joint = expected[i, j]
        score = np.log(joint * 350 / (counts[i] * counts[j])) / -np.log(joint / 350)
        assert edge["cooccurrenceCount"] == joint
        assert edge["score"] == pytest.approx(score)


def test_no_uint8_overflow_and_invalid_events():
    activity = EventActivity(2, 300)
    activity.add(np.arange(300), np.zeros(300, dtype=int))
    assert activity.active_counts()[0] == 300
    with pytest.raises(ValueError):
        activity.add(np.array([300]), np.array([0]))
    with pytest.raises(ValueError):
        activity.add(np.array([1]), np.array([2]))


def test_deterministic_immutable_export_and_corruption(tmp_path):
    nodes, edges, manifest = fixture_tables()
    writer = GraphBundleWriter()
    first = writer.write(tmp_path, nodes, edges, manifest)
    second = writer.write(tmp_path, nodes, edges, manifest)
    assert first == second
    loaded = writer.verify(first)
    assert loaded["nodeCount"] == 4
    assert pa.ipc.open_file(first / "edges.arrow").read_all().schema == EDGE_SCHEMA
    (first / "edges.arrow").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        writer.verify(first)
    with pytest.raises(ValueError):
        writer.write(tmp_path, nodes, edges, manifest)


def source_fixture(path):
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE datasets(name VARCHAR, extra_metadata JSON)")
        conn.execute("INSERT INTO datasets VALUES ('docs', ?)",
                     [json.dumps({"sae_model_id": "model", "sae_id": "sae"})])
        conn.execute("CREATE TABLE vector_collections(collection_name VARCHAR, dataset_name VARCHAR)")
        conn.execute("INSERT INTO vector_collections VALUES ('docs', 'docs')")
        conn.execute("CREATE TABLE items_docs(id VARCHAR PRIMARY KEY, document VARCHAR)")
        conn.execute("INSERT INTO items_docs VALUES ('a', 'A'), ('b', 'B')")
        conn.execute("CREATE TABLE sae_document_activations(collection_name VARCHAR, item_id VARCHAR, feature_index INTEGER, activation FLOAT)")
        conn.execute("INSERT INTO sae_document_activations VALUES ('docs','a',0,1), ('docs','b',1,1)")
        conn.execute("CREATE TABLE sae_features(model_id VARCHAR, sae_id VARCHAR, feature_index INTEGER, label VARCHAR)")
        conn.execute("INSERT INTO sae_features VALUES ('model','sae',0,'A')")


def test_source_coverage_and_provenance_guards(tmp_path):
    path = tmp_path / "source.duckdb"
    source_fixture(path)
    config = {"collectionName": "docs", "expectedDocumentCount": 2,
              "expectedSparseRowCount": 2, "allowLegacyProvenance": True,
              "threshold": 0., "linkedCollectionNames": []}
    with DuckDBDocumentSource(path, config, IDENTITY) as source:
        activity, labels, coverage, revision = source.read()
    assert activity.event_count == 2
    assert labels == ["A", "Feature 1", "Feature 2", "Feature 3"]
    assert coverage["processedEventCount"] == 2
    assert revision.startswith("sha256:")
    with pytest.raises(ValueError, match="legacy"):
        with DuckDBDocumentSource(path, {**config, "allowLegacyProvenance": False}, IDENTITY) as source:
            source.read()
    with duckdb.connect(str(path)) as conn:
        conn.execute("DELETE FROM sae_document_activations WHERE item_id='b'")
    with pytest.raises(ValueError, match="coverage|rows"):
        with DuckDBDocumentSource(path, config, IDENTITY) as source:
            source.read()
