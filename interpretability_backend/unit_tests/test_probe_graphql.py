"""Probe GraphQL resolvers, executed against the real schema with a fake DB.

Covers the multiclass additions: `ProbeInfo.task/classes/excludedClasses`,
`ProbeScores.predicted`, and the lazy `probeClassProbabilities` query (one
class's probabilities per request, rounded to keep payloads small). Queries
must keep working in the read-only demo.
"""

import asyncio

import pytest

from backend.API import mutations, queries, schema
from backend.API.read_only import READ_ONLY_ENV
from backend.services import probing_service
from backend.services.probing_service import ProbeRunResult

MULTICLASS_CONFIG = {
    "task": "multiclass",
    "classes": ["ANALOGY", "EUPHEMISM", "HYPERBOLE"],
    "excluded_classes": {"RARE": 3},
    "min_class_fraction": 0.01,
}


def _probe_row(kind: str, config: dict | None) -> dict:
    return {
        "collection_name": "col",
        "target_field": "category",
        "kind": kind,
        "config": config,
        "metrics": {"val_accuracy": 0.6},
        "direction": None,
        "scaler_mean": None,
        "scaler_scale": None,
        "intercept": None,
        "artifact_path": None,
        "n_train": 80,
        "n_val": 20,
        "created_at": "2026-10-10 12:00:00",
    }


class FakeProbeDB:
    def list_probes(self, collection_name):
        return [_probe_row("logreg", MULTICLASS_CONFIG), _probe_row("ridge", None)]

    def get_probe_scores(self, collection_name, target_field, kind):
        if kind == "logreg":
            return {
                "item_ids": ["a", "b"],
                "scores": [0.7, 0.55],
                "residuals": [None, None],
                "predicted": [0, 2],
            }
        return {
            "item_ids": ["a", "b"],
            "scores": [1.5, 2.5],
            "residuals": [0.1, None],
            "predicted": [None, None],
        }

    def get_probe_class_probabilities(self, collection_name, target_field, kind, class_index):
        if kind != "logreg" or class_index not in (0, 1, 2):
            return None
        return {"item_ids": ["a", "b"], "probabilities": [0.123456789, None]}


@pytest.fixture(autouse=True)
def fake_db(monkeypatch):
    monkeypatch.setattr(queries, "get_duckdb_client", lambda: FakeProbeDB())


def _execute(query: str, **variables):
    result = asyncio.run(schema.execute(query, variable_values=variables))
    assert result.errors is None, result.errors
    return result.data


PROBES_QUERY = """
query ($c: String!) {
  collectionProbes(collectionName: $c) {
    probes { kind task classes excludedClasses minClassFraction scoreField residualField }
  }
}
"""

SCORES_QUERY = """
query ($c: String!, $f: String!, $k: String!) {
  probeScores(collectionName: $c, targetField: $f, kind: $k) { itemIds scores residuals predicted }
}
"""

CLASS_PROBS_QUERY = """
query ($c: String!, $f: String!, $k: String!, $i: Int!) {
  probeClassProbabilities(collectionName: $c, targetField: $f, kind: $k, classIndex: $i) {
    itemIds probabilities
  }
}
"""


def test_collection_probes_expose_task_and_classes():
    probes = _execute(PROBES_QUERY, c="col")["collectionProbes"]["probes"]
    logreg, ridge = probes
    assert logreg["task"] == "multiclass"
    assert logreg["classes"] == ["ANALOGY", "EUPHEMISM", "HYPERBOLE"]
    assert logreg["excludedClasses"] == {"RARE": 3}
    assert logreg["minClassFraction"] == 0.01
    assert logreg["residualField"] is None
    assert ridge["task"] == "regression"
    assert ridge["classes"] is None


def test_probe_scores_include_predicted_class():
    data = _execute(SCORES_QUERY, c="col", f="category", k="logreg")["probeScores"]
    assert data["predicted"] == [0, 2]
    assert data["residuals"] is None


def test_legacy_scores_have_null_predicted():
    data = _execute(SCORES_QUERY, c="col", f="category", k="ridge")["probeScores"]
    assert data["predicted"] is None
    assert data["residuals"] == [0.1, None]


def test_class_probabilities_are_rounded():
    data = _execute(CLASS_PROBS_QUERY, c="col", f="category", k="logreg", i=1)
    assert data["probeClassProbabilities"] == {
        "itemIds": ["a", "b"],
        "probabilities": [0.1235, None],
    }


def test_invalid_class_index_returns_null():
    data = _execute(CLASS_PROBS_QUERY, c="col", f="category", k="logreg", i=7)
    assert data["probeClassProbabilities"] is None


def test_queries_work_in_read_only_mode(monkeypatch):
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    data = _execute(CLASS_PROBS_QUERY, c="col", f="category", k="logreg", i=0)
    assert data["probeClassProbabilities"] is not None


TRAIN_MUTATION = """
mutation ($input: TrainProbeInput!) {
  trainProbe(input: $input) { error probe { kind task classes excludedClasses createdAt } }
}
"""


class FakeTrainDB:
    def __init__(self, record):
        self.record = record

    def get_probe(self, collection_name, target_field, kind):
        return self.record


def _run_train(monkeypatch, record, error=None):
    monkeypatch.setattr(
        probing_service,
        "train_probe_for_collection",
        lambda config: ProbeRunResult(
            collection_name=config.collection_name,
            target_field=config.target_field,
            kind=config.kind,
            error=error,
        ),
    )
    monkeypatch.setattr(mutations, "get_duckdb_client", lambda: FakeTrainDB(record))
    variables = {"input": {"collectionName": "col", "targetField": "category", "kind": "logreg"}}
    return _execute(TRAIN_MUTATION, **variables)["trainProbe"]


def test_train_probe_reads_the_persisted_row_back(monkeypatch):
    result = _run_train(monkeypatch, _probe_row("logreg", MULTICLASS_CONFIG))
    assert result["error"] is None
    assert result["probe"] == {
        "kind": "logreg",
        "task": "multiclass",
        "classes": ["ANALOGY", "EUPHEMISM", "HYPERBOLE"],
        "excludedClasses": {"RARE": 3},
        "createdAt": "2026-10-10 12:00:00",
    }


def test_train_probe_without_persisted_row_returns_no_probe(monkeypatch):
    assert _run_train(monkeypatch, None)["probe"] is None


def test_failed_training_returns_the_error(monkeypatch):
    result = _run_train(monkeypatch, None, error="too few values")
    assert result == {"error": "too few values", "probe": None}
