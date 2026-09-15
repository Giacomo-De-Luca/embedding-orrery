"""Arrow contract, content-addressed manifests, and immutable bundle export."""

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa


NODE_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("index", pa.uint32(), nullable=False),
    pa.field("label", pa.string(), nullable=False),
    pa.field("modelId", pa.string(), nullable=False),
    pa.field("saeId", pa.string(), nullable=False),
    pa.field("featureIndex", pa.uint32(), nullable=False),
    pa.field("activeEventCount", pa.float64(), nullable=False),
])
EDGE_SCHEMA = pa.schema([
    pa.field("source", pa.string(), nullable=False),
    pa.field("target", pa.string(), nullable=False),
    pa.field("sourceIndex", pa.uint32(), nullable=False),
    pa.field("targetIndex", pa.uint32(), nullable=False),
    pa.field("weight", pa.float32(), nullable=False),
    pa.field("cooccurrenceCount", pa.float64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
JS_MAX_SAFE_INTEGER = 2**53 - 1


@dataclass(frozen=True)
class SaeIdentity:
    modelId: str
    saeId: str
    width: int

    def feature_id(self, feature_index: int) -> str:
        """Same identity as the frontend's existing steeringFeatureKey helper."""
        return f"{self.modelId}::{self.saeId}::{feature_index}"


@dataclass(frozen=True)
class GraphProvenance:
    sourceDatasetId: str
    sourceRevision: str
    eventUnit: str
    eventCount: int
    activationRule: dict[str, Any]
    sampling: dict[str, Any]
    associationMetric: dict[str, Any]
    weightNormalization: dict[str, Any]
    sparsification: dict[str, Any]


@dataclass(frozen=True)
class GraphManifest:
    graphId: str
    revision: str
    title: str
    nodeCount: int
    edgeCount: int
    sae: SaeIdentity
    provenance: GraphProvenance
    linkedCollectionNames: list[str] = field(default_factory=list)
    nodes: dict[str, str] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    schemaVersion: int = 1
    nodeKind: str = "sae_feature"
    directed: bool = False
    positions: str = "none"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphManifest":
        return cls(**{**value, "sae": SaeIdentity(**value["sae"]),
                      "provenance": GraphProvenance(**value["provenance"])})


class GraphBundleWriter:
    """Validate columnar data, then atomically publish graphId/revision bundles."""

    @staticmethod
    def json_bytes(value: Any) -> bytes:
        return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n").encode("utf-8")

    @staticmethod
    def checksum(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @classmethod
    def revision(cls, manifest: dict[str, Any]) -> str:
        return hashlib.sha256(cls.json_bytes({k: v for k, v in manifest.items()
                                              if k != "revision"})).hexdigest()

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @classmethod
    def _schema(cls, table: pa.Table, schema: pa.Schema) -> None:
        cls._require(len(set(table.column_names)) == table.num_columns, "duplicate columns")
        for expected in schema:
            cls._require(expected.name in table.column_names, f"missing {expected.name}")
            actual = table.schema.field(expected.name)
            cls._require(actual.type == expected.type, f"wrong Arrow type: {expected.name}")
            cls._require(table[expected.name].null_count == 0, f"null {expected.name}")

    @classmethod
    def _counts(cls, values: np.ndarray, upper: int, name: str) -> None:
        cls._require(bool(np.all(np.isfinite(values)) and np.all(values >= 0)
                          and np.all(values <= min(upper, JS_MAX_SAFE_INTEGER))
                          and np.all(values == np.floor(values))), f"invalid {name} counts")

    @staticmethod
    def distribution(values: np.ndarray) -> dict[str, Any]:
        if not values.size:
            return {"count": 0, "min": None, "p25": None, "median": None,
                    "p75": None, "p95": None, "max": None, "mean": None}
        quantiles = np.quantile(values, [0, .25, .5, .75, .95, 1])
        return {"count": int(values.size), **dict(zip(
            ["min", "p25", "median", "p75", "p95", "max"], quantiles.tolist())),
            "mean": float(values.mean())}

    @classmethod
    def validate(cls, nodes: pa.Table, edges: pa.Table,
                 manifest: GraphManifest) -> dict[str, Any]:
        check = cls._require
        cls._schema(nodes, NODE_SCHEMA)
        cls._schema(edges, EDGE_SCHEMA)
        check(manifest.schemaVersion == 1 and manifest.nodeKind == "sae_feature"
              and manifest.directed is False, "unsupported manifest kind/version")
        check(isinstance(manifest.sae.width, int) and not isinstance(manifest.sae.width, bool)
              and 0 < manifest.sae.width <= 2**32, "invalid SAE width")
        for identifier in (manifest.sae.modelId, manifest.sae.saeId):
            check(isinstance(identifier, str) and bool(identifier) and "::" not in identifier,
                  "invalid SAE identity component")
        check(nodes.num_rows == manifest.nodeCount == manifest.sae.width,
              "node count must equal full SAE dictionary width")
        check(edges.num_rows == manifest.edgeCount, "edge count mismatch")
        check(manifest.provenance.eventUnit in ("document", "token"), "invalid event unit")
        event_count = manifest.provenance.eventCount
        check(type(event_count) is int and 0 <= event_count <= JS_MAX_SAFE_INTEGER,
              "invalid eventCount")
        for value in (manifest.provenance.sourceDatasetId, manifest.provenance.sourceRevision):
            check(isinstance(value, str) and bool(value), "missing source provenance")
        check(len(set(manifest.linkedCollectionNames)) == len(manifest.linkedCollectionNames),
              "duplicate linked collections")
        ids = nodes["id"].to_pylist()
        check(len(set(ids)) == len(ids), "duplicate node IDs")
        check(np.array_equal(nodes["index"].to_numpy(), np.arange(nodes.num_rows)),
              "node index must equal row")
        features = nodes["featureIndex"].to_numpy()
        check(np.array_equal(np.sort(features), np.arange(manifest.sae.width)),
              "feature indices must enumerate full dictionary exactly once")
        check(all(value == manifest.sae.modelId for value in nodes["modelId"].to_pylist()),
              "model identity mismatch")
        check(all(value == manifest.sae.saeId for value in nodes["saeId"].to_pylist()),
              "SAE identity mismatch")
        check(all(node_id == manifest.sae.feature_id(int(feature))
                  for node_id, feature in zip(ids, features)), "canonical node identity mismatch")
        counts = nodes["activeEventCount"].to_numpy()
        cls._counts(counts, event_count, "node")
        coordinates = set(nodes.column_names) & {"x", "y", "z"}
        check(manifest.positions in ("none", "xyz"), "invalid coordinate declaration")
        check(coordinates == (set() if manifest.positions == "none" else {"x", "y", "z"}),
              "coordinate columns must be complete or absent")
        for coordinate in coordinates:
            cls._schema(nodes, pa.schema([pa.field(coordinate, pa.float32())]))
            check(bool(np.isfinite(nodes[coordinate].to_numpy()).all()), "nonfinite coordinate")
        sources = edges["sourceIndex"].to_numpy().astype(np.int64)
        targets = edges["targetIndex"].to_numpy().astype(np.int64)
        check(bool(np.all(sources < targets)), "canonical edge ordering/self-link violation")
        check(bool(np.all(targets < nodes.num_rows)), "dangling endpoint")
        pairs = sources.astype(np.uint64) * np.uint64(nodes.num_rows) + targets.astype(np.uint64)
        check(len(np.unique(pairs)) == len(pairs), "duplicate undirected pairs")
        check(all(value == ids[index] for value, index in zip(edges["source"].to_pylist(), sources)),
              "source ID/index mismatch")
        check(all(value == ids[index] for value, index in zip(edges["target"].to_pylist(), targets)),
              "target ID/index mismatch")
        joints = edges["cooccurrenceCount"].to_numpy()
        cls._counts(joints, event_count, "edge")
        check(bool(np.all(joints > 0) and np.all(joints <= np.minimum(counts[sources], counts[targets]))),
              "impossible joint counts")
        weights = edges["weight"].to_numpy()
        check(bool(np.all(np.isfinite(weights)) and np.all(weights > 0) and np.all(weights <= 1)),
              "weights must be finite and in (0,1]")
        scores = edges["score"].to_numpy()
        check(bool(np.all(np.isfinite(scores))), "nonfinite association score")
        degrees = np.bincount(np.concatenate([sources, targets]), minlength=nodes.num_rows)
        return {"valid": True, "nodeCount": nodes.num_rows, "edgeCount": edges.num_rows,
                "eventCount": event_count, "isolatedNodeCount": int(np.count_nonzero(degrees == 0)),
                "inactiveNodeCount": int(np.count_nonzero(counts == 0)),
                "degreeDistribution": cls.distribution(degrees),
                "weightDistribution": cls.distribution(weights),
                "scoreDistribution": cls.distribution(scores),
                "activeEventCountDistribution": cls.distribution(counts),
                "cooccurrenceCountDistribution": cls.distribution(joints),
                "sourceCoverage": manifest.provenance.sampling.get("sourceCoverage", {}),
                "contractChecks": {key: True for key in (
                    "arrowTypes", "nonNullColumns", "fullDictionary", "canonicalIdentity",
                    "denseRenderingIndices", "endpointIdentity", "uniqueUndirectedPairs",
                    "safeIntegerCounts", "countBounds", "finiteScores", "boundedWeights",
                    "completeOrAbsentCoordinates")}}

    def write(self, root: Path, nodes: pa.Table, edges: pa.Table,
              manifest: GraphManifest) -> Path:
        self._require(bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", manifest.graphId)),
                      "graphId must be a path-safe slug")
        report = self.validate(nodes, edges, manifest)
        parent = root / manifest.graphId
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
        try:
            artifacts = {}
            for name, table in (("nodes", nodes), ("edges", edges)):
                path = staging / f"{name}.arrow"
                with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table.combine_chunks(), max_chunksize=65536)
                artifacts[name] = {"path": path.name, "sha256": self.checksum(path)}
            resolved = replace(manifest, **artifacts)
            resolved = replace(resolved, revision=self.revision(asdict(resolved)))
            (staging / "manifest.json").write_bytes(self.json_bytes(asdict(resolved)))
            (staging / "validation-report.json").write_bytes(self.json_bytes(report))
            (staging / "README.md").write_text(self.readme(resolved, report), encoding="utf-8")
            destination = parent / resolved.revision
            self.verify(staging)
            if destination.exists():
                self.verify(destination)
                for path in staging.iterdir():
                    self._require(path.read_bytes() == (destination / path.name).read_bytes(),
                                  "immutable bundle differs; refusing to overwrite")
            else:
                staging.rename(destination)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def verify(cls, directory: Path) -> dict[str, Any]:
        value = json.loads((directory / "manifest.json").read_text())
        cls._require(value["revision"] == cls.revision(value), "manifest revision mismatch")
        tables = {}
        for name in ("nodes", "edges"):
            artifact = value[name]
            cls._require(artifact["path"] == f"{name}.arrow", "unexpected artifact path")
            path = directory / artifact["path"]
            cls._require(cls.checksum(path) == artifact["sha256"], "artifact checksum mismatch")
            with pa.memory_map(str(path), "r") as source:
                tables[name] = pa.ipc.open_file(source).read_all()
        report = cls.validate(tables["nodes"], tables["edges"], GraphManifest.from_dict(value))
        cls._require(json.loads((directory / "validation-report.json").read_text()) == report,
                     "validation report mismatch")
        return report

    @staticmethod
    def readme(manifest: GraphManifest, report: dict[str, Any]) -> str:
        provenance = manifest.provenance
        return f"""# {manifest.title}

{manifest.nodeCount:,} SAE feature nodes; {manifest.edgeCount:,} undirected edges.
{provenance.eventCount:,} {provenance.eventUnit} events from `{provenance.sourceDatasetId}`.
{report['isolatedNodeCount']:,} isolated nodes, including {report['inactiveNodeCount']:,} inactive features.

## Files and identity

- `manifest.json`: portable v1 metadata, source provenance, policy and SHA-256 checksums.
- `nodes.arrow`: Arrow IPC feature dictionary. `index` is the row address;
  `featureIndex` is SAE identity. `id` is `modelId::saeId::featureIndex`.
- `edges.arrow`: Arrow IPC canonical pairs (`sourceIndex < targetIndex`),
  event counts, association scores and positive Float32 attraction weights.
- `validation-report.json`: coverage, contract checks and graph distributions.

No coordinates are supplied; a renderer may initialize its force simulation.
Use the explicit endpoint indices without another indexing pass.

## Derivation

Two nodes connect when both features are active somewhere in the same document.
Counts refer to stored max-pooled activations passing the manifest threshold.
Each feature counts once per event; no additional top-k activation truncation is applied.
The score is NPMI = log(c*N/(a*b)) / -log(c/N), using natural logs.
Pairs with c=N are assigned score zero (no variation), then excluded.
Positive scores are clipped to [0,1] and cast to Float32 for rendering.
Edges must pass the configured support/score cutoffs and mutual top-k selection.
Ties prefer larger joint counts, then smaller feature indices.

## Provenance limits

This legacy source has no immutable inference-run log. The model/SAE identity
comes from dataset metadata, the full feature dictionary and the configured
acknowledgment. Historical checkpoint commits, dtype, token exclusions and
truncation were not recorded and must not be inferred from current code/cache.
The manifest fingerprints the actual stored corpus and activation rows.
All source documents have stored rows; zero-activation source documents or
failed/unprocessed documents would require a separate processing ledger.
Events made empty by this build's threshold are retained in eventCount.
This graph establishes document association, not same-token activation or causality.

Source revision: `{provenance.sourceRevision}`.
Bundle revision: `{manifest.revision}`.
"""
