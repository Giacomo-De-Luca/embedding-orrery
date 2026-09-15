"""Configuration-driven build of portable co-occurrence graph variants."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..resource_paths import RESOURCE_DIR
from .contract import NODE_SCHEMA, GraphBundleWriter, GraphManifest, GraphProvenance, SaeIdentity
from .counting import SparseGraphBuilder, Sparsification
from .source import DuckDBDocumentSource


class GraphBuildRunner:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config = json.loads(self.config_path.read_text())
        if self.config.get("schemaVersion") != 1:
            raise ValueError("unsupported graph build configuration version")
        self.identity = SaeIdentity(**self.config["sae"])
        self.policies = [Sparsification(**variant["sparsification"])
                         for variant in self.config["variants"]]
        if not self.policies:
            raise ValueError("at least one graph variant is required")
        graph_ids = [variant["graphId"] for variant in self.config["variants"]]
        if len(set(graph_ids)) != len(graph_ids):
            raise ValueError("duplicate graph IDs")
        if len({(p.minimumJointCount, p.minimumScore) for p in self.policies}) != 1:
            raise ValueError("variants must share support and score thresholds")

    def run(self) -> list[dict]:
        source_config = self.config["source"]
        source_path = RESOURCE_DIR / source_config.get("database", "main.duckdb")
        output = RESOURCE_DIR / self.config.get("outputDirectory", "graphs")
        with DuckDBDocumentSource(source_path, source_config, self.identity) as source:
            activity, labels, coverage, revision = source.read(lambda message: print(message, flush=True))
        print(f"Audited {coverage['sparseRowCount']:,} rows; computing exact NPMI neighbors", flush=True)
        builder = SparseGraphBuilder(activity)
        ranked = builder.rank(max(p.topK for p in self.policies),
                              self.policies[0].minimumJointCount, self.policies[0].minimumScore,
                              lambda done, total: print(f"Ranked {done:,}/{total:,} features", flush=True))
        width = self.identity.width
        ids = [self.identity.feature_id(i) for i in range(width)]
        nodes = pa.Table.from_pydict({
            "id": ids, "index": np.arange(width, dtype=np.uint32), "label": labels,
            "modelId": [self.identity.modelId] * width, "saeId": [self.identity.saeId] * width,
            "featureIndex": np.arange(width, dtype=np.uint32),
            "activeEventCount": builder.counts.astype(np.float64),
        }, schema=NODE_SCHEMA)
        writer = GraphBundleWriter()
        code_hashes = {path.name: writer.checksum(path)
                       for path in sorted(Path(__file__).parent.glob("*.py"))}
        results = []
        for variant, policy in zip(self.config["variants"], self.policies):
            edges = builder.edges(ranked, policy, ids)
            provenance = GraphProvenance(
                source_config["collectionName"], revision, "document", activity.event_count,
                {"aggregation": "stored_document_max", "comparison": ">",
                 "threshold": source_config["threshold"], "countOncePerEvent": True,
                 "historicalInference": self.config["historicalInference"]},
                {"method": "all_stored_source_documents", "additionalTokenTruncation": None,
                 "additionalActivationTopK": None, "sourceCoverage": coverage,
                 "builderSha256": code_hashes,
                 "libraries": {"numpy": np.__version__, "pyarrow": pa.__version__}},
                {"name": "npmi", "formula": "ln(c*N/(a*b)) / -ln(c/N)",
                 "logBase": "e", "constantPairScore": 0, "smoothing": None},
                {"name": "positive_npmi", "formula": "Float32(clip(score, 0, 1))"},
                {**asdict(policy), "tieBreak": ["score_desc", "count_desc", "feature_index_asc"]})
            manifest = GraphManifest(variant["graphId"], "pending", variant["title"],
                                     width, edges.num_rows, self.identity, provenance,
                                     source_config["linkedCollectionNames"])
            path = writer.write(output, nodes, edges, manifest)
            report = writer.verify(path)
            if report["degreeDistribution"]["max"] > policy.topK:
                raise ValueError("mutual neighbor degree bound violated")
            result = {"graphId": manifest.graphId, "manifestPath": str(path / "manifest.json"),
                      "nodeCount": width, "edgeCount": edges.num_rows,
                      "isolatedNodeCount": report["isolatedNodeCount"]}
            results.append(result)
            print(json.dumps(result), flush=True)
        return results
