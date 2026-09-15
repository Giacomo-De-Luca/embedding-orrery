# Graph dataset utilities

- `contract.py`: `SaeIdentity`, `GraphProvenance`, `GraphManifest`, explicit Arrow
  schemas and `GraphBundleWriter` validation, checksums and atomic immutable export.
- `counting.py`: `EventActivity` stores one bit per feature/event;
  `SparseGraphBuilder` scores exact joint counts and retains mutual top-k neighbors.
  `RankedNeighbors` holds only width × maximum-k indices, counts and scores.
- `source.py`: `DuckDBDocumentSource` audits and fingerprints the read-only corpus
  and sparse source. Missing processing coverage or identity acknowledgment fails.
- `pipeline.py`: `GraphBuildRunner` loads a JSON configuration, reads once, ranks
  once and writes compatible graph variants under the configured resource root.

Full contract, algorithm, limitations and commands:
`documentation/SAE_GRAPH_DATASETS.md` at the repository root.
