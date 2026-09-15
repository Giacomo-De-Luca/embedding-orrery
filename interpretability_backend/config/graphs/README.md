# SAE graph build configurations

`acl_document.json` defines one audited document source and two graph variants.
The `sae` identity includes the full dictionary width. `source` pins corpus and
sparse-row counts, activity threshold and compatible feature collections.
`historicalInference` distinguishes recorded identity from unavailable run details.
Each `variants` entry configures mutual top-k edges, support and NPMI cutoffs.

`allowLegacyProvenance` requires explicit acknowledgment of missing historical
inference details. Counts protect against accidentally building from the demo's
top-256 activation seed. All runtime paths resolve under `ORRERY_RESOURCE_DIR`
(the normal local backend resources directory when unset).

Run from the project root with `uv run python -m
interpretability_backend.scripts.build_sae_graphs`. The only CLI option is
`--config` to select a different JSON file.
