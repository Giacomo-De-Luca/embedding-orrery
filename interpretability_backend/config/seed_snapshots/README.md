# Seed snapshot manifests

These JSON files are the source of truth for reproducible Orrery seed exports.
Run the builder from the repository root with `uv run python` and keep the
backend stopped while it reads the live DuckDB and Chroma stores.

- `default.json` rebuilds the small snapshot committed under `resources/seed`.
- `demo.json` builds the larger public-demo payload and can publish it to a
  private Hugging Face Dataset repository.

Every collection always includes its DuckDB dataset, item, and collection
configuration rows. The optional `include` object controls `vectors`,
`projections`, `topics`, and `probes`; defaults are `true`, `true`, `true`, and
`false`, respectively.

`sae_data` entries select an SAE by `model_id` plus `sae_id`. Feature metadata
defaults on, activation examples default off, document activations name the
exported document collections to copy, and `explanation_vector_collection`
must name an exported Chroma collection whose SAE metadata matches.

To publish `demo.json`, set `ORRERY_SEED_REPO_ID` and a write-scoped `HF_TOKEN`,
then run `publish_seed_snapshot.py`. Publication writes `demo.lock.json` beside
the config. Commit that generated lock so Docker and the Hugging Face Space use
the immutable Dataset revision.
