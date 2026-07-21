# Config-driven seed snapshots

Seed manifests under `interpretability_backend/config/seed_snapshots/` select
the exact live collections and optional analysis data copied into a portable
DuckDB + Chroma snapshot. The builder resolves each collection's backing
dataset automatically and stages the complete export before replacing an old
snapshot.

## Build manifests

`default.json` rebuilds the small Git-tracked seed. `demo.json` builds the
larger Hugging Face demo snapshot.

```bash
# Stop the backend first.
uv run python -m interpretability_backend.scripts.build_seed_snapshot

uv run python -m interpretability_backend.scripts.build_seed_snapshot \
  --config interpretability_backend/config/seed_snapshots/demo.json
```

Each collection always copies its dataset row, per-dataset items table, and
vector-collection configuration. Its optional `include` object supports:

| Field | Default | Data copied |
|---|---:|---|
| `vectors` | `true` | Chroma vectors and collection metadata |
| `projections` | `true` | coordinates and projection metadata |
| `topics` | `true` | extraction config, topic info, assignments |
| `probes` | `false` | probe definitions and per-item scores |

Disabled projections/topics also clear the corresponding collection flags in
the generated database.

## Optional SAE data

`sae_data` selects shared feature data by `(model_id, sae_id)`:

- `features` defaults to `true`.
- `activation_examples` defaults to `false` and requires features.
- `document_activations` lists exported document collections whose sparse
  activation rows should be included.
- `explanation_vector_collection` names a normal exported collection. Its
  Chroma metadata must match the selected model and SAE.

The builder rejects unknown keys, duplicate selections, missing collections,
invalid SAE references, empty required vector collections, and mismatched
explanation metadata before replacing an existing snapshot.

## Integrity and atomic replacement

Exports are written into a sibling staging directory. After both stores are
complete, `snapshot-manifest.json` records the source Git commit, source config
hash, row/vector counts, file sizes, and SHA-256 checksums. Verification runs
before the staging directory atomically replaces the previous output. A failed
or interrupted build leaves the last good snapshot untouched.

The committed seed's Chroma store is copied to temporary storage before the
build and remains the fallback for a selected live collection that exists but
has no production vectors.

## Private Hugging Face Dataset publication

`demo.json` reads its repository ID from `ORRERY_SEED_REPO_ID`. Set that plus a
write-scoped token and publish an already-built, verified snapshot:

```bash
export ORRERY_SEED_REPO_ID=<owner/private-seed-repo>
export HF_TOKEN=hf_...
uv run python -m interpretability_backend.scripts.publish_seed_snapshot \
  --config interpretability_backend/config/seed_snapshots/demo.json
```

The publisher creates the private Dataset repository when needed, verifies its
configured visibility, retries transient uploads with bounded backoff, replaces
only `snapshots/demo/**`, and writes `demo.lock.json` beside the config. The lock
contains the immutable Hub commit and snapshot-manifest checksum; it never
contains a token. Commit the lock to activate automatic Space deployment.

The demo Docker build reads that lock, mounts the Space's read-only
`HF_SEED_TOKEN` as a BuildKit secret, downloads the exact Dataset revision, and
verifies every checksum before baking the seed into the image. Until the first
lock exists, local demo builds use the small committed seed as a migration
fallback; the automated Space job remains disabled so the current public demo
is not downgraded accidentally.
