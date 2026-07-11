# HuggingFace Space Demo (read-only)

A public, read-only demo of Orrery ships as a single-container HuggingFace
Docker Space. Demo = Explore page only, three seeded collections (emotion,
xkcd colors, and the 13,980-abstract `acl_abstracts_emnlp_findings` EMNLP
collection with 60 LLM-labeled topics).

## Architecture

One image (root `Dockerfile` — separate from the compose stack, which is
unchanged), one public port:

```
nginx :7860 ── /graphql, /health ──▶ uvicorn :8000  (FastAPI + Strawberry)
          └── everything else ─────▶ node server.js :3000  (Next standalone)
```

- `deploy/hf-space/start.sh` launches uvicorn + node in the background and
  `exec`s nginx as PID 1.
- `deploy/hf-space/nginx.conf` is non-root friendly (pid/temp under `/tmp`);
  `/upload` is deliberately unrouted. WS upgrade headers are configured on
  `/graphql` even though demo mode never opens a socket (insurance for
  duplicated Spaces with writes enabled).
- HF Spaces runs containers as uid 1000: the image creates `user`, points
  `ORRERY_RESOURCE_DIR=/home/user/data` and `HF_HOME=/home/user/hf` at
  writable paths. Space restarts wipe `/home/user/data` → the seed bootstrap
  re-seeds a pristine demo (intended). Free-tier Spaces sleep after ~48 h
  idle; cold start re-seeds too.
- `all-MiniLM-L6-v2` is pre-baked into an image layer so semantic search on
  the emotion collection works offline with a fast first query.
- Linux installs **CPU-only torch** (`[tool.uv.index]` `pytorch-cpu` +
  `[tool.uv.sources]` marker in `pyproject.toml`) — no CUDA payload. macOS
  dev resolution is unaffected; the compose image benefits too.

## The two flags

| Flag | Layer | Effect |
|---|---|---|
| `ORRERY_READ_ONLY=1` | backend, runtime env | `ReadOnlyExtension` (`backend/API/read_only.py`) short-circuits **all GraphQL mutations** before execution; `main.py` skips mounting `/upload`. Read per-operation, so a Space **Variable** `ORRERY_READ_ONLY=0` re-enables writes on a duplicated Space without a rebuild. Tested in `unit_tests/test_read_only.py`. |
| `NEXT_PUBLIC_DEMO_MODE=1` | frontend, **build-time** ARG | Cosmetic layer (enforcement is server-side): hides SAE/Collections nav tabs (`PageNav`), redirects `/collections` + `/sae` → `/` (`next.config.ts`), hides SaveColorDefaultButton, ProbeSection, and the scatter right-click SAE menu. Baked into the JS bundle — changing it requires an image rebuild. |

The frontend is built with `NEXT_PUBLIC_GRAPHQL_URL=/graphql` (same-origin
through nginx). The WS URL keeps its localhost default: every
subscription-using component is hidden in demo mode and `graphql-ws` connects
lazily, so no socket is ever opened.

## Demo seed (`resources/seed_demo/`, ~313 MB)

Built by the (now parameterized) seed script — run **with the backend
stopped**:

```bash
uv run python -m interpretability_backend.scripts.build_seed_snapshot \
    --collections emotion xkcd_hilbert_gemini acl_abstracts_emnlp_findings \
    --datasets emotion xkcd_hilbert acl_abstracts_emnlp_findings \
    --output interpretability_backend/resources/seed_demo
```

No flags → the committed 25 MB seed rebuilds (emotion + xkcd). Script notes:

- Copies use `INSERT INTO … BY NAME` — production tables gain columns over
  time via guarded ALTERs (e.g. `topic_extractions.quality_metrics`), so
  positional `SELECT *` copies break against a freshly created schema.
- Chroma vectors copy in 5k batches (Chroma caps one `add()` at ~5.4k).
- If a collection is **empty in the production Chroma store**, the script
  falls back to the committed seed as vector source (observed live: the
  production `emotion` collection has 0 vectors; its vectors survive only in
  the committed seed). The fallback reads from a **temp copy** of the
  committed seed snapshotted before the output dir is wiped — so the default
  no-flags rebuild works even though it overwrites its own fallback source,
  and the tracked `chroma.sqlite3` never gets dirtied (Chroma has no
  read-only open mode).
- SAE tables are never copied — the demo ships no SAE data, hence hiding the
  `/sae` page.
- FTS/BM25 indexes are not carried (ILIKE text search works regardless).

**Never commit `seed_demo/`** — its HNSW file (~176 MB) exceeds GitHub's
100 MB hard limit. `.gitignore` ignores it; it ships only in the Space repo,
where `huggingface_hub` uploads it via LFS automatically.

## Deploying

```bash
# one-time: hf auth login (or export HF_TOKEN)
uv run python deploy/hf-space/deploy.py --repo-id <user>/orrery-demo --create
```

`deploy.py` uploads the filtered working tree (`upload_folder` +
`ignore_patterns` keeping out the 23 GB live DuckDB, node_modules, docs, …),
then overwrites `README.md` with `deploy/hf-space/README_SPACE.md` (Space
frontmatter: `sdk: docker`, `app_port: 7860`) and `.dockerignore` with
`Dockerfile.dockerignore`.

**Ignore-file layout**: the root `.dockerignore` excludes `seed_demo/` so the
compose backend image never bakes the 313 MB seed. `Dockerfile.dockerignore`
is the seed-**including** variant — BuildKit prefers `<Dockerfile>.dockerignore`
when building the root `Dockerfile`, so local `docker build .` gets the seed;
`deploy.py` also uploads it as the Space repo's `.dockerignore` so the HF
builder gets the same context regardless. Keep the two files in sync (they
differ only in the seed_demo lines).

After the first deploy, set the Space **secret** `GEMINI_API_KEY` (semantic
search on the two Gemini collections; everything else works without it).

## Local verification

```bash
docker build -t orrery-hf .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=... orrery-hf

curl -s -X POST localhost:7860/graphql -H 'Content-Type: application/json' \
  -d '{"query":"{ collections { name count } }"}'          # 3 collections
curl -s -X POST localhost:7860/graphql -H 'Content-Type: application/json' \
  -d '{"query":"mutation { deleteCollection(collectionName: \"x\") }"}'  # read-only error
curl -s -o /dev/null -w '%{http_code}' -X POST localhost:7860/upload      # 404
curl -s -o /dev/null -w '%{http_code}' localhost:7860/collections         # 307 → /
```

Then check http://localhost:7860 in a browser: Explore-only nav, EMNLP topics
render, semantic search works on emotion (offline) and, with the key, on the
Gemini collections.

## Known exposures (accepted for the demo)

- **No rate limiting on `/graphql`**: `semanticSearch` is a query (allowed),
  and each text-query search on the two Gemini collections spends one Gemini
  API call from the Space secret. A scripted client could burn quota. If this
  becomes a problem, add an nginx `limit_req` on `location /graphql` — but
  note HF fronts the Space with a proxy, so `$binary_remote_addr` may be the
  proxy IP (one shared bucket for all visitors); key on
  `$http_x_forwarded_for` instead.
- **No process supervision in `start.sh`**: if uvicorn or node crashes, nginx
  (PID 1) keeps the container "up" serving 502s until HF restarts it. Fine
  for a demo; a `wait -n` wrapper that exits on any child death would let the
  platform restart the Space automatically.
