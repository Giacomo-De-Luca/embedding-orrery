# HuggingFace Space Demo (read-only)

A public, read-only demo of Orrery ships as a single-container HuggingFace
Docker Space. Demo = Explore page only, three seeded collections (emotion,
xkcd colors, and the 13,980-abstract `acl_abstracts_emnlp_findings` EMNLP
collection with 60 LLM-labeled topics).

## What was built (implementation summary)

Five independent pieces, each usable on its own:

| Piece | Files | What it does |
|---|---|---|
| Server-side read-only gate | `backend/API/read_only.py`, wired in `backend/API/__init__.py`; upload gating in `backend/main.py`; `generateStream` early-refusal in `backend/API/subscriptions.py` | With `ORRERY_READ_ONLY` truthy, every GraphQL mutation is rejected before execution (never touches resolvers/DB) and `/upload` isn't mounted. This is the actual security boundary — the GraphQL endpoint is public. Tests: `unit_tests/test_read_only.py`. |
| Frontend demo mode | `lib/utils/demoMode.ts` (`IS_DEMO`), gates in `PageNav`, `next.config.ts` redirects, `VisualizationControls`, `AnalyticsSidebar`, `DashboardPanel` | Cosmetic layer: Explore-only nav, `/collections` + `/sae` redirect to `/`, write-UI hidden. Build-time flag (`NEXT_PUBLIC_DEMO_MODE` Docker ARG). |
| Demo seed | `config/seed_snapshots/demo.json` + snapshot builder/publisher | The three demo collections are generated from a validated manifest, checksummed, and published to a private Dataset repository at an immutable revision. |
| Single image | root `Dockerfile`, `deploy/hf-space/nginx.conf`, `deploy/hf-space/start.sh` | nginx :7860 fronts uvicorn :8000 + Next standalone :3000; the build downloads and verifies the locked seed with a secret-mounted token. |
| Deploy tooling | `deploy/hf-space/deploy.py`, `.github/workflows/containers.yml` | Uploads filtered GitHub source after tests; the Space rebuilds automatically once `demo.lock.json` is committed. |

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
through nginx). When no explicit WebSocket URL is supplied, the browser derives
`ws://` or `wss://` from the page origin.

## Demo seed (`resources/seed_demo/`, ~313 MB)

Built from `config/seed_snapshots/demo.json` — run **with the backend stopped**:

```bash
uv run python -m interpretability_backend.scripts.build_seed_snapshot \
    --config interpretability_backend/config/seed_snapshots/demo.json
```

No flags loads `default.json` and rebuilds the committed seed. Script notes:

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
- SAE tables are controlled by the optional `sae_data` list. The current demo
  manifest leaves it empty, hence hiding the `/sae` page.
- FTS/BM25 indexes are not carried (ILIKE text search works regardless).
- A failed build never destroys the prior snapshot: export, checksum, and
  verification finish in a staging directory before atomic replacement.

**Never commit `seed_demo/`** — its HNSW file (~176 MB) exceeds GitHub's
100 MB hard limit. Publish it to a private Dataset repository and commit only
the small generated `demo.lock.json`.

## Deploying

```bash
# On the machine with the live stores:
export ORRERY_SEED_REPO_ID=<user-or-org>/orrery-demo-seed
export HF_TOKEN=hf_...
uv run python -m interpretability_backend.scripts.publish_seed_snapshot \
  --config interpretability_backend/config/seed_snapshots/demo.json

# Commit demo.lock.json, then create/deploy the Space:
uv run python deploy/hf-space/deploy.py --repo-id <user>/orrery-demo --create
```

`deploy.py` uploads the filtered working tree (`upload_folder` +
`ignore_patterns` keeping out the 23 GB live DuckDB, node_modules, docs, …),
then overwrites `README.md` with `deploy/hf-space/README_SPACE.md` (Space
frontmatter: `sdk: docker`, `app_port: 7860`) and uploads the root
`.dockerignore`. Any legacy `seed_demo/` files are removed from the Space repo.

The single root `.dockerignore` excludes `seed_demo/` from every build context.
The demo Dockerfile downloads the locked private Dataset revision using the
Space's read-only `HF_SEED_TOKEN` BuildKit secret and verifies its manifest.

Before the first locked-seed deploy, set the Space **secret** `HF_SEED_TOKEN`
to a read-scoped token for the private Dataset. Set `GEMINI_API_KEY` separately
for semantic search on the two Gemini collections.

## Local verification

```bash
docker build --secret id=HF_SEED_TOKEN,env=HF_SEED_TOKEN -t orrery-hf .
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

## Capacity: how large a collection fits

The free CPU Space is 2 vCPU / 16 GB RAM / ephemeral disk. The binding
constraints, in the order they actually bite:

1. **The full-collection load, not HF.** The Explore page fetches the entire
   collection in one GraphQL response (documents + metadata + projections; no
   pagination). At ~1.5–3 KB/doc of JSON, 100k docs ≈ 200–300 MB serialized
   per request — tens of seconds on 2 vCPU, per visitor, before gzip. This is
   the real ceiling and it's architectural, not a platform limit.
2. **Seed / image size.** At 3072-d (Gemini) a document costs ~20 KB in the
   seed (12.3 KB vectors + HNSW + DuckDB rows + Chroma's sqlite copy) —
   measured: EMNLP's 13,980 docs added ~290 MB. The seed bakes into the image
   ~1:1 and gets copied to `/home/user/data` on every cold start. At 384-d
   (MiniLM-class) it's ~8× cheaper.
3. **RAM.** Chroma holds queried collections' vectors in memory:
   `N × dims × 4 B` (100k × 3072-d ≈ 1.2 GB). Comfortable within 16 GB until
   well past the point where (1) already hurts.
4. **Hard platform limits** (none binding in practice): LFS caps a single
   file at 50 GB (the HNSW `data_level0.bin` grows ~12.3 KB/doc at 3072-d);
   no published image-size limit, but build + cold-start time scale with it.

Rules of thumb at 3072-d: **≤50k docs is comfortable** (seed ~1 GB, loads in
seconds), **100–200k is workable** (the repo's own fps benchmarks validated
Plotly WebGL up to a real 212k-point collection and 1M synthetic, but the
first load will take tens of seconds on the free tier and concurrent visitors
compound it), beyond that reduce dimensions (a 384-d collection of 500k docs
costs about what a 3072-d 60k one does) or accept that a proper fix means
paginated/streamed point loading, which is an app change, not a deploy change.

## Updating the Space

After backend/frontend tests and Docker Hub image publication succeed on
`main`, `.github/workflows/containers.yml` runs `deploy.py` with the GitHub
`HF_TOKEN` secret and `HF_SPACE_REPO_ID` variable. The job skips safely until
`demo.lock.json` exists. Each Space commit rebuilds the image; the build reads
the lock and downloads the exact private Dataset revision.

Application-code updates are therefore automatic. Snapshot data updates remain
deliberate because hosted CI cannot access the live DuckDB/Chroma stores:
rebuild and publish from the data-holding machine, commit the changed lock, and
push it to `main`.

## Re-enabling the Collections and SAE pages

The demo hides them via two independent layers; both must be flipped, and a
few practical gaps need filling. Do this on a **private or duplicated** Space
— re-enabling writes on the public demo hands `deleteCollection`, embedding
jobs, and your API keys' quota to every visitor.

1. **Backend writes** — set Space **Variable** `ORRERY_READ_ONLY=0`
   (runtime-read, overrides the image ENV; no rebuild needed). Mutations and
   the `/upload` router come back.
2. **Frontend UI** — `NEXT_PUBLIC_DEMO_MODE` is baked into the JS bundle at
   build time. Set it as a Space **Variable** `NEXT_PUBLIC_DEMO_MODE=0` (HF
   passes variables as build args where the Dockerfile declares a matching
   `ARG`, which ours does) and trigger a **Factory rebuild** — or edit the
   `ARG NEXT_PUBLIC_DEMO_MODE=1` default in the Space repo's `Dockerfile`.
   Nav tabs, routes, probes, and save-default all return with the one flag.
3. **nginx** — `/upload` is deliberately unrouted; add to
   `deploy/hf-space/nginx.conf`:
   `location /upload { proxy_pass http://127.0.0.1:8000; client_max_body_size 300m; }`
   (the default 1 MB body cap would reject any real dataset upload).
4. **Persistence** — without the persistent-storage add-on, everything users
   embed vanishes on restart/sleep (the ephemeral reseed that's a feature for
   the demo is a bug for a workspace). With the add-on, point
   `ORRERY_RESOURCE_DIR` at the mounted `/data`.
5. **Provider keys** — add Space secrets for whichever embedding providers
   should work (`GEMINI_API_KEY`, `CHROMA_OPENAI_API_KEY`, …).
   SentenceTransformers models run locally with no key (CPU: fine for
   MiniLM-class models, slow for large ones).
6. **SAE page reality check** — the demo seed ships **no SAE tables**, so an
   unhidden `/sae` page starts empty. Options: add a `sae_data` entry to
   `demo.json` selecting `features` and, if wanted, `activation_examples`
   (the token-window examples are the bulky part), rebuild/publish the locked
   snapshot, or use the live `prepareSaeData` flow (downloads GBs from
   Neuronpedia S3 — wants persistent storage so it survives restarts). Chat +
   steering additionally need `loadModel`: Gemma-3-4b fits in 16 GB RAM but
   is impractically slow on 2 vCPUs — that feature realistically needs a GPU
   Space tier. Progress bars and chat streaming use WebSocket subscriptions,
   which the nginx config already proxies.

## Onboarding: welcome dialog, presets, spotlight tour

The demo ships a three-layer onboarding system (implemented; module map in
the frontend `CLAUDE.md` under "Demo onboarding"):

1. **URL presets** (`?preset=<id>`, definitions in
   `lib/utils/tourPresets.ts`). A preset expands client-side into a full
   view: collection, colour scheme (fed through the same initial-refs path
   as explicit URL colour params — explicit URL params win), projection
   method/mode, and store flags (nebula, cluster labels). Shipped ids:
   `emnlp-topics`, `xkcd-manifold`, `emotion`. The param persists while the
   user stays on the preset's collection and is dropped on switch. In demo
   builds the bare-URL default collection is `emotion` (small, and its
   MiniLM search model runs inside the Space — no Gemini quota).
2. **First-visit welcome dialog** (`app/components/DemoIntro.tsx`). Auto-opens
   once per browser (`localStorage` key `orrery.demo-intro.v1`), demo builds
   only, never on top of a deep link (any `collection`/`colorBy`/`preset`/
   `tour` param suppresses it). Four entries: start the tour, the two preset
   missions, or dismiss. Reopenable via `?intro=1` (any build) and the
   header `?` button. Opening it fires a one-shot warm-up query so the
   emotion search model cold-starts before the tour reaches the search step.
3. **Spotlight tour** (react-joyride v3, `?tour=1` in any build — the welcome
   dialog auto-offers it in demo builds). Five steps defined as data in
   `lib/utils/tourSteps.ts`, rendered by `app/components/TourController.tsx`
   (dynamically imported, so normal visits don't load the library). Steps
   prepare state programmatically (apply the emotion preset, run a semantic
   search, open the Analytics panel) and narrate the outcome — the user is
   never asked to operate controls. Targets are `data-tour` attributes in
   `AppHeader`/`DashboardPanel`. The search step only ever queries the
   emotion collection. Completion/dismissal is recorded under
   `orrery.demo-tour.v1`; on viewports below 768 px the tour downgrades to
   the dialog.

Two supporting mechanisms:

- **Unknown URL params survive**: the Explore page's URL sync merges its
  owned params into the existing query string (`lib/utils/urlViewParams.ts`)
  instead of rebuilding it, and strips the one-shot `tour`/`intro` params.
- **HF parent URL sync**: inside the Space iframe the app posts its query
  string to `https://huggingface.co` via `postMessage` on every URL change
  (`lib/utils/hfSpaceUrlSync.ts` + `useHfSpaceUrlSync`), so the address bar
  on the Space page is shareable. The README's "Start here" links use
  `{{SPACE_URL}}` / `{{SPACE_DIRECT_URL}}` placeholders that
  `deploy/hf-space/deploy.py` resolves at upload time (the repo id only
  exists in CI variables).

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
