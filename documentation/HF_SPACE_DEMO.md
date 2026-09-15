# HuggingFace Space Demo (read-only)

A public, read-only demo of Orrery ships as a single-container HuggingFace
Docker Space. Demo = the Explore page plus the read-only `/sae` explorer, six
seeded collections: emotion, xkcd colors, the 13,980-abstract
`acl_abstracts_emnlp_findings` EMNLP collection (60 LLM-labeled topics), the
`Gemma_9_16k_embedded` SAE label map, the 212,478-sense `wordnet_senses_full`
galaxy, and `Glasgow_norm_all-gemini-2` (4,682 words, nine psycholinguistic
norms, pre-trained probes).

## What was built (implementation summary)

Six independent pieces, each usable on its own:

| Piece | Files | What it does |
|---|---|---|
| Social preview | `embedding_visualization/app/opengraph-image.jpg` (+ `.alt.txt`), `lib/utils/siteMetadata.ts`, `app/layout.tsx`, `README_SPACE.md` `thumbnail:`, `deploy.py` `ensure_site_url_variable` | Link cards when the demo is shared: the Space card uses the Hub-served image via the README `thumbnail`; the direct `*.hf.space` host emits absolute Open Graph/Twitter tags, with `metadataBase` baked from the Space Variable `NEXT_PUBLIC_SITE_URL`. See "Social preview" below. |
| Server-side read-only gate | `backend/API/read_only.py`, wired in `backend/API/__init__.py`; upload gating in `backend/main.py`; `generateStream` early-refusal in `backend/API/subscriptions.py` | With `ORRERY_READ_ONLY` truthy, every GraphQL mutation is rejected before execution (never touches resolvers/DB) and `/upload` isn't mounted. This is the actual security boundary — the GraphQL endpoint is public. Tests: `unit_tests/test_read_only.py`. |
| Frontend demo mode | `lib/utils/demoMode.ts` (`IS_DEMO`), gates in `PageNav`, `next.config.ts` redirects, `VisualizationControls`, `AnalyticsSidebar`, `DashboardPanel` | Cosmetic layer: Explore-only nav, `/collections` + `/sae` redirect to `/`, write-UI hidden. Build-time flag (`NEXT_PUBLIC_DEMO_MODE` Docker ARG). |
| Demo seed | `config/seed_snapshots/demo.json` + snapshot builder/publisher | The demo collections (plus SAE tables and Glasgow probes) are generated from a validated manifest, checksummed, and published to a private Dataset repository at an immutable revision. |
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
  duplicated Spaces with writes enabled). gzip is on (`gzip_proxied any` is
  load-bearing — HF's front proxy adds a `Via` header, and nginx skips
  compressing proxied requests without it): the full-collection GraphQL JSON
  compresses ~5–8×, which is what makes large collections shippable at all.
  Next's node server gzips its own pages/chunks; nginx passes those through.
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

`deploy.py` first pins the Space Variable `NEXT_PUBLIC_SITE_URL` to the
Space's `*.hf.space` origin (see "Social preview" below; skipped with a
warning if the token cannot write settings), then uploads
`deploy/hf-space/README_SPACE.md` as `README.md` (Space frontmatter:
`sdk: docker`, `app_port: 7860`, `thumbnail`; it goes first so a card the Hub
rejects fails the deploy before the tree is touched), then the filtered
working tree (`upload_folder` + `ignore_patterns` keeping out the 23 GB live
DuckDB, node_modules, docs, …), then the root `.dockerignore`. Any legacy
`seed_demo/` files are removed from the Space repo.

The single root `.dockerignore` excludes `seed_demo/` from every build context.
The demo Dockerfile downloads the locked private Dataset revision using the
Space's read-only `HF_SEED_TOKEN` BuildKit secret and verifies its manifest.

Before the first locked-seed deploy, set the Space **secret** `HF_SEED_TOKEN`
to a read-scoped token for the private Dataset. Set `GEMINI_API_KEY` separately
for semantic search on the two Gemini collections.

## Social preview (link cards)

Two different URLs get shared, and their link cards are built by two
different systems:

| Shared URL | Who builds the card | Where the image comes from |
|---|---|---|
| `https://huggingface.co/spaces/<id>` (the Space page) | the Hub, from the README frontmatter | `thumbnail:` in `README_SPACE.md` — the `{{SOCIAL_PREVIEW_URL}}` placeholder, resolved by `deploy.py` to `https://huggingface.co/spaces/<id>/resolve/main/embedding_visualization/app/opengraph-image.jpg`. Hub-served, so the card renders even while the Space sleeps. Without `thumbnail` the Hub generates a gradient card from `emoji` + `colorFrom`/`colorTo`. |
| `https://<sub>.hf.space` (the "Open fullscreen" direct host; also any self-hosted deployment) | the app's own `<meta>` tags | Next's file-convention image `app/opengraph-image.jpg` (+ `.alt.txt`), emitted by `app/layout.tsx` as `og:image` (`/opengraph-image.jpg?<content-hash>`, with type/width/height/alt) and inherited by the `twitter:*` tags (`summary_large_image`). |

The image is one file, `embedding_visualization/app/opengraph-image.jpg`:
1200×630 (the 1.91:1 Open Graph size), ~180 KB. It is a crop of the README
hero `gallery/dimensionality.png` — box `(0, 200, 2780, 1660)` (header row,
right-edge modebar and bottom-left logo excluded), Lanczos-resized, JPEG q88.
Regenerate from any gallery screenshot the same way, but keep the file name:
the README placeholder, the Next route and `deploy.SOCIAL_PREVIEW_PATH` all
point at it, and `unit_tests/test_hf_space_deploy.py` asserts that the file
exists and is not filtered out of the Space upload by `IGNORE_PATTERNS`.

`metadataBase` is the load-bearing part for the direct host. Next resolves
social images against it and, when it is unset, **production builds emit
`http://localhost:3000/opengraph-image.jpg…`** (Next's documented fallback;
dev mode always uses localhost regardless, so `next dev` cannot verify this).
The origin has to be known at *build* time (`NEXT_PUBLIC_*` values are baked
into the bundle). HF's built-in `SPACE_HOST` is runtime-only, but Space
**Variables** are passed to Docker builds as build-args — so
`deploy.py::ensure_site_url_variable` writes
`NEXT_PUBLIC_SITE_URL=https://<sub>.hf.space` as a Space Variable before every
deploy (idempotent: rewritten only when missing or stale; a token without
settings access degrades to a warning and the app still deploys), and the
root `Dockerfile` declares the matching `ARG`. `lib/utils/siteMetadata.ts`
turns the value into `metadataBase` (bare hosts get `https://`; anything that
is not http(s) is ignored). Caveats:

- A **duplicated Space** inherits the original's public Variables, so its
  direct-host cards point at the original's image until its own `deploy.py`
  run (or a manual Variable edit + rebuild) resets it.
- The compose stack takes the same build arg (`NEXT_PUBLIC_SITE_URL` in
  `docker-compose.yml`, empty by default — see `documentation/DOCKER.md`).

Verify after a deploy (crawlers cache cards aggressively — use the X card
validator, LinkedIn Post Inspector, or Facebook's Sharing Debugger to force a
refetch):

```bash
curl -sL https://huggingface.co/spaces/<id> | grep -oE '<meta[^>]*og:image[^>]*>'
curl -sL https://<sub>.hf.space/ | grep -oE '<meta[^>]*(og:image|twitter:image)"[^>]*>'
curl -sI "<the og:image URL printed above>" | head -3       # 200, image/jpeg
```

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
   per request — tens of seconds on 2 vCPU, per visitor. nginx gzip (~5–8× on
   JSON) fixes the wire size but not the serialization time. This is the real
   ceiling and it's architectural, not a platform limit.
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
6. **SAE inference** — the read-only `/sae` explorer is live in the demo (see
   "SAE feature exploration" below); what re-enabling would add is the
   inference layer. Chat + steering + prompt search need `loadModel`:
   Gemma-3-4b fits in 16 GB RAM but is impractically slow on 2 vCPUs — that
   feature realistically needs a GPU Space tier. Progress bars and chat
   streaming use WebSocket subscriptions, which the nginx config already
   proxies.

## SAE feature exploration (read-only, live in the demo)

The `/sae` Feature Explorer runs in the demo on seeded DuckDB/Chroma tables —
no model, no torch, no mutations. `demo.json` carries two `sae_data` entries:

- **`gemma-3-4b-it` / `9-gemmascope-2-res-16k`** — the paper's steering SAE
  and the explorable one: 16,384 features (~97% Neuronpedia-labeled),
  activation examples pruned to the top-10 per feature
  (`max_activation_examples_per_feature`), plus the MiniLM-embedded label
  collection `Gemma_9_16k_embedded` as `explanation_vector_collection` —
  semantic feature search embeds queries with the Space-local MiniLM model
  (no API quota). The label collection also appears in the Explore dropdown
  as a browsable map of the SAE's label space, and right-clicking its points
  cross-links into the explorer.
- **`gemma-3-1b` / `22-gemmascope-2-res-16k`** — features only (no activation
  examples), included purely because it owns the EMNLP document activations
  (`document_activations: ["acl_abstracts_emnlp_findings"]`, pruned to each
  document's top-256 features via `document_activation_top_k`) that power the
  Explore page's feature→document search. The `/sae` page hides zero-example
  pairs in demo builds, so this pair never appears in the explorer UI.
  **Upgrade path**: recompute document activations against the 4b SAE
  (`computeDocumentActivations`), move `document_activations` to the 4b
  entry, drop this one, reseed.

Demo gating on `/sae`: the Prompt search mode is hidden and the feature label
field is read-only. Everything else — browse by index, text + semantic
search, logits, densities, activation examples, quantiles, similar features —
is served read-only. A demo-only `?` button in the page header links back to
`/?intro=1` (the mission menu lives on the Explore page).

### Demo chat: pre-recorded steered sessions

The steered-chat sidebar opens in the demo, but plays back instead of
generating: `useChatSessions({ demo: true })` fetches
`public/demo/chat-sessions.json` (a committed static fixture) instead of
GraphQL, and every write is a no-op. The composer is disabled with an
explanatory placeholder, the model-status poll never mounts, edit/regenerate/
compare are hidden, History opens by default and lists the recorded sessions
read-only. No inference machinery activates — the generation WebSocket
subscription only ever opens on send.

**Recording a session** (`lib/utils/demoChatSessions.ts` documents the
format): run the full app locally with inference, steer, chat, then use the
chat header's **Download chat as JSON** button — each fixture entry is
exactly one such export, and the fixture file is a JSON array of them. A
missing or malformed fixture degrades to an empty History; a malformed entry
is skipped without blanking the rest.

## Onboarding: welcome dialog, presets, spotlight tour

The demo ships a three-layer onboarding system (implemented; module map in
the frontend `CLAUDE.md` under "Demo onboarding"):

1. **URL presets** (`?preset=<id>`, definitions in
   `lib/utils/tourPresets.ts`). A preset expands client-side into a full
   view: collection, colour scheme (fed through the same initial-refs path
   as explicit URL colour params — explicit URL params win), projection
   method/mode, and store flags (nebula, cluster labels). Shipped ids:
   `emnlp-topics`, `xkcd-manifold`, `emotion`, `sae-map`, `wordnet-pos`,
   `glasgow-norms`. The param persists while the user stays on the preset's
   collection and is dropped on switch. In demo builds the bare-URL default
   collection is `emotion` (small, and its MiniLM search model runs inside
   the Space — no Gemini quota). Presets are deep-link-only: the welcome
   dialog lists guided tours, not raw preset views (the former EMNLP/xkcd
   preset buttons were removed; their ids remain for links).
2. **First-visit welcome dialog** (`app/components/DemoIntro.tsx`). Auto-opens
   once per browser (`localStorage` key `orrery.demo-intro.v1`), demo builds
   only, never on top of a deep link (any `collection`/`colorBy`/`preset`/
   `tour` param suppresses it). Five entries, all guided: the 90-second tour,
   "Inspect SAE features" (gated on both the SAE pair and the label-map
   collection being present), "Decode human ratings" (the probing tour, gated
   on the Glasgow collection), "Tour the WordNet galaxy" (gated on the
   WordNet collection; its copy carries the heavy-load warning), or dismiss.
   Reopenable via `?intro=1` (any build) and the
   header `?` button. Opening it fires a one-shot warm-up query so the
   emotion search model cold-starts before the tour reaches the search step.
3. **Spotlight tour** (react-joyride v3, `?tour=1` in any build — the welcome
   dialog auto-offers it in demo builds). Seven steps defined as data in
   `lib/utils/tourSteps.ts`, rendered by `app/components/TourController.tsx`
   (dynamically imported, so normal visits don't load the library) with a
   custom frosted-glass tooltip reusing the plot hover tooltip's
   `.frosted-tooltip` surface. The tour runs on the **EMNLP collection**
   (`TOUR_PRESET_ID = 'emnlp-topics'`, applied when the tour starts) and
   follows map → search → feature-search → focus-topic → temporal →
   density → finale: steps prepare state programmatically and narrate the
   outcome — the user is never asked to operate controls. The **map** step carries the corpus
   narration (abstract count, 60 LLM-named topics, haze); the **search**
   step (second) owns the tour-collection wait, then fires the semantic
   query without awaiting it — the tooltip shows immediately, the search
   input's own spinner (inside the spotlight) shows progress, and the glow +
   camera dive land as a reveal; **feature-search** opens the Search panel
   and demos the SAE feature-name → document search
   (`TourRuntime.runFeatureSearch`, query "humor" → the model's "humor and
   jokes" feature → abstracts ranked by stored activation, spotlight on the
   `feature-search` anchor) — pure DuckDB over the seeded
   `sae_document_activations`, zero API cost, works read-only;
   **focus-topic** isolates one topic cluster
   programmatically (polling until the topics query has landed; the muting
   auto-refit reframes the camera on it); **temporal** opens the Analytics
   panel and brushes the earliest third of the detected temporal field
   (`TourRuntime.applyTemporalWindow`; its card sits on the right-edge
   `plot-side` anchor like the other map-narrating steps — the prepare waits
   on the `temporal-chart` element so the timeline is on-screen first);
   **density** flips to the 2D projection with density contours
   (`TourRuntime.setDensityView` = mode `2d` + `densityMode`), deliberately
   leaving the Analytics panel open; the finale switches collections to
   **emotion** as the "collections are spaces" payoff — the demo default,
   whose search model runs inside the Space so visitors keep querying
   without Gemini quota (its preset pins `densityMode` off, restoring 3D).
   All tour-applied state — search glow, camera framing
   (`TourRuntime.resetCamera` → `cameraResetSignal` prop into
   `ScatterPlot3D`), topic isolation, temporal window, density view — is
   also cleared on any tour end, so a mid-tour skip can't strand it. The
   search step queries ONLY the tour collection — a deliberate cost of one
   Gemini embed call per tour run, hard-guarded in `tourSteps.ts` so no
   other collection is auto-queried. Targets are `data-tour` attributes in
   `AppHeader`/`DashboardPanel`/`AnalyticsSidebar`. The Explore page's topic
   list (which the focus-topic step and topic search depend on) comes from
   the `collectionTopics` GraphQL query (`lib/hooks/useCollectionTopics.ts`)
   — the manifest's legacy `topic_summary` blob is never written by
   post-DuckDB-migration stores. Steps whose waits have native feedback
   suppress joyride's built-in waiting spinner (`suppressWaitLoader` in
   `tourSteps.ts`) — the page's own full-screen loader covers the
   collection-switching waits, and both at once read as a bug.
   Completion/dismissal is recorded under `orrery.demo-tour.v1`; on
   viewports below 768 px the tour downgrades to the dialog. Demo builds
   also default to **dark mode** (`providers.tsx`) and label search results
   out of the box (`showLabels` store default). Demo builds use a
   **two-row header**: nav pills, the `?` button, and the theme toggle sit
   in a second right-aligned row beside the Plotly modebar, clear of the
   HuggingFace pill that `header: mini` floats over the iframe's top-right
   (row 1 keeps a clearance spacer under it when embedded).
4. **"Inspect SAE" tour** (`?tour=sae`, steps in `lib/utils/saeTourSteps.ts`,
   storage key `orrery.demo-sae-tour.v1`). Two segments chained across a page
   navigation, because joyride cannot survive one. **Segment 1** runs on the
   Explore page over the SAE label map: applies the `sae-map` preset
   (`Gemma_9_16k_embedded`), semantic-searches "poetry" on the Space-local
   MiniLM model, and teaches the right-click → "View Feature" gesture; its
   Done handler navigates to `/sae` with the top match deep-linked
   (`saeInspectPath` → `modelId`/`saeId`/`featureIndex` + `tour=sae`), using
   the `TourController.onDone(outcome)` argument to hand off only on a real
   finish, never on skip. **Segment 2** runs on `/sae`: feature anatomy
   (label, density, logits) → activation examples → the pre-recorded steered
   chat (opens the sidebar and replays the first fixture session) → the link
   back to the map. A direct `/sae?tour=sae` entry still works — the anatomy
   step falls back to running the semantic search itself.
5. **WordNet galaxy tour** (`?tour=wordnet`, steps in
   `lib/utils/wordnetTourSteps.ts`, storage key
   `orrery.demo-wordnet-tour.v1`). Single-page, on `wordnet_senses_full`
   (212,478 senses — the paper's Figure-1 dataset, the demo's heaviest map;
   load waits get 90 s ceilings). Narrative: the map opens already wearing
   POS colours (the `wordnet-pos` preset, which matches the collection's
   saved default — deliberately NO recolor on arrival, an earlier
   uncoloured-reveal design flickered), step 2 narrates the POS continents
   (its prepare only restores POS/no-nebula for Back-nav), step 3 is the
   Figure-1 view: recolor by `topic_label` (thousands of LLM-named topics —
   the heaviest recolor the demo does, 45 s prepare ceiling) + nebula haze +
   cluster labels (`TourRuntime.setNebulaMode`/`setShowClusterLabels`),
   step 4 fire-and-forget searches "geometry" (the collection is
   MiniLM-embedded — the exact model the image bakes, so search is free and
   offline; hard collection guard as always), step 5 resets the camera for
   the parting nebula shot. Completion leaves the nebula view; a mid-tour
   skip re-applies the preset (guarded on the collection having actually
   loaded, so bailing during the long load never re-triggers it).
6. **Probing tour** (`?tour=probe`, steps in `lib/utils/probeTourSteps.ts`,
   storage key `orrery.demo-probe-tour.v1`). Single-page, on
   `Glasgow_norm_all-gemini-2` (4,682 words, nine Glasgow norms as numeric
   metadata) — the paper's probing experiment as a guided walk. Steps: actual
   concreteness ratings (the `glasgow-norms` preset pins the managua
   diverging scale — the collection's curated look; without an explicit
   scale the recommended-scale path lands on the rainbow-like sinebow) → the
   ridge probe's predicted scores in the SAME managua palette so "the
   picture barely changes" (`probe_concreteness_ridge_score`, R² = 0.80
   held-out) → zero-centered diverging residuals → valence
   (`probe_valence_ridge_score`, R² = 0.76, managua) → the Direction Probes
   bench in the Analytics panel (`probe-section` anchor; demo builds render
   the REAL `ProbeSection` read-only — shipped probes stay listed and
   colourable, only fitting/deleting disappear). Entirely free and
   read-only: probes + per-word scores are trained offline and shipped in
   the seed (`demo.json` sets `include.probes` for the collection), and the
   steps only recolor by fields `useProbes` merges client-side — recoloring
   steps wait on `TourRuntime.hasColorField` since probe fields arrive via
   their own queries, and skip rather than paint a missing field. There is
   deliberately NO search step: the collection is Gemini-embedded, and
   auto-searching it would spend metered quota. (The initial score-fetch
   fan-out bug this tour exposed — a single `useLazyQuery` executor
   cancelling all but the last of N parallel calls — is fixed in
   `useProbes` with independent `client.query` calls.)

- **Phone-sized viewports** get a fourth surface: a one-time "Best viewed on
  desktop" card (`app/components/MobileNotice.tsx`), mounted on both `/` and
  `/sae`. Gating is the shared `shouldShowMobileNotice` predicate in
  `demoOnboarding.ts` — demo builds only, `viewportWidth < TOUR_MIN_VIEWPORT`
  (768), and not yet dismissed (`orrery.demo-mobile-notice.v1`, one key across
  both pages). It is **dismissible, not blocking**: the app is usable on a
  phone, just cramped. `getOnboardingAction` checks it **first**, so on a
  phone it pre-empts `?tour=1` / `?intro=1` / `?tour=sae` / `?tour=wordnet` /
  `?tour=probe` — the tours are
  already downgraded below 768 px anyway, and a `?tour=sae` deep link stays on
  Explore rather than forwarding until the card has been dismissed once. On
  the next visit the ordinary gating resumes (so a phone visitor sees the
  welcome dialog then, not a second copy of the card).
  Mobile support is otherwise deliberately partial — see "Known exposures".
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
- **Mobile is functional, not designed.** Below 768 px the header wraps, the
  three panels open as near-full-width overlays, and the desktop-notice card
  sets expectations — but the plot chrome is still laid out for a desktop:
  the legend and Plotly modebar keep their fixed `right-4` / `top: 80px`
  offsets and overlap an open panel, the `useZoomLimit` clamp only listens for
  `wheel` so pinch-zoom is unbounded, the SAE context menu needs a right-click,
  and the 3D GL scene + Three.js haze overlay are untested for memory on
  mobile Safari. Fixing those (plus defaulting to 2D on small screens) is the
  scoped follow-up; nothing here is load-bearing for the demo's desktop path.
