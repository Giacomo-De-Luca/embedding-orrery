# Production Docker

Orrery publishes separate backend and frontend images. Docker Compose runs
those two application containers with a stock nginx gateway, named volumes for
mutable backend data/model caches, and an optional SAE cache warm-up job.

## Local production build

```bash
docker compose up --build
```

- Application and same-origin API gateway: http://localhost:3000
- Direct backend health/API access: http://localhost:8000/health and `/graphql`

The frontend is built with `/graphql` and `/upload` same-origin routes. The
browser derives `ws://` or `wss://` from its current origin, while nginx routes
HTTP, uploads, and WebSocket upgrades to the backend. This makes the same
frontend image portable across localhost, HTTPS domains, and reverse proxies.

For a source checkout started with `npm run dev`, public endpoint variables are
normally absent, so `endpointUrls.ts` targets the separately running backend at
`http://localhost:8000` (GraphQL HTTP/WebSocket and uploads). Docker builds pass
`/graphql` explicitly and pass empty WebSocket/API-base values explicitly; those
values select the same-origin gateway behavior above. The distinction between
an absent value and an explicitly empty value is intentional.

## Published Docker Hub images

Set the Docker Hub user or organization that owns `orrery-backend` and
`orrery-frontend`, then combine the normal stack with its image-only override:

```bash
export ORRERY_IMAGE_NAMESPACE=<dockerhub-user-or-org>
export ORRERY_IMAGE_TAG=latest
docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d
```

The override resets local `build` definitions, pulls both published images, and
leaves the gateway/volumes/environment in the shared base file. Upgrade an
installation explicitly:

```bash
docker compose -f docker-compose.yml -f docker-compose.hub.yml pull
docker compose -f docker-compose.yml -f docker-compose.hub.yml up -d
```

`main` publishes `edge` plus immutable full-commit `sha-*` tags. Publishing a
GitHub Release whose tag starts with `v` publishes the semantic version, its
major/minor alias, and `latest`. Images target `linux/amd64` and `linux/arm64`.

## Reset behavior

```bash
# Stop containers but retain created collections and model caches.
docker compose down

# Remove data/model-cache volumes; the next startup restores the committed seed.
docker compose down -v
docker compose up --build
```

The backend copies `interpretability_backend/resources/seed/` into the data
volume only when `/data/main.duckdb` is absent. Existing data is never replaced.

## Optional SAE cache profile

```bash
HF_TOKEN=... docker compose --profile sae up --build
```

`sae-warmup` reuses the backend image and shared volumes. It prefetches
`google/gemma-3-4b-it` and prepares the configured layer 9 residual 16k SAE,
then exits without loading the model into memory. `HUGGINGFACE_HUB_TOKEN` and
`HUGGINGFACE_API_KEY` remain accepted aliases.

| Volume | Mounted at | Contents |
|---|---|---|
| `orrery_backend_data` | `/data` | DuckDB, ChromaDB, uploads, jobs, SAE labels/vectors |
| `orrery_hf_cache` | `/models/huggingface` | Hugging Face model and SAE cache |

## Runtime paths and provider keys

| Variable | Docker value | Purpose |
|---|---|---|
| `ORRERY_RESOURCE_DIR` | `/data` | Mutable backend resource root |
| `ORRERY_SEED_DIR` | `/app/interpretability_backend/resources/seed` | Read-only initial snapshot |
| `ORRERY_DIRECTIONS_DIR` | `/app/interpretability_backend/resources/directions` | Steering presets |
| `HF_HOME` | `/models/huggingface` | Model cache volume |

Provider keys (`GEMINI_API_KEY`, `CHROMA_OPENAI_API_KEY`,
`CHROMA_COHERE_API_KEY`, and `CHROMA_HUGGINGFACE_API_KEY`) pass only to the
backend at runtime; they are not frontend build arguments.

## Publishing automation

`.github/workflows/containers.yml` gates publication on backend/frontend tests.
Configure these repository settings before enabling releases:

- Secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HF_TOKEN`.
- Variables: `ORRERY_IMAGE_NAMESPACE`, `HF_SPACE_REPO_ID`.

The Space deployment job skips safely until the immutable demo seed lock has
been published and committed. The Space itself also needs a one-time,
read-scoped `HF_SEED_TOKEN` secret for its private Dataset repository.

The Docker frontend still uses `ORRERY_DOCKER_BUILD=1` to bypass the existing
Next.js type/lint build backlog. Tests remain a separate CI gate; a successful
container build is not evidence that the entire frontend is type-clean.
