# syntax=docker/dockerfile:1.7
#
# Single-image build for the read-only HuggingFace Space demo.
# One container serves everything through nginx on port 7860:
#   nginx :7860  →  /graphql, /health  →  uvicorn :8000 (FastAPI + GraphQL)
#                →  everything else    →  Next.js standalone :3000
#
# The local development / full-application setup is docker-compose.yml
# (Dockerfile.backend + embedding_visualization/Dockerfile) — this file is
# deliberately separate and changes nothing about the compose stack.
#
# The demo seed is downloaded from its immutable private Hugging Face Dataset
# revision. Until the first demo.lock.json is published, builds use the small
# committed seed as a migration fallback:
#   docker build --secret id=HF_SEED_TOKEN,env=HF_SEED_TOKEN -t orrery-hf .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... orrery-hf
#
# See documentation/HF_SPACE_DEMO.md.

########## Frontend deps (mirrors embedding_visualization/Dockerfile) ##########
FROM node:22-slim AS frontend-deps

WORKDIR /fe

COPY embedding_visualization/package.json embedding_visualization/package-lock.json ./
COPY embedding_visualization/forked ./forked
RUN npm ci

########## Frontend build ##########
FROM node:22-slim AS frontend-builder

WORKDIR /fe

ENV NEXT_TELEMETRY_DISABLED=1
ENV ORRERY_DOCKER_BUILD=1
# Same-origin API access through the nginx proxy. The browser derives ws/wss
# from its own origin when NEXT_PUBLIC_GRAPHQL_WS_URL is empty.
ARG NEXT_PUBLIC_GRAPHQL_URL=/graphql
ARG NEXT_PUBLIC_GRAPHQL_WS_URL=
ARG NEXT_PUBLIC_API_BASE_URL=
ARG NEXT_PUBLIC_DEMO_MODE=1
ENV NEXT_PUBLIC_GRAPHQL_URL=$NEXT_PUBLIC_GRAPHQL_URL
ENV NEXT_PUBLIC_GRAPHQL_WS_URL=$NEXT_PUBLIC_GRAPHQL_WS_URL
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_DEMO_MODE=$NEXT_PUBLIC_DEMO_MODE

COPY --from=frontend-deps /fe/node_modules ./node_modules
COPY embedding_visualization/ .
RUN npm run build:docker

########## Backend deps ##########
FROM python:3.12-slim AS backend-deps

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml uv.lock ./

# Linux resolves the CPU-only torch wheels (see [tool.uv.sources] in
# pyproject.toml) — no CUDA payload in this image.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-install-project

########## Runner ##########
FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Node runtime for the Next standalone server (both images are Debian
# bookworm, so the binary's shared-library deps are satisfied).
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# HF Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/data /home/user/hf /tmp/nginx \
    && chown -R user:user /home/user /tmp/nginx

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
# Writable paths for the non-root user; /home/user/data is wiped on Space
# restart, which re-seeds a pristine demo (intended).
ENV ORRERY_RESOURCE_DIR=/home/user/data
ENV ORRERY_SEED_DIR=/app/interpretability_backend/resources/seed_demo
ENV ORRERY_DIRECTIONS_DIR=/app/interpretability_backend/resources/directions
ENV HF_HOME=/home/user/hf
# Server-side read-only gate (blocks all GraphQL mutations + /upload).
# Duplicated Spaces can override this with a Space Variable ORRERY_READ_ONLY=0.
ENV ORRERY_READ_ONLY=1

COPY --from=backend-deps --chown=user:user /app/.venv /app/.venv

# Pre-bake the emotion collection's embedding model so semantic search on it
# works offline and the first query is fast. Baked into an image layer, so it
# survives Space restarts (unlike /home/user/data).
USER user
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
USER root

COPY --chown=user:user pyproject.toml uv.lock README.md ./
COPY --chown=user:user interpretability_backend ./interpretability_backend

# A committed demo.lock.json makes the token mandatory inside the downloader;
# before that first publication, the command copies resources/seed as a small
# buildable fallback. BuildKit exposes Space secrets at /run/secrets without
# persisting them in this layer.
USER user
RUN --mount=type=secret,id=HF_SEED_TOKEN,mode=0444,required=false \
    --mount=type=cache,target=/home/user/hf,uid=1000,gid=1000 \
    uv run --frozen python -m interpretability_backend.scripts.download_seed_snapshot \
      --config interpretability_backend/config/seed_snapshots/demo.json \
      --token-file /run/secrets/HF_SEED_TOKEN \
      --fallback interpretability_backend/resources/seed
USER root

COPY --from=frontend-builder --chown=user:user /fe/public ./frontend/public
COPY --from=frontend-builder --chown=user:user /fe/.next/standalone ./frontend/
COPY --from=frontend-builder --chown=user:user /fe/.next/static ./frontend/.next/static

COPY deploy/hf-space/nginx.conf /etc/nginx/nginx.conf
COPY --chown=user:user deploy/hf-space/start.sh /app/start.sh
RUN chmod +x /app/start.sh

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["/app/start.sh"]
