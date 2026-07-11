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
# Build (Dockerfile.dockerignore takes precedence over .dockerignore for this
# file, letting the ~313 MB demo seed into the context; build it first at
# interpretability_backend/resources/seed_demo — see HF_SPACE_DEMO.md):
#   docker build -t orrery-hf .
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
# Same-origin GraphQL through the nginx proxy; relative URL resolves against
# the page origin in the browser. The WS URL keeps its localhost default —
# demo mode hides every subscription-using component, so no socket is opened.
ARG NEXT_PUBLIC_GRAPHQL_URL=/graphql
ARG NEXT_PUBLIC_GRAPHQL_WS_URL=ws://localhost:8000/graphql
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
