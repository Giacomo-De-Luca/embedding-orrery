"""FastAPI backend with GraphQL endpoint for embedding visualization.

Supports:
- GraphQL queries and mutations over HTTP
- GraphQL subscriptions over WebSocket for real-time progress updates
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from .API import schema
from .API.graphql_router import OrreryGraphQLRouter
from .API.read_only import is_read_only
from .API.response_cache_extension import response_cache_active
from .API.upload import router as upload_router
from .utils.http_compression import CompressionSettings, JsonGzipMiddleware
from .utils.resource_paths import PACKAGE_ROOT
from .utils.response_cache import GraphQLResponseCacheMiddleware, load_response_cache_settings
from .utils.seed_bootstrap import ensure_seed_loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Populate the data stores from the shipped seed snapshot on first run."""
    ensure_seed_loaded()
    yield


# Create FastAPI app
app = FastAPI(
    title="Embedding Visualization API",
    description="GraphQL API for exploring word embeddings with ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# Replay cached GraphQL responses on read-only deployments, where the data is
# immutable. Added first, so it runs innermost: CORS still decorates hits and
# the gzip middleware passes the stored gzip bodies through.
response_cache_settings = load_response_cache_settings()
app.add_middleware(
    GraphQLResponseCacheMiddleware,
    settings=response_cache_settings,
    is_active=lambda: response_cache_active(response_cache_settings),
)

# Configure CORS (including WebSocket origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip JSON responses (local dev talks to uvicorn directly, without nginx)
app.add_middleware(
    JsonGzipMiddleware,
    settings=CompressionSettings.from_file(PACKAGE_ROOT / "config" / "http_compression.json"),
)

# Create GraphQL router with WebSocket subscription support (orjson responses)
graphql_app = OrreryGraphQLRouter(
    schema,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
)

# Mount upload router (disabled in read-only demo deployments)
if not is_read_only():
    app.include_router(upload_router)

# Mount GraphQL endpoint
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Embedding Visualization GraphQL API",
        "graphql_endpoint": "/graphql",
        "graphql_playground": "/graphql (visit in browser)",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("interpretability_backend.backend.main:app", host="0.0.0.0", port=8000, reload=True)
