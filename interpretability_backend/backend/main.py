"""FastAPI backend with GraphQL endpoint for embedding visualization.

Supports:
- GraphQL queries and mutations over HTTP
- GraphQL subscriptions over WebSocket for real-time progress updates
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from .API import schema
from .API.upload import router as upload_router
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

# Create GraphQL router with WebSocket subscription support
graphql_app = GraphQLRouter(
    schema,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ],
)

# Mount upload router
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
