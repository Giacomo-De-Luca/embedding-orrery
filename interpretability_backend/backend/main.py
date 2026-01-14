"""FastAPI backend with GraphQL endpoint for embedding visualization."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from .API import schema
from .API.upload import router as upload_router


# Create FastAPI app
app = FastAPI(
    title="Embedding Visualization API",
    description="GraphQL API for exploring word embeddings with ChromaDB",
    version="1.0.0"
)

# Configure CORS
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

# Create GraphQL router
graphql_app = GraphQLRouter(schema)

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
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "interpretability_backend.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
