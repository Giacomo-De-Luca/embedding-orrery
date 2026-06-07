#!/bin/bash
# Start the GraphQL backend server

echo "Starting Embedding Visualization GraphQL Backend..."
echo "Server will be available at: http://localhost:8000"
echo "GraphQL Playground: http://localhost:8000/graphql"
echo ""

uv run uvicorn interpretability_backend.backend.main:app --host 0.0.0.0 --port 8000 --reload \
  --reload-include '*.py'
