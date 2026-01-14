# Embedding Platform Frontend

Interactive web application for exploring embedding collections in 2D/3D with clustering, semantic search, and real-time filtering.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

## Features

- **High-Performance Visualization**: Render 100k+ points using WebGL (Plotly.js).
- **Interactive Projections**: Switch between 2D and 3D views, PCA and UMAP.
- **Semantic Clustering**: Density-based clustering with automatic topic labeling (WASM).
- **Real-Time Search**: 
    - **Keyword Search**: Filter by text content.
    - **Semantic Search**: Find similar items using vector similarity (via backend).
- **Dataset Management**:
    - Embed HuggingFace datasets directly from the UI.
    - Upload local files (CSV, JSON, Parquet).

## Configuration

The frontend connects to the GraphQL backend at `http://localhost:8000/graphql`.
Ensure the backend is running (`./start_backend.sh`) for full functionality (Semantic Search, Embedding).

### Color Configuration
Edit `lib/categoryColors.ts` to customize color palettes for different metadata categories.

## Troubleshooting

- **No Data Shown**: Ensure the backend is running and you have embedded at least one dataset.
- **"Network Error"**: Check that `http://localhost:8000/graphql` is accessible.
- **Performance**: For very large datasets (>100k points), switch to 2D mode for better responsiveness.
