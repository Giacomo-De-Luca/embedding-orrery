# Embedding Platform Frontend

Interactive Next.js web application for exploring embedding collections in 2D/3D with clustering, semantic search, and real-time filtering. Built with modern React patterns, TypeScript, and a modular architecture.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server (requires backend at http://localhost:8000)
npm run dev

# Open http://localhost:3000

# Production build
npm run build
npm start
```

**First Launch**: The first time you open the app, there are no collections to display. Click the "Embed" button in the top-right header to navigate to `/test-embed` and create your first collection from HuggingFace datasets or local files.

## Features

### 🎨 High-Performance Visualization
- **WebGL Rendering**: Render 150k+ points using Plotly.js with logarithmic marker sizing
- **2D/3D Modes**: Toggle between 2D (aspect ratio preserved) and 3D (smooth camera animation)
- **Projection Methods**: PCA, UMAP, or manual dimension selection (requires raw embeddings)
- **Density Clustering**: WASM-based clustering (~500ms for 150k points)
- **Responsive Design**: ResizeObserver-based sizing, preserves zoom/pan on updates

### 🔍 Advanced Search
- **Unified Search Workflow**:
  1. Type in sidebar → local text filtering
  2. Auto-select first match
  3. Semantic search triggered on selection
  4. Results shown in sidebar list + bottom table
- **Semantic Search**:
  - By query text (embeds query and searches)
  - By ID (faster, uses existing embedding)
  - Multiple distance metrics: Cosine, L2, Inner Product
- **Multi-Source Highlighting**:
  - Text matches: Solid highlight (similarity = 1.0)
  - Semantic results: Gradient blue→gold based on similarity
  - Multi-layer glow effects with calculateLuminosity

### 🎨 Dynamic Coloring
- **Color Scales**:
  - Categorical: Presets (e.g., POS colors) + D3 20-color palette
  - Sequential: Viridis scale (0→1) for continuous data
  - Diverging: RdBu scale (-1→0→1) for bipolar data
- **Auto-Detection**: Intelligently detects label and category fields (2-100 unique values)
- **OKLch Color System**: Perceptually-uniform colors for dark/light themes

### 📊 Dataset Management
- **HuggingFace Integration**: Browse, preview, and embed any HF dataset
  - Dataset info (configs, splits, features)
  - Preview rows before embedding
  - Column selection with text template support
  - Portion strategies (first N, last N, random sample, full)
- **Local File Support**: Upload and embed CSV, JSON, Parquet files
  - Text embedding with column selection
  - Image embedding with ViT models
  - Pre-computed vector support
- **Collection Manager**: View, edit metadata, and delete collections

### 🎯 Interactive UI
- **Frosted Glass Tooltips**: Custom tooltip with warm gold tint
- **Constellation Lines**: Connect selected point to similar items
- **Resizable Panels**: DashboardPanel with plot, legend, and results table
- **Interactive Legend**: Click categories to mute/unmute, shows point counts per category
- **Show Only Highlighted**: Toggle to hide non-matching points
- **Show Labels**: Display text labels above highlighted points
- **Dark Mode**: next-themes with seamless switching

## Architecture

### Tech Stack
- **Framework**: Next.js 15, React 19, TypeScript 5
- **Visualization**: Plotly.js + react-plotly.js (WebGL)
- **Data**: Apollo Client 4 (GraphQL)
- **UI**: Shadcn UI (30+ Radix primitives), Tailwind CSS 4
- **Tables**: @tanstack/react-table
- **Colors**: d3-scale, d3-scale-chromatic
- **Clustering**: WASM (from embedding-atlas)
- **Layout**: react-resizable-panels
- **Notifications**: sonner

### Project Structure
```
embedding_visualization/
├── app/
│   ├── components/           # 21 UI components
│   ├── test-embed/           # Dataset embedding interface
│   │   ├── page.tsx
│   │   └── components/       # 8 embedding-specific components
│   ├── utils/                # Utility functions, clustering
│   ├── page.tsx              # Main visualization dashboard
│   ├── layout.tsx            # Root layout
│   ├── providers.tsx         # Apollo + theme providers
│   └── globals.css           # Tailwind + custom CSS
├── lib/
│   ├── hooks/                # 13 custom React hooks
│   ├── ui-primitives/        # 30+ Shadcn UI components
│   ├── graphql/              # GraphQL queries/mutations
│   ├── types/                # TypeScript interfaces
│   ├── utils/                # Color mapping, field detection
│   └── density-clustering/   # WASM clustering module
├── components/               # Legacy UI (scroll-area, dialog)
├── public/                   # Static assets
└── package.json              # Dependencies, scripts
```

### Key Components

**Visualization** (app/components/):
- `DashboardPanel`: Main layout orchestrator with resizable panels
- `ScatterPlot2D`: 2D Plotly visualization with density clustering
- `ScatterPlot3D`: 3D Plotly with spherical camera interpolation
- `EmbeddingSidebar`: Floating sidebar with controls + selected point info
- `VisualizationControls`: Projection, dimensions, color scale controls
- `Legend`: Dynamic category legend with point counts and click-to-toggle muting
- `SimilarItemsTable`: Sortable table of semantic search results
- `TextSearchResultsList`: Scrollable list of text matches
- `FrostedTooltip`: Custom frosted glass tooltip

**Embedding Interface** (app/test-embed/components/):
- `DatasetEmbeddingForm`: HuggingFace dataset selection
- `LocalFileEmbeddingForm`: Local file upload
- `CollectionManager`: View/edit/delete collections
- `EmbeddingModelSelector`: Provider and model selection

### Custom Hooks

**Data & Loading**:
- `useEmbeddingData`: Load collection, auto-detect display config
- `useCollections`: Load available collections
- `useVisualizationPoints`: Transform data to visualization points
- `useDensityClustering`: WASM clustering
- `useEmbedDataset`: GraphQL mutations for embedding

**Search & Interaction**:
- `useAppSearch`: Unified search orchestration
- `useSemanticSearch`: GraphQL semantic search (by query or ID)
- `useHighlightedIndices`: Combine text + semantic highlights
- `useCategoryData`: Compute category values and counts for Legend

**Utilities**:
- `useContainerDimensions`: ResizeObserver-based sizing
- `use-debounce-callback/value`: Debouncing
- `use-mobile`: Mobile detection
- `use-unmount`: Cleanup on unmount

### Data Flow
```
page.tsx (orchestration layer)
  ↓
useCollections() → get available collections
  ↓
useEmbeddingData() → load selected collection from GraphQL
  ↓
useVisualizationPoints() → compute 2D/3D points
  ↓
useAppSearch() → manage semantic search state
  ↓
useHighlightedIndices() → combine highlights
  ↓
DashboardPanel → render plots + sidebar + tables
```

### Type System

**Core Types** (lib/types/types.ts):
- `EmbeddingData`: metadata, ids, documents, itemMetadata, projections, displayConfig
- `Point2D/Point3D`: x, y, z, id, label, document, category, index, metadata
- `VisualizationState`: method, mode, colorByField, colorScaleType, searchQuery, mutedCategories, etc.
- `DisplayConfig`: labelField, categoryField, categoryValues, categoryName
- `SemanticSearchResult`: id, label, document, category, similarity, distance
- `HighlightMap`: Map<index, similarity>

## Configuration

### Backend Connection
The frontend connects to the GraphQL backend at `http://localhost:8000/graphql`.

Ensure the backend is running for full functionality:
```bash
cd /path/to/project/root
./start_backend.sh
```

### GraphQL Endpoint
Edit `lib/utils/apollo-client.ts` to change the GraphQL endpoint:
```typescript
const client = new ApolloClient({
  uri: "http://localhost:8000/graphql",
  // ...
});
```

### Color Customization
**Categorical colors**: Edit `lib/utils/color-mapping.ts` to customize presets
- POS colors: noun→blue, verb→green, adj→orange, adv→purple
- Unknown categories: D3 20-color palette

**Sequential/diverging scales**: Uses d3-scale-chromatic
- Sequential: Viridis (0→1)
- Diverging: RdBu (-1→0→1)

**OKLch color system**: Edit `app/globals.css` for theme colors
```css
:root {
  --background: oklch(98% 0.01 90);
  --foreground: oklch(20% 0.02 90);
  /* ... */
}
```

## Development

### Available Scripts
```bash
npm run dev     # Development server (http://localhost:3000)
npm run build   # Production build
npm start       # Start production server
npm run lint    # Run ESLint
npm run test    # Run tests (if configured)
```

### Adding a New Component
1. Create component in `app/components/` or `lib/ui-primitives/`
2. Import in parent component
3. Add types to `lib/types/types.ts` if needed
4. Use existing hooks for data loading, search, etc.

### Adding a New Hook
1. Create hook in `lib/hooks/`
2. Follow naming convention: `use-kebab-case.ts` or `useCamelCase.ts`
3. Export from hook file
4. Import and use in components

### Styling Guidelines
- Use Tailwind CSS utility classes for styling
- Use OKLch color system for theme-aware colors
- Follow Shadcn UI patterns for component composition
- Use `cn()` utility from `lib/utils/utils.ts` for conditional classes

## Troubleshooting

### Common Issues

**No Data Shown**:
- Ensure the backend is running at `http://localhost:8000`
- Check browser console for GraphQL errors
- Verify collections exist: Navigate to `/test-embed` → Collection Manager tab
- Try embedding a test dataset (e.g., "emotion" or "squad" from HuggingFace)

**Network Error / GraphQL Connection Failed**:
- Check that backend is running: `curl http://localhost:8000/health`
- Verify GraphQL endpoint: Open `http://localhost:8000/graphql` in browser
- Check firewall settings
- Review backend logs for errors

**Performance Issues (>100k points)**:
- Switch to 2D mode for better responsiveness
- Reduce marker size in plot config
- Enable "Show Only Highlighted" to reduce rendered points
- Use text search to filter before visualizing
- Consider using portion strategies when embedding (first N, random sample)

**Projection Not Updating**:
- Check that projections were computed during embedding
- For manual projection, ensure raw embeddings are available in collection
- Verify projection data exists in collection metadata

**Colors Not Updating**:
- Check that `colorByField` is set in VisualizationState
- Verify field has 2-100 unique values (required for categorical)
- For numeric coloring, select a numeric field and appropriate scale type

**Search Not Working**:
- Verify backend connection (semantic search requires GraphQL)
- Check that collection has valid embeddings
- Try different distance metrics (Cosine, L2, IP)
- Ensure search query is not empty

### Browser Compatibility
- **Recommended**: Chrome, Edge, Firefox (latest versions)
- **WebGL Required**: For visualization (most modern browsers)
- **WASM Required**: For clustering (all modern browsers)

### Performance Tips
- Use 2D mode for datasets >100k points
- Enable WebGL in browser settings if disabled
- Close other tabs/applications for better GPU performance
- Use "Show Only Highlighted" when exploring specific clusters
- Reduce browser zoom level if rendering is slow

## Contributing

When contributing to this frontend:
1. Follow TypeScript strict mode conventions
2. Add proper type annotations for all functions
3. Use existing hooks and utilities when possible
4. Keep components focused and single-purpose
5. Add JSDoc comments for complex logic
6. Test with both light and dark themes
7. Verify responsive behavior on different screen sizes

## License

See main project LICENSE file.
