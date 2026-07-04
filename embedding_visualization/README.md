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

**First Launch**: The first time you open the app, there are no collections to display. Click the "Embed" button in the top-right header to navigate to `/collections` and create your first collection from HuggingFace datasets or local files.

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
  - Categorical: Presets (e.g., POS colors) + D3 category10/20 palettes + 21 Crameri categorical palettes (100 colors each)
  - Sequential: 7 D3 scales (Viridis, Plasma, Turbo, etc.) + 24 Crameri scientific scales
  - Diverging: 6 D3 scales (BlueGold, RdBu, Spectral, etc.) + 10 Crameri diverging scales
  - Monochrome: Single-color opacity gradient (10%→100%)
- **Crameri Scientific Colormaps**: 60+ perceptually-uniform scales, lazy-loaded on demand (~8-12KB each)
- **Auto-Detection**: Intelligently detects label and category fields; numeric fields (≥20 unique) → sequential, string fields (≤100 unique) → categorical
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
- **Topic Extraction**: Extract topics from existing collections (HDBSCAN + c-TF-IDF + optional LLM labeling)
- **Topic Reduction**: Merge similar topics (AgglomerativeClustering or auto-HDBSCAN)
- **Real-Time Progress**: WebSocket-based progress tracking during embedding and topic extraction
- **Resumable Jobs**: Panel showing interrupted jobs with one-click resume

### 🎯 Interactive UI
- **Frosted Glass Tooltips**: Custom tooltip with warm gold tint
- **Constellation Lines**: Connect selected point to similar items
- **Resizable Panels**: DashboardPanel with plot, legend, and results table
- **Interactive Legend**: Click categories to mute/unmute, shows point counts per category
- **Show Only Highlighted**: Toggle to hide non-matching points
- **Show Labels**: Display text labels above highlighted points
- **Show Contours**: Toggle density cluster contours (2D only)
- **Hide Unclustered**: Toggle to hide noise points (topic_id=-1)
- **Tooltip Fields**: Configurable metadata fields in hover tooltip
- **Dark Mode**: next-themes with seamless switching

## Architecture

### Tech Stack
- **Framework**: Next.js 15, React 19, TypeScript 5
- **Visualization**: Plotly.js + react-plotly.js (WebGL)
- **Data**: Apollo Client 4 (GraphQL)
- **UI**: Shadcn UI (34 Radix primitives), Tailwind CSS 4
- **Tables**: @tanstack/react-table
- **Colors**: d3-scale, d3-scale-chromatic, Crameri scientific colormaps (60+ lazy-loaded)
- **Clustering**: WASM (from embedding-atlas)
- **Layout**: react-resizable-panels
- **Themes**: next-themes (dark mode)
- **Notifications**: sonner

### Project Structure
```
embedding_visualization/
├── app/
│   ├── components/           # 22 UI components
│   ├── collections/           # Dataset embedding interface
│   │   ├── page.tsx
│   │   └── components/       # 17 embedding-specific components
│   ├── page.tsx              # Main visualization dashboard
│   ├── layout.tsx            # Root layout
│   ├── providers.tsx         # Apollo + theme providers
│   └── globals.css           # Tailwind + OKLch theme CSS
├── lib/
│   ├── hooks/                # 14 custom React hooks
│   ├── ui-primitives/        # 34 Shadcn UI components
│   ├── graphql/              # GraphQL queries + mutations
│   ├── types/                # TypeScript interfaces
│   ├── utils/                # 9 utility modules (colors, fields, plots, etc.)
│   ├── colorMaps/            # Crameri scientific colormaps (60+ JSON files)
│   └── density-clustering/   # WASM clustering module
├── public/                   # Static assets
└── package.json              # Dependencies, scripts
```

### Key Components

**Visualization** (app/components/ - 22 components):
- `DashboardPanel`: Main layout orchestrator with resizable panels (plot, legend, results table)
- `ScatterPlot2D`: 2D Plotly visualization with WebGL, density clustering, aspect ratio preservation
- `ScatterPlot3D`: 3D Plotly with smooth spherical camera interpolation and cubic easing
- `EmbeddingSidebar`: Floating sidebar with controls + selected point info (offcanvas collapsible)
- `VisualizationControls`: Projection method, dimensions, manual selection, color scale controls
- `ColorScaleSelector`: Color scale type/name selector (categorical/sequential/diverging/monochrome)
- `Legend`: Dynamic category legend with point counts, click-to-toggle muting, gradient legend
- `SimilarItemsTable`: Sortable table of semantic search results with dynamic metadata columns
- `TextSearchResultsList`: Scrollable list of text matches in sidebar
- `SelectedPointCard`: Displays selected point details with metadata
- `FrostedTooltip`: Custom frosted glass tooltip with warm gold tint
- `AppHeader`: Collection selector, semantic search bar, theme toggle, embed button
- `SearchSidebar`: Search interface sidebar panel
- `DebouncedSearchInput`: Search input with debounce for performance
- `SidebarInfo`: Sidebar info panel
- `StatusLayout`: Full-page status wrapper
- `LoadingScreen`: Loading indicators
- `ErrorScreen`: Config-based error display
- `VisualizationStatus`: Visualization status display
- `AppFooter`: Footer component

**Embedding Interface** (app/collections/components/ - 17 components):
- `HuggingFaceTab`: Complete HuggingFace dataset embedding UI
- `LocalFileTab`: Local file embedding UI
- `CollectionManagerTab`: Collection management with edit/delete
- `TopicExtractionCard`: UI for extracting topics from existing collections
- `TopicConfigForm`: Topic extraction configuration (min size, keywords, LLM, reduction)
- `EmbeddingProgressModal`: Real-time progress modal with WebSocket subscription
- `EmbeddingProgressCard`: Progress card for embedding status
- `JobsPanel`: Interrupted jobs panel with resume capability
- `DataSourceTabs`: Tab selector with icons
- `FileUploadZone`: Drag-drop file upload
- `DataTypeSelector`: TEXT/IMAGE/VECTOR selection
- `ColumnSelector`: Column selection + text template
- `PortionSelector`: Row range/sample selection
- `SplitSelector`: HF split selector
- `DatasetInfoDisplay`: Preview data display
- `InlineEditableField`: Inline editable metadata fields (text/select)
- `AddFieldForm`: Add custom metadata fields with validation

### Custom Hooks

**Data & Loading**:
- `useEmbeddingData`: Load collection, auto-detect display config
- `useCollections`: Load available collections
- `useVisualizationPoints`: Transform data to visualization points
- `useDensityClustering`: WASM clustering (~500ms for 150k points)
- `useEmbedDataset`: GraphQL mutations for embedding, topic extraction, topic reduction

**Search & Interaction**:
- `useAppSearch`: Unified search orchestration
- `useSemanticSearch`: GraphQL semantic search (by query or ID)
- `useHighlightedIndices`: Combine semantic search + topic highlights
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
- Visualization state lives in the **Zustand store** (`lib/stores/useVisualizationStore.ts`), not a types.ts interface — method, mode, colorByField, the `ColorScale` union, mutedCategories, tooltipFields, hideUnclustered, categoricalPalette, etc.
- `ColorScale`: discriminated union `{ type: 'categorical' } | { type: 'sequential'; scaleName } | { type: 'diverging'; scaleName } | { type: 'monochrome'; baseColor }` (replaces the old flat colorScaleType/sequentialScaleName/divergingScaleName/monochromeColor fields)
- `DisplayConfig`: labelField, categoryField, categoryValues, categoryName
- `SemanticSearchResult`: id, label, document, category, similarity, distance, metadata
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
**Categorical colors**: Edit `lib/utils/categoryColors.ts` to customize presets
- POS colors: noun→blue, verb→orange, adj→green, adv→red, adj.sat→purple
- Topic preset: -1 (unclustered) → gray, others get dynamic colors
- D3 palettes: category10 (≤10 values), category20 (≤20 values)
- Crameri categorical: 21 palettes with 100 distinct colors each

**Sequential/diverging/monochrome scales**: D3 + Crameri scientific colormaps
- Sequential: 7 D3 (sinebow, viridis, cividis, turbo, plasma, inferno, magma) + 24 Crameri
- Diverging: 6 D3 (blueGold, rdBu, spectral, piYG, puOr, brBG) + 10 Crameri
- Monochrome: Single-color opacity gradient (10%→100%)

**Adding Crameri scales**: Add JSON file to `lib/colorMaps/colormaps/`, update `index.json`

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
- Verify collections exist: Navigate to `/collections` → Collection Manager tab
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
- If all points appear gray, check `buildCategoryColorMap()` in `categoryColors.ts` — presets only override specific values, others get dynamic colors
- Ensure Crameri colormaps are loaded before use: `loadCrameriColormap()` must be called before `getCrameriPlotlyScale()`

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
