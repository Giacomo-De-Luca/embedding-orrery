'use client';

import { useEffect, useCallback, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { AppHeader } from './components/AppHeader';
import { AppFooter } from './components/AppFooter';
import { DashboardPanel, type ActivePanel } from './components/DashboardPanel';
import { SidebarInset, SidebarProvider } from '@/lib/ui-primitives/sidebar';
import { useEmbeddingData } from '../lib/hooks/useEmbeddingData';
import { useCollections } from '../lib/hooks/useCollections';
import { useVisualizationPoints } from '../lib/hooks/useVisualizationPoints';
import { useHighlightedIndices } from '../lib/hooks/useHighlightedIndices';
import { useAppSearch } from '../lib/hooks/useAppSearch';
import { useTopicSearch } from '../lib/hooks/useTopicSearch';
import { useTextSearch } from '../lib/hooks/useTextSearch';
import { usePromptHighlight, buildPromptHighlightResults } from '../lib/hooks/usePromptHighlight';
import { useDocumentFeatureSearch } from '../lib/hooks/useDocumentFeatureSearch';
import { isInTemporalRange } from '../lib/utils/temporalFilters';
import { useVisualizationStore } from '../lib/stores/useVisualizationStore';
import type { HighlightMap } from '../lib/types/types';
import { getSaeInfo, getSaeInfoFromMetadata } from '../lib/utils/saeCollections';
import { serializeColorScale, deserializeColorScale } from '../lib/utils/colorScaleUrl';

const EMPTY_METADATA: Record<string, unknown>[] = [];

export default function Home() {
  const { collections, loading: collectionsLoading, error: collectionsError } = useCollections();
  const searchParams = useSearchParams();
  const router = useRouter();
  const collectionFromUrl = searchParams.get('collection');
  const colorByFromUrl = searchParams.get('colorBy');
  const isInitialLoad = useRef(true);
  const initialColorByRef = useRef(colorByFromUrl);
  // Color-scheme params captured from the URL on first render (applied once data loads)
  const initialColorScaleRef = useRef(deserializeColorScale({
    scale: searchParams.get('scale'),
    scaleName: searchParams.get('scaleName'),
    color: searchParams.get('color'),
  }));
  const initialPaletteRef = useRef(searchParams.get('palette'));

  // Default to the first available collection
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);

  // Select collection from URL param, or auto-select first collection
  useEffect(() => {
    if (collections && !selectedCollection) {
      if (collectionFromUrl && collections[collectionFromUrl]) {
        setSelectedCollection(collectionFromUrl);
      } else {
        const firstCollection = Object.keys(collections)[0];
        if (firstCollection) {
          setSelectedCollection(firstCollection);
        }
      }
    }
  }, [collections, selectedCollection, collectionFromUrl]);

  // --- Zustand store for visualization state ---
  const store = useVisualizationStore;
  const method = store((s) => s.method);
  const mode = store((s) => s.mode);
  const colorByField = store((s) => s.colorByField);
  const colorScale = store((s) => s.colorScale);
  const categoricalPalette = store((s) => s.categoricalPalette);
  const searchQuery = store((s) => s.searchQuery);
  const textSearchConfig = store((s) => s.textSearchConfig);
  const distanceMetric = store((s) => s.distanceMetric);
  const temporalRange = store((s) => s.temporalRange);

  const { data, loading, error, colorFieldOptions, defaultTooltipFields } = useEmbeddingData(
    selectedCollection,
    method,
    mode,
  );

  // Sync URL when collection or colorBy changes
  useEffect(() => {
    if (!selectedCollection) return;
    const params = new URLSearchParams();
    params.set('collection', selectedCollection);
    // During initial load, preserve colorBy from the original URL until state catches up
    const effectiveColorBy = colorByField
      ?? (isInitialLoad.current ? initialColorByRef.current : null);
    if (effectiveColorBy) {
      params.set('colorBy', effectiveColorBy);
      // Encode the color scheme alongside the field. During initial load the
      // store hasn't applied the URL state yet, so prefer the original URL refs.
      const effectiveScale = isInitialLoad.current
        ? (initialColorScaleRef.current ?? colorScale)
        : colorScale;
      const effectivePalette = isInitialLoad.current
        ? (initialPaletteRef.current ?? categoricalPalette)
        : categoricalPalette;
      const colorParams = serializeColorScale(effectiveScale, effectivePalette ?? undefined);
      if (colorParams.scale) params.set('scale', colorParams.scale);
      if (colorParams.scaleName) params.set('scaleName', colorParams.scaleName);
      if (colorParams.color) params.set('color', colorParams.color);
      if (colorParams.palette) params.set('palette', colorParams.palette);
    }
    const newSearch = `?${params.toString()}`;
    // Only navigate if the URL actually changed
    if (newSearch !== window.location.search) {
      router.replace(newSearch, { scroll: false });
    }
  }, [selectedCollection, colorByField, colorScale, categoricalPalette, router]);

  // Get topics for selected collection
  const selectedCollectionTopics = useMemo(() => {
    if (!collections || !selectedCollection) return undefined;
    return collections[selectedCollection]?.topics;
  }, [collections, selectedCollection]);

  // Query prompt name for semantic search (null=none, 'auto'=auto-detect, or explicit value)
  const [queryPromptName, setQueryPromptName] = useState<string | null>(null);

  // Panel state for dual sidebars (controls vs search)
  const [activePanel, setActivePanel] = useState<ActivePanel>(null);

  const toggleControls = useCallback(() => {
    setActivePanel(prev => prev === 'controls' ? null : 'controls');
  }, []);

  const toggleSearch = useCallback(() => {
    setActivePanel(prev => prev === 'search' ? null : 'search');
  }, []);

  const toggleAnalytics = useCallback(() => {
    setActivePanel(prev => prev === 'analytics' ? null : 'analytics');
  }, []);

  // Keyboard shortcuts: ⌘B for controls, ⌘K for search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        toggleControls();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        toggleSearch();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'j') {
        e.preventDefault();
        toggleAnalytics();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleControls, toggleSearch, toggleAnalytics]);

  // Topic search hook (instantiated before useAppSearch so topicFilters is available)
  const topicSearch = useTopicSearch(
    selectedCollectionTopics,
    selectedCollection,
    distanceMetric ?? 'COSINE',
    queryPromptName,
    data?.metadata?.embedding_prompt,
  );

  const { points2d, points3d } = useVisualizationPoints(data, { method, searchQuery });

  // Resolve a search result ID to its visualization point (used by useAppSearch
  // to auto-select the first result in the same React batch as setSemanticSearchResults).
  const resolvePoint = useCallback((id: string) => {
    const pts = mode === '3d' ? points3d : points2d;
    return pts.find(p => p.id === id);
  }, [points2d, points3d, mode]);

  // Use the new custom hook for search logic
  const {
    selectedPoint,
    setSelectedPoint,
    semanticSearchResults,
    setSemanticSearchResults,
    searchQueryLabel,
    searchType,
    handleSemanticSearch,
    handlePointClick,
    searchLoading,
    resetSearch
  } = useAppSearch(
    selectedCollection,
    colorByField ?? null,
    distanceMetric ?? 'COSINE',
    queryPromptName,
    data?.metadata?.embedding_prompt,
    topicSearch.topicFilters,
    temporalRange,
    resolvePoint,
  );

  // Server-side text search
  const {
    highlightedIndices: textSearchHighlightedIndices,
    loading: textSearchLoading,
  } = useTextSearch(selectedCollection, searchQuery, textSearchConfig, data?.ids);

  // SAE prompt activation highlight — prefer metadata-based lookup, fall back to hardcoded
  const saeInfo = getSaeInfoFromMetadata(data?.metadata) ?? getSaeInfo(selectedCollection);
  const [promptMaxDensity, setPromptMaxDensity] = useState<number | null>(null);
  const promptHighlight = usePromptHighlight(saeInfo, data?.itemMetadata ?? EMPTY_METADATA, promptMaxDensity);

  // Document feature search (two-hop: label → features → documents)
  const featureSearch = useDocumentFeatureSearch(selectedCollection, saeInfo);

  // Wrap search handlers to clear other highlights on new search actions
  const wrappedHandlePointClick = useCallback(
    (point: Parameters<typeof handlePointClick>[0]) => {
      promptHighlight.clear();
      featureSearch.clearFeatures();
      handlePointClick(point);
    },
    [handlePointClick, promptHighlight.clear, featureSearch.clearFeatures],
  );
  const wrappedHandleSemanticSearch = useCallback(
    (query: string) => {
      promptHighlight.clear();
      featureSearch.clearFeatures();
      handleSemanticSearch(query);
    },
    [handleSemanticSearch, promptHighlight.clear, featureSearch.clearFeatures],
  );
  const handleFeatureSearchResultClick = useCallback(
    (rowIndex: number) => {
      const pts = mode === '3d' ? points3d : points2d;
      const point = pts.find(p => p.index === rowIndex);
      if (point) {
        handlePointClick(point);
      }
    },
    [mode, points2d, points3d, handlePointClick],
  );

  // Build table-ready results from prompt highlight features
  const promptHighlightResults = useMemo(() => {
    if (!data || promptHighlight.topFeatures.length === 0) return null;
    return buildPromptHighlightResults(
      promptHighlight.topFeatures,
      data.ids,
      data.documents,
      data.itemMetadata,
      data.displayConfig.labelField,
    );
  }, [promptHighlight.topFeatures, data]);

  // Compute text search results from highlighted indices, filtered by temporal range
  const textSearchResults = useMemo(() => {
    if (!textSearchHighlightedIndices || textSearchHighlightedIndices.size === 0) return [];
    const points = mode === '2d' ? points2d : points3d;
    return points.filter(p =>
      textSearchHighlightedIndices.has(p.index) &&
      (!temporalRange || isInTemporalRange(p.metadata, temporalRange))
    );
  }, [textSearchHighlightedIndices, points2d, points3d, mode, temporalRange]);


  // Combine semantic search highlights, topic highlights, and text search highlights.
  // Text search glow only activates when no semantic search is active — clicking a
  // text result triggers semantic search which naturally takes over the glow.
  // Selected point is excluded — it has its own overlay traces in ScatterPlot3D.
  const baseHighlightedIndices: HighlightMap | undefined = useHighlightedIndices(
    semanticSearchResults,
    data,
    semanticSearchResults && semanticSearchResults.length > 0 ? null : textSearchHighlightedIndices,
  );
  // Prompt highlight > feature search > semantic + text highlights
  const combinedHighlightedIndices = promptHighlight.highlightMap ?? featureSearch.highlightMap ?? baseHighlightedIndices;

  // Initialize tooltipFields with smart defaults when data loads
  useEffect(() => {
    if (defaultTooltipFields.length > 0) {
      store.getState().initTooltipFields(defaultTooltipFields);
    }
  }, [defaultTooltipFields]);

  // Reset state when collection changes (skip on initial URL-driven load so colorBy isn't cleared)
  useEffect(() => {
    if (isInitialLoad.current) return;
    resetSearch();
    promptHighlight.clear();
    featureSearch.clearFeatures();
    setQueryPromptName(null);
    store.getState().resetForCollectionChange();
  }, [selectedCollection, resetSearch, promptHighlight.clear, featureSearch.clearFeatures]);

  // Apply colorBy from URL once data loads, then mark initial load complete
  useEffect(() => {
    if (!isInitialLoad.current || colorFieldOptions.length === 0) return;
    isInitialLoad.current = false;
    const initialColorBy = initialColorByRef.current;
    if (initialColorBy) {
      const fieldOption = colorFieldOptions.find(f => f.field === initialColorBy);
      if (fieldOption) {
        const urlScale = initialColorScaleRef.current;
        // When the URL carries an explicit scale, set the field WITHOUT a recommended
        // scale (so setColorByField preserves the current colorScale rather than
        // overwriting it with the field default) and then apply the URL scale.
        store.getState().setColorByField(
          initialColorBy,
          urlScale ? undefined : fieldOption.recommendedScale,
        );
        if (urlScale) store.getState().setColorScale(urlScale);
        const urlPalette = initialPaletteRef.current;
        if (urlPalette) store.getState().setCategoricalPalette(urlPalette);
      }
    }
  }, [colorFieldOptions]);

  // Auto-reset of mutedCategories on colorByField change is handled by the store subscription

  return (
    <SidebarProvider>
      <SidebarInset className=" relative ">
        <div className="absolute top-0 left-0 right-0 z-50 p-2 pointer-events-none">
          <div className="pointer-events-auto  rounded-lg ">
            <AppHeader
              collections={collections}
              collectionsLoading={collectionsLoading}
              collectionsError={collectionsError}
              selectedCollection={selectedCollection}
              onCollectionChange={setSelectedCollection}
              totalWords={data?.metadata.total_items}
              embeddingDim={data?.metadata.embedding_dim}
              onSemanticSearch={wrappedHandleSemanticSearch}
              searchLoading={searchLoading}
              activePanel={activePanel}
              onToggleControls={toggleControls}
              onToggleSearch={toggleSearch}
              onToggleAnalytics={toggleAnalytics}
              saeInfo={saeInfo}
            />
          </div>
        </div>
        <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
          {loading ? (
            <div className="flex flex-1 items-center justify-center rounded-xl border bg-card p-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                <p className="text-muted-foreground">Loading embedding data...</p>
              </div>
            </div>
          ) : error ? (
            <div className="rounded-xl border border-destructive/50 bg-destructive/10 p-6">
              <h3 className="text-destructive font-semibold mb-2">Error Loading Data</h3>
              <p className="text-destructive/90 mb-4">{error.message}</p>
              <p className="text-sm text-muted-foreground">
                Make sure you have run the projection computation script:
                <code className="block mt-2 bg-background p-2 rounded border">
                  uv run python interpretability/compute_projections.py
                </code>
              </p>
            </div>
          ) : data ? (
            <>
                <DashboardPanel
                  points2d={points2d}
                  points3d={points3d}
                  highlightedIndices={combinedHighlightedIndices}
                  textSearchHighlights={textSearchHighlightedIndices}
                  textSearchLoading={textSearchLoading}
                  onPointClick={wrappedHandlePointClick}
                  selectedPoint={selectedPoint}
                  semanticSearchResults={semanticSearchResults}
                  searchQueryLabel={searchQueryLabel}
                  embeddingDim={data.metadata.embedding_dim}
                  saeInfo={saeInfo}
                  promptHighlightStatus={promptHighlight.status}
                  promptHighlightError={promptHighlight.error}
                  promptHighlightActivePrompt={promptHighlight.activePrompt}
                  onPromptHighlightSubmit={promptHighlight.submit}
                  onPromptHighlightClear={promptHighlight.clear}
                  promptHighlightResults={promptHighlightResults}
                  promptMaxDensity={promptMaxDensity}
                  onPromptMaxDensityChange={setPromptMaxDensity}
                  featureSearch={featureSearch}
                  onFeatureSearchResultClick={handleFeatureSearchResultClick}
                  metadata={{
                    pca_2d_variance: data.metadata.pca_2d_variance,
                    pca_3d_variance: data.metadata.pca_3d_variance,
                  }}
                  searchQuery={searchQuery}
                  highlightedCount={combinedHighlightedIndices?.size}
                  colorFieldOptions={colorFieldOptions}
                  textSearchResults={textSearchResults}
                  onTextResultClick={wrappedHandlePointClick}
                  activePanel={activePanel}
                  queryPromptName={queryPromptName}
                  onQueryPromptNameChange={setQueryPromptName}
                  availableFields={data.availableFields}
                  itemMetadata={data.itemMetadata}
                  topics={selectedCollectionTopics}
                  topicSearchMode={topicSearch.mode}
                  onTopicSearchModeChange={topicSearch.setMode}
                  topicDirectQuery={topicSearch.directQuery}
                  onTopicDirectQueryChange={topicSearch.setDirectQuery}
                  topicFilteredTopics={topicSearch.filteredTopics}
                  topicSemanticQuery={topicSearch.semanticQuery}
                  onTopicSemanticQueryChange={topicSearch.setSemanticQuery}
                  onTopicSemanticSearch={topicSearch.searchTopicsBySimilarity}
                  topicSemanticResults={topicSearch.semanticResults}
                  topicSemanticLoading={topicSearch.semanticLoading}
                  selectedTopicIds={topicSearch.selectedTopicIds}
                  onToggleTopic={topicSearch.toggleTopic}
                  onSelectAllTopics={topicSearch.selectAll}
                  onClearAllTopics={topicSearch.clearAll}
                />
              {/*<AppFooter
                    timestamp={data.metadata.timestamp}
                    selectedCollection={selectedCollection}
                />*/}
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center rounded-xl border bg-muted p-12">
              <p className="text-muted-foreground">Select a collection to view embeddings</p>
            </div>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
