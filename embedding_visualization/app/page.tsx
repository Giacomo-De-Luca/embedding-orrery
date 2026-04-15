'use client';

import { useEffect, useCallback, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useShallow } from 'zustand/react/shallow';
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
import { isInTemporalRange } from '../lib/utils/temporalFilters';
import { useVisualizationStore } from '../lib/stores/useVisualizationStore';
import { defaultColorScaleForType } from '../lib/types/types';
import type { VisualizationState, HighlightMap } from '../lib/types/types';



export default function Home() {
  const { collections, loading: collectionsLoading, error: collectionsError } = useCollections();
  const searchParams = useSearchParams();
  const router = useRouter();
  const collectionFromUrl = searchParams.get('collection');
  const colorByFromUrl = searchParams.get('colorBy');
  const isInitialLoad = useRef(true);
  const initialColorByRef = useRef(colorByFromUrl);

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
  const searchQuery = store((s) => s.searchQuery);
  const distanceMetric = store((s) => s.distanceMetric);
  const temporalRange = store((s) => s.temporalRange);

  // Compat bridge: construct old VisualizationState shape for un-migrated children
  const visualizationState = useVisualizationStore(useShallow((s): VisualizationState => ({
    method: s.method,
    mode: s.mode,
    selectedDimensions: s.selectedDimensions,
    colorByField: s.colorByField,
    colorScaleType: s.colorScale.type,
    sequentialScaleName: s.colorScale.type === 'sequential' ? s.colorScale.scaleName : undefined,
    divergingScaleName: s.colorScale.type === 'diverging' ? s.colorScale.scaleName : undefined,
    monochromeColor: s.colorScale.type === 'monochrome' ? s.colorScale.baseColor : undefined,
    categoricalPalette: s.categoricalPalette,
    searchQuery: s.searchQuery,
    distanceMetric: s.distanceMetric,
    showOnlyHighlighted: s.showOnlyHighlighted,
    showLabels: s.showLabels,
    showContours: s.showContours,
    mutedCategories: s.mutedCategories,
    tooltipFields: s.tooltipFields,
    hideUnclustered: s.hideUnclustered,
    nestedColorMode: s.nestedColorMode,
    nebulaMode: s.nebulaMode,
    showClusterLabels: s.showClusterLabels,
    temporalRange: s.temporalRange,
    hideFilteredPoints: s.hideFilteredPoints,
    mutedPointOpacity: s.mutedPointOpacity,
  })));

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
    }
    const newSearch = `?${params.toString()}`;
    // Only navigate if the URL actually changed
    if (newSearch !== window.location.search) {
      router.replace(newSearch, { scroll: false });
    }
  }, [selectedCollection, colorByField, router]);

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
    data,
    selectedCollection,
    distanceMetric ?? 'COSINE',
    queryPromptName,
    data?.metadata?.embedding_prompt,
  );

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
  );

  // Compat bridge: convert old Partial<VisualizationState> writes to store actions
  const updateState = useCallback((newState: Partial<VisualizationState>) => {
    const s = useVisualizationStore.getState();
    // Map old flat color fields to the new union if any color fields are being set
    const patch: Partial<import('../lib/stores/useVisualizationStore').VisualizationStoreState> = {};
    for (const [key, value] of Object.entries(newState)) {
      switch (key) {
        case 'colorScaleType': {
          // When colorScaleType changes, rebuild the entire colorScale union with defaults
          const type = value as import('../lib/types/types').ColorScaleType;
          patch.colorScale = defaultColorScaleForType(type);
          break;
        }
        case 'sequentialScaleName':
          if (s.colorScale.type === 'sequential') {
            patch.colorScale = { type: 'sequential', scaleName: value as import('../lib/utils/categoryColors').SequentialScaleName };
          }
          break;
        case 'divergingScaleName':
          if (s.colorScale.type === 'diverging') {
            patch.colorScale = { type: 'diverging', scaleName: value as import('../lib/utils/categoryColors').DivergingScaleName };
          }
          break;
        case 'monochromeColor':
          if (s.colorScale.type === 'monochrome') {
            patch.colorScale = { type: 'monochrome', baseColor: value as string };
          }
          break;
        default:
          // Pass through all other fields directly
          (patch as Record<string, unknown>)[key] = value;
      }
    }
    s.updatePartial(patch);
  }, []);

  const visualizationPoints = useVisualizationPoints(data, visualizationState);
  const { filteredPoints2d, filteredPoints3d, highlightedIndices } = visualizationPoints;

  // Compute text search results from highlighted indices, filtered by temporal range
  const textSearchResults = useMemo(() => {
    if (!highlightedIndices || highlightedIndices.size === 0) return [];
    const points = mode === '2d' ? filteredPoints2d : filteredPoints3d;
    return points.filter(p =>
      highlightedIndices.has(p.index) &&
      (!temporalRange || isInTemporalRange(p.metadata, temporalRange))
    );
  }, [highlightedIndices, filteredPoints2d, filteredPoints3d, mode, temporalRange]);

  // Auto-select first semantic search result when a text-query search completes
  // This triggers the camera fly-to animation in ScatterPlot3D
  useEffect(() => {
    if (semanticSearchResults && semanticSearchResults.length > 0 && searchType === 'text') {
      const firstResultId = semanticSearchResults[0].id;
      const points = mode === '3d' ? filteredPoints3d : filteredPoints2d;
      const matchingPoint = points.find(p => p.id === firstResultId);
      if (matchingPoint) {
        setSelectedPoint(matchingPoint);
      }
    }
  }, [semanticSearchResults, filteredPoints2d, filteredPoints3d, mode, setSelectedPoint, searchType]);

  // Combine semantic search highlights and topic highlights (text search handled by muting, not glow)
  // Selected point is excluded — it has its own overlay traces in ScatterPlot3D
  const combinedHighlightedIndices: HighlightMap | undefined = useHighlightedIndices(
    semanticSearchResults,
    data,
    topicSearch.topicHighlightMap
  );

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
    setQueryPromptName(null);
    store.getState().resetForCollectionChange();
  }, [selectedCollection, resetSearch]);

  // Apply colorBy from URL once data loads, then mark initial load complete
  useEffect(() => {
    if (!isInitialLoad.current || colorFieldOptions.length === 0) return;
    isInitialLoad.current = false;
    const initialColorBy = initialColorByRef.current;
    if (initialColorBy) {
      const fieldOption = colorFieldOptions.find(f => f.field === initialColorBy);
      if (fieldOption) {
        store.getState().setColorByField(initialColorBy, fieldOption.recommendedScale);
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
              onSemanticSearch={handleSemanticSearch}
              searchLoading={searchLoading}
              activePanel={activePanel}
              onToggleControls={toggleControls}
              onToggleSearch={toggleSearch}
              onToggleAnalytics={toggleAnalytics}
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
                  state={visualizationState}
                  points2d={filteredPoints2d}
                  points3d={filteredPoints3d}
                  highlightedIndices={combinedHighlightedIndices}
                  textSearchHighlights={highlightedIndices}
                  onPointClick={handlePointClick}
                  selectedPoint={selectedPoint}
                  semanticSearchResults={semanticSearchResults}
                  searchQueryLabel={searchQueryLabel}
                  onStateChange={updateState}
                  embeddingDim={data.metadata.embedding_dim}
                  metadata={{
                    pca_2d_variance: data.metadata.pca_2d_variance,
                    pca_3d_variance: data.metadata.pca_3d_variance,
                  }}
                  searchQuery={searchQuery}
                  highlightedCount={combinedHighlightedIndices?.size}
                  colorFieldOptions={colorFieldOptions}
                  textSearchResults={textSearchResults}
                  onTextResultClick={handlePointClick}
                  activePanel={activePanel}
                  queryPromptName={queryPromptName}
                  onQueryPromptNameChange={setQueryPromptName}
                  availableFields={data.availableFields}
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
