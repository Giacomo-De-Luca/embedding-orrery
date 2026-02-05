'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
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
import type { VisualizationState, HighlightMap } from '../lib/types/types';



export default function Home() {
  const { collections, loading: collectionsLoading, error: collectionsError } = useCollections();
  const searchParams = useSearchParams();
  const router = useRouter();
  const collectionFromUrl = searchParams.get('collection');
  const colorByFromUrl = searchParams.get('colorBy');
  const isInitialLoad = useRef(true);

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

  const { data, loading, error, colorFieldOptions, defaultTooltipFields } = useEmbeddingData(selectedCollection);

  const [visualizationState, setVisualizationState] = useState<VisualizationState>({
    method: 'umap',
    mode: '3d',
    colorByField: null,
    searchQuery: '',
    selectedDimensions: [0, 1, 2],
    distanceMetric: 'COSINE',
  });

  // Sync URL when collection or colorBy changes
  useEffect(() => {
    if (selectedCollection) {
      const params = new URLSearchParams();
      params.set('collection', selectedCollection);
      if (visualizationState.colorByField) {
        params.set('colorBy', visualizationState.colorByField);
      } else {
        params.delete('colorBy');
      }
      router.replace(`?${params.toString()}`, { scroll: false });
    }
  }, [selectedCollection, visualizationState.colorByField, router]);

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

  // Keyboard shortcuts: ⌘B for controls, ⌘K for search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        toggleControls();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        toggleSearch();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleControls, toggleSearch]);

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
    visualizationState.colorByField ?? null,
    visualizationState.distanceMetric ?? 'COSINE',
    queryPromptName,
    data?.metadata?.embedding_prompt_name
  );

  // Update visualization state
  const updateState = useCallback((newState: Partial<VisualizationState>) => {
    setVisualizationState(prev => ({ ...prev, ...newState }));
  }, []);

  const visualizationPoints = useVisualizationPoints(data, visualizationState);
  const { filteredPoints2d, filteredPoints3d, highlightedIndices } = visualizationPoints;

  // Compute text search results from highlighted indices
  const textSearchResults = useMemo(() => {
    if (!highlightedIndices || highlightedIndices.size === 0) return [];
    const points = visualizationState.mode === '2d' ? filteredPoints2d : filteredPoints3d;
    return points.filter(p => highlightedIndices.has(p.index));
  }, [highlightedIndices, filteredPoints2d, filteredPoints3d, visualizationState.mode]);

  // Track previous search query to detect changes
  const prevSearchQuery = useRef<string | undefined>(undefined);

  // Auto-select first text search result and trigger semantic search
  useEffect(() => {
    const currentQuery = visualizationState.searchQuery?.trim();
    const queryChanged = currentQuery !== prevSearchQuery.current;
    prevSearchQuery.current = currentQuery;

    // Only auto-select when query changes and we have results
    if (queryChanged && currentQuery && textSearchResults.length > 0) {
      handlePointClick(textSearchResults[0]);
    }
  }, [visualizationState.searchQuery, textSearchResults, handlePointClick]);

  // Combine text search highlights with semantic search highlights
  // Pass selectedPoint's index so it's included in highlights (semantic search returns similar items, not the query itself)
  const combinedHighlightedIndices: HighlightMap | undefined = useHighlightedIndices(
    highlightedIndices,
    semanticSearchResults,
    data,
    selectedPoint?.index
  );

  // Initialize tooltipFields with smart defaults when data loads
  useEffect(() => {
    if (defaultTooltipFields.length > 0) {
      setVisualizationState(prev => {
        // Only set if tooltipFields hasn't been initialized yet (don't override user selections)
        if (prev.tooltipFields === undefined) {
          return { ...prev, tooltipFields: defaultTooltipFields };
        }
        return prev;
      });
    }
  }, [defaultTooltipFields]);

  // Reset state when collection changes (skip on initial URL-driven load so colorBy isn't cleared)
  useEffect(() => {
    if (isInitialLoad.current) return;
    resetSearch();
    setQueryPromptName(null);
    setVisualizationState(prev => ({ ...prev, colorByField: null, mutedCategories: [], tooltipFields: undefined }));
  }, [selectedCollection, resetSearch]);

  // Apply colorBy from URL once data loads, then mark initial load complete
  useEffect(() => {
    if (!isInitialLoad.current || colorFieldOptions.length === 0) return;
    isInitialLoad.current = false;
    if (colorByFromUrl) {
      const fieldOption = colorFieldOptions.find(f => f.field === colorByFromUrl);
      if (fieldOption) {
        setVisualizationState(prev => ({
          ...prev,
          colorByField: colorByFromUrl,
          colorScaleType: fieldOption.recommendedScale,
        }));
      }
    }
  }, [colorFieldOptions, colorByFromUrl]);

  // Reset muted categories and hideUnclustered when colorByField changes (categories are now different)
  useEffect(() => {
    setVisualizationState(prev => ({ ...prev, mutedCategories: [], hideUnclustered: false }));
  }, [visualizationState.colorByField]);

  // Auto-select first search result when semantic search completes
  // Only for text searches - point clicks already set selectedPoint in the handler
  useEffect(() => {
    if (semanticSearchResults && semanticSearchResults.length > 0 && searchType === 'text') {
      const firstResultId = semanticSearchResults[0].id;
      const points = visualizationState.mode === '3d' ? filteredPoints3d : filteredPoints2d;
      const matchingPoint = points.find(p => p.id === firstResultId);
      if (matchingPoint) {
        setSelectedPoint(matchingPoint);
      }
    }
  }, [semanticSearchResults, filteredPoints2d, filteredPoints3d, visualizationState.mode, setSelectedPoint, searchType]);

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
                  searchQuery={visualizationState.searchQuery}
                  highlightedCount={combinedHighlightedIndices?.size}
                  colorFieldOptions={colorFieldOptions}
                  textSearchResults={textSearchResults}
                  onTextResultClick={handlePointClick}
                  activePanel={activePanel}
                  queryPromptName={queryPromptName}
                  onQueryPromptNameChange={setQueryPromptName}
                  availableFields={data.availableFields}
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
