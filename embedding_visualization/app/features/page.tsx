'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery, useLazyQuery } from '@apollo/client/react';
import Link from 'next/link';
import { ArrowLeft, Sparkles, Sun, Moon } from 'lucide-react';
import { useTheme } from 'next-themes';
import {
  GET_SAE_MODELS,
  GET_SAE_FEATURE,
  GET_SAE_ACTIVATIONS,
  SEARCH_SAE_FEATURES,
  SEMANTIC_SEARCH,
  GET_SAE_FEATURE_DENSITIES,
  GET_SAE_ACTIVATIONS_BY_QUANTILE,
} from '@/lib/graphql/queries';
import type {
  SaeModelInfo, SaeFeature, SaeActivation,
  SaeFeatureSearchResult, SaeActivationQuantileGroup,
  SteeringConfig, SteeringFeature,
} from '@/lib/types/types';
import { FeatureHeader } from './components/FeatureHeader';
import { FeatureDetailCard } from './components/FeatureDetailCard';
import { ActivationExamples } from './components/ActivationExamples';
import { FeatureSearchResults, type SemanticFeatureResult } from './components/FeatureSearchResults';
import { FeatureStatistics } from './components/FeatureStatistics';
import { SimilarFeatures } from './components/SimilarFeatures';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Separator } from '@/lib/ui-primitives/separator';
import { SAE_TO_COLLECTION, getSemanticCollectionName } from '@/lib/utils/saeCollections';
import { ChatPanel, steeringFeatureKey } from './components/ChatInterface';

function ModeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const isDark = (resolvedTheme ?? 'light') === 'dark';

  return (
    <Button
      variant="circular"
      size="icon"
      className="relative ml-auto"
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      suppressHydrationWarning
    >
      <Sun className="h-[1.2rem] w-[1.2rem] scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
      <Moon className="absolute h-[1.2rem] w-[1.2rem] scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
    </Button>
  );
}

export default function FeaturesPage() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // URL state
  const modelIdParam = searchParams.get('modelId');
  const saeIdParam = searchParams.get('saeId');
  const featureParam = searchParams.get('featureIndex');

  // Local state
  const [selectedModelSae, setSelectedModelSae] = useState<string | null>(
    modelIdParam && saeIdParam ? `${modelIdParam}::${saeIdParam}` : null,
  );
  const [featureIndex, setFeatureIndex] = useState<number | null>(
    featureParam != null ? parseInt(featureParam, 10) : null,
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'text' | 'semantic'>('text');
  const [hoveredActivationValue, setHoveredActivationValue] = useState<number | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [steeringConfig, setSteeringConfig] = useState<SteeringConfig>({ features: [] });
  const [chatWidth, setChatWidth] = useState(448); // 28rem
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  const openChat = useCallback(() => setChatOpen(true), []);
  const closeChat = useCallback(() => setChatOpen(false), []);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    isDraggingRef.current = true;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const newWidth = Math.max(320, Math.min(window.innerWidth * 0.5, window.innerWidth - ev.clientX));
      setChatWidth(newWidth);
    };
    const onMouseUp = () => {
      isDraggingRef.current = false;
      setIsDragging(false);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  // Parse selected model/sae
  const [modelId, saeId] = selectedModelSae?.split('::') ?? [null, null];

  // Semantic collection for this model/sae
  const semanticCollectionName = useMemo(
    () => selectedModelSae ? getSemanticCollectionName(selectedModelSae) : null,
    [selectedModelSae],
  );

  // ---------- Queries ----------

  const { data: modelsData, loading: modelsLoading } = useQuery<{ saeModels: SaeModelInfo[] }>(
    GET_SAE_MODELS,
  );
  const models = useMemo(() => modelsData?.saeModels ?? [], [modelsData]);

  // Auto-select first model if none chosen
  useEffect(() => {
    if (!selectedModelSae && models.length > 0) {
      const key = `${models[0].modelId}::${models[0].saeId}`;
      setSelectedModelSae(key);
    }
  }, [models, selectedModelSae]);

  const [fetchFeature, { data: featureData, loading: featureLoading }] = useLazyQuery<{
    saeFeature: SaeFeature | null;
  }>(GET_SAE_FEATURE);

  const [fetchActivations, { data: activationsData, loading: activationsLoading }] = useLazyQuery<{
    saeActivations: SaeActivation[];
  }>(GET_SAE_ACTIVATIONS);

  const [fetchSearch, { data: searchData, loading: searchLoading }] = useLazyQuery<{
    saeFeatureSearch: SaeFeatureSearchResult[];
  }>(SEARCH_SAE_FEATURES);

  // Semantic search
  const [fetchSemanticSearch, { data: semanticSearchData, loading: semanticSearchLoading }] = useLazyQuery<{
    semanticSearch: Array<{ id: string; document: string | null; metadata: Record<string, unknown>; similarity: number }>;
  }>(SEMANTIC_SEARCH);

  // Densities (for histogram, fetched once per model/sae)
  const [fetchDensities, { data: densitiesData, loading: densitiesLoading }] = useLazyQuery<{
    saeFeatureDensities: number[];
  }>(GET_SAE_FEATURE_DENSITIES, { fetchPolicy: 'cache-first' });

  // Quantile activations (fetched on demand)
  const [fetchQuantiles, { data: quantilesData, loading: quantilesLoading }] = useLazyQuery<{
    saeActivationsByQuantile: SaeActivationQuantileGroup[];
  }>(GET_SAE_ACTIVATIONS_BY_QUANTILE, { fetchPolicy: 'cache-first' });

  const feature = featureData?.saeFeature ?? null;
  const activations = activationsData?.saeActivations ?? [];
  const searchResults = searchData?.saeFeatureSearch ?? [];
  const allDensities = densitiesData?.saeFeatureDensities ?? [];
  const quantileGroups = quantilesData?.saeActivationsByQuantile;

  // Semantic search results mapped to display format
  const semanticSearchResults: SemanticFeatureResult[] = useMemo(() => {
    if (!semanticSearchData?.semanticSearch) return [];
    return semanticSearchData.semanticSearch.map((r) => ({
      featureIndex: Number(r.metadata?.index ?? 0),
      label: r.document ?? null,
      density: (r.metadata?.density as number) ?? null,
      similarity: r.similarity,
    }));
  }, [semanticSearchData]);

  // ---------- Effects ----------

  // Fetch feature + activations when index or model changes
  useEffect(() => {
    if (modelId && saeId && featureIndex != null) {
      fetchFeature({ variables: { modelId, saeId, featureIndex } });
      fetchActivations({ variables: { modelId, saeId, featureIndex, limit: 20 } });
    }
  }, [modelId, saeId, featureIndex, fetchFeature, fetchActivations]);

  // Fetch densities once when model/sae changes
  useEffect(() => {
    if (modelId && saeId) {
      fetchDensities({ variables: { modelId, saeId } });
    }
  }, [modelId, saeId, fetchDensities]);

  // Sync URL
  useEffect(() => {
    const params = new URLSearchParams();
    if (modelId) params.set('modelId', modelId);
    if (saeId) params.set('saeId', saeId);
    if (featureIndex != null) params.set('featureIndex', featureIndex.toString());
    const newSearch = `?${params.toString()}`;
    if (newSearch !== window.location.search) {
      router.replace(newSearch, { scroll: false });
    }
  }, [modelId, saeId, featureIndex, router]);

  // ---------- Handlers ----------

  const handleModelSaeChange = useCallback((value: string) => {
    setSelectedModelSae(value);
    setFeatureIndex(0);
  }, []);

  const handleFeatureIndexChange = useCallback((index: number) => {
    setFeatureIndex(index);
  }, []);

  const handleSearch = useCallback(() => {
    const q = searchQuery.trim();
    if (!q) return;
    if (searchMode === 'semantic' && semanticCollectionName) {
      fetchSemanticSearch({
        variables: { collectionName: semanticCollectionName, query: q, nResults: 50 },
      });
    } else if (modelId && saeId) {
      fetchSearch({
        variables: { modelId, saeId, query: q, limit: 50 },
      });
    }
  }, [modelId, saeId, searchQuery, searchMode, semanticCollectionName, fetchSearch, fetchSemanticSearch]);

  const handleSearchSelect = useCallback((index: number) => {
    setFeatureIndex(index);
  }, []);

  const handleRequestQuantiles = useCallback(() => {
    if (modelId && saeId && featureIndex != null) {
      fetchQuantiles({
        variables: { modelId, saeId, featureIndex, nQuantiles: 5, perQuantileLimit: 5 },
      });
    }
  }, [modelId, saeId, featureIndex, fetchQuantiles]);

  // ---------- Steering handlers ----------

  const handleAddFeature = useCallback((f: SteeringFeature) => {
    const key = steeringFeatureKey(f);
    setSteeringConfig((prev) => ({
      features: [...prev.features.filter((x) => steeringFeatureKey(x) !== key), f],
    }));
  }, []);

  const handleRemoveFeature = useCallback((key: string) => {
    setSteeringConfig((prev) => ({
      features: prev.features.filter((x) => steeringFeatureKey(x) !== key),
    }));
  }, []);

  const handleUpdateStrength = useCallback((key: string, strength: number) => {
    setSteeringConfig((prev) => ({
      features: prev.features.map((f) =>
        steeringFeatureKey(f) === key ? { ...f, strength } : f,
      ),
    }));
  }, []);

  // Find max feature count for navigation bounds
  const currentModel = models.find(
    (m: SaeModelInfo) => m.modelId === modelId && m.saeId === saeId,
  );
  const maxFeatureIndex = currentModel?.featureCount;

  // Collection link for cross-navigation
  const collectionLink = selectedModelSae ? SAE_TO_COLLECTION[selectedModelSae] ?? null : null;

  // Active search results depend on mode
  const isSemanticSearch = searchMode === 'semantic';
  const activeSearchLoading = isSemanticSearch ? semanticSearchLoading : searchLoading;
  const hasActiveResults = isSemanticSearch ? semanticSearchResults.length > 0 : searchResults.length > 0;
  const activeResultCount = isSemanticSearch ? semanticSearchResults.length : searchResults.length;

  // ---------- Render ----------

  return (
    <div
      className="flex h-screen bg-background"
      style={{ '--chat-width': `${chatWidth}px` } as React.CSSProperties}
    >
      {/* Main content */}
      <div className="flex h-full flex-1 min-w-0 flex-col">
        {/* Top nav */}
        <header className="border-b px-4 py-3 flex items-center gap-3 shrink-0">
          <Link
            href="/"
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Visualization
          </Link>
          <Separator orientation="vertical" className="h-5" />
          <h1 className="font-semibold text-sm">SAE Feature Explorer</h1>
          <ModeToggle />
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="max-w-7xl mx-auto px-4 py-4 space-y-4">
            {modelsLoading ? (
              <div className="flex items-center gap-2 py-8 justify-center">
                <Spinner className="h-5 w-5" />
                <span className="text-sm text-muted-foreground">Loading SAE models...</span>
              </div>
            ) : models.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No SAE data found. Ingest features first.</p>
              </div>
            ) : (
              <>
                <FeatureHeader
                  models={models}
                  selectedModelSae={selectedModelSae}
                  onModelSaeChange={handleModelSaeChange}
                  featureIndex={featureIndex}
                  onFeatureIndexChange={handleFeatureIndexChange}
                  searchQuery={searchQuery}
                  onSearchQueryChange={setSearchQuery}
                  onSearch={handleSearch}
                  maxFeatureIndex={maxFeatureIndex}
                  collectionLink={collectionLink}
                  searchMode={searchMode}
                  onSearchModeChange={setSearchMode}
                  hasSemanticSearch={!!semanticCollectionName}
                />

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {/* Left: Search results */}
                  <div className="lg:col-span-1 space-y-2">
                    <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {hasActiveResults
                        ? `${isSemanticSearch ? 'Semantic' : 'Search'} Results (${activeResultCount})`
                        : 'Search Features'}
                    </h3>
                    {activeSearchLoading ? (
                      <div className="flex justify-center py-4">
                        <Spinner className="h-4 w-4" />
                      </div>
                    ) : hasActiveResults ? (
                      <FeatureSearchResults
                        results={searchResults}
                        onSelect={handleSearchSelect}
                        selectedIndex={featureIndex}
                        mode={searchMode}
                        semanticResults={isSemanticSearch ? semanticSearchResults : undefined}
                      />
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Search by label or browse with the arrow buttons.
                      </p>
                    )}
                  </div>

                  {/* Right: Feature detail + statistics + similar + activations */}
                  <div className="lg:col-span-2 space-y-4">
                    {featureLoading ? (
                      <div className="flex justify-center py-8">
                        <Spinner className="h-5 w-5" />
                      </div>
                    ) : feature ? (
                      <>
                        <div className="border rounded-lg p-4 bg-card">
                          <FeatureDetailCard feature={feature} />
                        </div>

                        <FeatureStatistics
                          feature={feature}
                          activations={activations}
                          allDensities={allDensities}
                          densitiesLoading={densitiesLoading}
                          hoveredActivationValue={hoveredActivationValue}
                        />

                        {semanticCollectionName && (
                          <SimilarFeatures
                            collectionName={semanticCollectionName}
                            featureIndex={feature.featureIndex}
                            featureLabel={feature.label}
                            onSelectFeature={handleSearchSelect}
                            selectedIndex={featureIndex}
                          />
                        )}

                        <div>
                          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                            Activations
                            {activations.length > 0 && (
                              <span className="ml-1">({activations.length})</span>
                            )}
                          </h3>
                          {activationsLoading ? (
                            <div className="flex justify-center py-4">
                              <Spinner className="h-4 w-4" />
                            </div>
                          ) : (
                            <ActivationExamples
                              activations={activations}
                              quantileGroups={quantileGroups}
                              quantileLoading={quantilesLoading}
                              onRequestQuantiles={handleRequestQuantiles}
                              onHoverActivation={setHoveredActivationValue}
                            />
                          )}
                        </div>
                      </>
                    ) : featureIndex != null ? (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        Feature #{featureIndex} not found.
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        Select a feature to view details.
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </main>
      </div>

      {/* Chat sidebar */}
      <div
        className="group/chat"
        data-state={chatOpen ? 'open' : 'closed'}
      >
        {/* Spacer — in document flow, transitions width to push main content */}
        <div className={`h-full w-(--chat-width) shrink-0 bg-transparent group-data-[state=closed]/chat:w-0 ${isDragging ? '' : 'transition-[width] duration-300 ease-[var(--ease-spring)]'}`} />

        {/* Container — fixed, slides in from right */}
        <div
          className={`fixed inset-y-0 right-0 z-10 w-(--chat-width) group-data-[state=closed]/chat:right-[calc(var(--chat-width)*-1)] ${isDragging ? '' : 'transition-[right] duration-300 ease-[var(--ease-spring)]'}`}
          aria-hidden={!chatOpen}
        >
          {/* Resize handle */}
          <div
            onMouseDown={handleResizeStart}
            className={`absolute inset-y-0 -left-1 z-20 w-2 cursor-col-resize
              before:absolute before:inset-y-0 before:left-1/2 before:w-px before:-translate-x-1/2
              before:transition-colors before:duration-150
              before:bg-transparent hover:before:bg-border active:before:bg-primary
              ${isDragging ? 'before:!bg-primary' : ''}`}
          />
          <div className="flex h-full flex-col border-l bg-background">
            <ChatPanel
              steeringConfig={steeringConfig}
              modelId={modelId}
              saeId={saeId}
              currentFeature={feature}
              onAddFeature={handleAddFeature}
              onRemoveFeature={handleRemoveFeature}
              onUpdateStrength={handleUpdateStrength}
              onClose={closeChat}
            />
          </div>
        </div>
      </div>

      {/* Floating chat button */}
      {!chatOpen && (
        <Button
          variant="circular"
          size="icon-lg"
          onClick={openChat}
          className="!fixed bottom-6 right-6 z-40 shadow-[var(--shadow-float)]"
        >
          <Sparkles className="size-4" />
          <span className="sr-only">Open steered chat</span>
        </Button>
      )}
    </div>
  );
}
