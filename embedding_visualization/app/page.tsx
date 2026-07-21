'use client';

import { Suspense, useEffect, useCallback, useMemo, useRef, useState } from 'react';
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
import { useProbes } from '../lib/hooks/useProbes';
import { useTextSearch, TEXT_SEARCH_GLOW_CAP } from '../lib/hooks/useTextSearch';
import { usePromptHighlight, buildPromptHighlightResults } from '../lib/hooks/usePromptHighlight';
import { useDocumentFeatureSearch } from '../lib/hooks/useDocumentFeatureSearch';
import { isInTemporalRange } from '../lib/utils/temporalFilters';
import { useVisualizationStore } from '../lib/stores/useVisualizationStore';
import type { HighlightMap, ColorScale, ColorScaleType } from '../lib/types/types';
import { getSaeInfo, getSaeInfoFromMetadata, isSaeFeatureCollection } from '../lib/utils/saeCollections';
import { serializeColorScale, deserializeColorScale, resolveDefaultColorScheme } from '../lib/utils/colorScaleUrl';
import { mergeViewSearch, shouldDropPreset, type OwnedViewParams } from '../lib/utils/urlViewParams';
import { useHfSpaceUrlSync } from '../lib/hooks/useHfSpaceUrlSync';
import {
  getPreset,
  seedInitialColorState,
  resolveInitialCollection,
  presetStoreOps,
  TOUR_PRESET_ID,
  type PresetDefinition,
} from '../lib/utils/tourPresets';
import {
  getOnboardingAction,
  readIntroSeen,
  markIntro,
  type OnboardingAction,
} from '../lib/utils/demoOnboarding';
import { IS_DEMO } from '../lib/utils/demoMode';
import { DemoIntro } from './components/DemoIntro';
import type { TourRuntime } from '../lib/utils/tourSteps';
import dynamic from 'next/dynamic';

// Loaded on demand so regular visits never pay for the tour library.
const TourController = dynamic(
  () => import('./components/TourController'),
  { ssr: false },
);

const EMPTY_METADATA: Record<string, unknown>[] = [];

/** Execute a preset's store mutations (method/mode/flags) imperatively. */
function runPresetStoreOps(preset: PresetDefinition) {
  const state = useVisualizationStore.getState();
  for (const op of presetStoreOps(preset)) {
    if (op.kind === 'method') state.setMethod(op.value);
    else if (op.kind === 'mode') state.setMode(op.value);
    else state.setFlag(op.flag, op.value);
  }
}

/**
 * Apply a colour scheme to the store. A recommended scale is only resolved
 * when no explicit scale was provided (e.g. a shared link or preset carrying
 * only a field); otherwise the scale is applied as-is.
 */
function applyColorScheme(
  field: string,
  scale: ColorScale | null,
  palette: string | null,
  colorFieldOptions: ReadonlyArray<{ field: string; recommendedScale?: ColorScaleType }>,
) {
  const state = useVisualizationStore.getState();
  const recommended = scale
    ? undefined
    : colorFieldOptions.find(f => f.field === field)?.recommendedScale;
  state.setColorByField(field, recommended);
  if (scale) state.setColorScale(scale);
  if (palette) state.setCategoricalPalette(palette);
}

export default function Home() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-background text-muted-foreground">
        Loading...
      </div>
    }>
      <HomeContent />
    </Suspense>
  );
}

function HomeContent() {
  const { collections, loading: collectionsLoading, error: collectionsError } = useCollections();
  const searchParams = useSearchParams();
  const router = useRouter();
  // Mirror the app URL to the HF Space parent page (no-op outside an iframe).
  useHfSpaceUrlSync();
  const collectionFromUrl = searchParams.get('collection');
  const colorByFromUrl = searchParams.get('colorBy');
  const isInitialLoad = useRef(true);
  // Demo preset (`?preset=`) latched once at first render; its colour block
  // seeds the same initial refs as explicit URL colour params (URL wins).
  const initialPresetRef = useRef(getPreset(searchParams.get('preset')));
  // Colour scheme captured from the URL/preset on first render (applied once data loads)
  const initialColorSeed = useRef(seedInitialColorState({
    urlColorBy: colorByFromUrl,
    urlScale: deserializeColorScale({
      scale: searchParams.get('scale'),
      scaleName: searchParams.get('scaleName'),
      color: searchParams.get('color'),
    }),
    urlPalette: searchParams.get('palette'),
    preset: initialPresetRef.current,
  })).current;
  const initialColorByRef = useRef(initialColorSeed.colorBy);
  const initialColorScaleRef = useRef(initialColorSeed.scale);
  const initialPaletteRef = useRef(initialColorSeed.palette);
  // One-shot onboarding decision (`?tour=1` / `?intro=1` / demo first visit),
  // latched before the URL-sync effect strips the one-shot params.
  const [onboarding] = useState<OnboardingAction>(() =>
    typeof window === 'undefined'
      ? null
      : getOnboardingAction({
          isDemo: IS_DEMO,
          search: window.location.search,
          introSeen: readIntroSeen(),
          viewportWidth: window.innerWidth,
        }),
  );
  const [introOpen, setIntroOpen] = useState(onboarding === 'intro');
  const [tourRequested, setTourRequested] = useState(onboarding === 'tour');
  // A preset applied mid-session (welcome-dialog buttons); its colour block is
  // applied by a dedicated effect once the collection's data has loaded.
  const [pendingPreset, setPendingPreset] = useState<PresetDefinition | null>(null);

  // Apply the initial preset's store flags once, after persisted preferences
  // rehydrate (preset > persisted prefs; flags have no URL representation).
  useEffect(() => {
    const preset = initialPresetRef.current;
    if (!preset) return;
    const apply = () => runPresetStoreOps(preset);
    if (useVisualizationStore.persist.hasHydrated()) {
      apply();
      return;
    }
    return useVisualizationStore.persist.onFinishHydration(apply);
  }, []);

  // Default to the first available collection
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);

  // Select the initial collection: URL > preset > demo default > first key
  useEffect(() => {
    if (collections && !selectedCollection) {
      const initial = resolveInitialCollection({
        urlCollection: collectionFromUrl,
        presetCollection: initialPresetRef.current?.collection ?? null,
        isDemo: IS_DEMO,
        manifestKeys: Object.keys(collections),
      });
      if (initial) {
        setSelectedCollection(initial);
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

  const { data, loading, error, colorFieldOptions, defaultTooltipFields, loadedCollection } = useEmbeddingData(
    selectedCollection,
    method,
    mode,
  );

  // Embedding-space probes: score/residual fields are merged client-side
  // (never into the persisted field_analysis cache).
  const probes = useProbes(selectedCollection, data);
  const mergedColorFieldOptions = useMemo(
    () => [...colorFieldOptions, ...probes.fieldOptions],
    [colorFieldOptions, probes.fieldOptions],
  );

  // Sync URL when collection or colorBy changes. Owned params are merged into
  // the current search string so unknown params (and `preset` while it still
  // applies) survive; one-shot params (`tour`/`intro`) are always stripped.
  useEffect(() => {
    if (!selectedCollection) return;
    const owned: OwnedViewParams = { collection: selectedCollection };
    // During initial load, preserve colorBy from the original URL until state catches up
    const effectiveColorBy = colorByField
      ?? (isInitialLoad.current ? initialColorByRef.current : null);
    if (effectiveColorBy) {
      owned.colorBy = effectiveColorBy;
      // Encode the color scheme alongside the field. During initial load the
      // store hasn't applied the URL state yet, so prefer the original URL refs.
      const effectiveScale = isInitialLoad.current
        ? (initialColorScaleRef.current ?? colorScale)
        : colorScale;
      const effectivePalette = isInitialLoad.current
        ? (initialPaletteRef.current ?? categoricalPalette)
        : categoricalPalette;
      const colorParams = serializeColorScale(effectiveScale, effectivePalette ?? undefined);
      owned.scale = colorParams.scale ?? null;
      owned.scaleName = colorParams.scaleName ?? null;
      owned.color = colorParams.color ?? null;
      owned.palette = colorParams.palette ?? null;
    } else {
      owned.colorBy = null;
      owned.scale = null;
      owned.scaleName = null;
      owned.color = null;
      owned.palette = null;
    }
    // `?preset=` stays while the user is on the preset's collection, so the
    // link keeps reproducing its non-URL flags (nebula, cluster labels, …).
    if (shouldDropPreset(initialPresetRef.current?.collection, selectedCollection)) {
      owned.preset = null;
    }
    const newSearch = mergeViewSearch(window.location.search, owned);
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

  const { points2d, points3d } = useVisualizationPoints(
    probes.augmentedData ?? data,
    { method, mode, searchQuery },
  );

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
  // Prompt activation only applies when the points themselves are SAE features;
  // document collections linked to an SAE keep saeInfo for the feature search.
  const saeFeatureInfo = isSaeFeatureCollection(data?.availableFields) ? saeInfo : null;
  // Default 0.01 matches the SAE page; clearing the input disables the filter.
  const [promptMaxDensity, setPromptMaxDensity] = useState<number | null>(0.01);
  const promptHighlight = usePromptHighlight(saeFeatureInfo, data?.itemMetadata ?? EMPTY_METADATA, promptMaxDensity);

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
      // Returned so imperative callers (the demo tour) can await completion.
      return handleSemanticSearch(query);
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
  // Broad text searches (> TEXT_SEARCH_GLOW_CAP matches) skip the glow and act as
  // a pure filter: muting still uses the full match set via textSearchHighlights.
  // Selected point is excluded — it has its own overlay traces in ScatterPlot3D.
  const textSearchGlowIndices =
    textSearchHighlightedIndices && textSearchHighlightedIndices.size <= TEXT_SEARCH_GLOW_CAP
      ? textSearchHighlightedIndices
      : null;
  const baseHighlightedIndices: HighlightMap | undefined = useHighlightedIndices(
    semanticSearchResults,
    data,
    semanticSearchResults && semanticSearchResults.length > 0 ? null : textSearchGlowIndices,
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
    // A preset queued for a different collection no longer applies — without
    // this, revisiting its collection much later would replay its colours.
    setPendingPreset(prev => (prev && prev.collection !== selectedCollection ? null : prev));
  }, [selectedCollection, resetSearch, promptHighlight.clear, featureSearch.clearFeatures]);

  // Apply the colour scheme for the active collection once its data is loaded.
  // Precedence: URL params (first load only) > collection's saved default > none.
  // Re-applies only when `selectedCollection` actually changes to a new value, so
  // manual edits persist for the lifetime of a selection. The ref keeps the effect
  // inert across `collections` refetches (e.g. after saving a default).
  const defaultsAppliedFor = useRef<string | null>(null);
  useEffect(() => {
    // Act only once the loaded fields belong to the selected collection AND have
    // populated. `loadedCollection` guards against the previous collection's stale
    // options; the length check covers the uncached field-analysis path, where
    // colorFieldOptions arrive a tick after loadedCollection (via setTimeout in
    // useEmbeddingData). The guard sits before the ref write so a blocked run never
    // marks the collection as done.
    if (!selectedCollection || loadedCollection !== selectedCollection || colorFieldOptions.length === 0) return;
    if (defaultsAppliedFor.current === selectedCollection) return;
    defaultsAppliedFor.current = selectedCollection;

    const wasInitialLoad = isInitialLoad.current;
    isInitialLoad.current = false;

    // Resolve which scheme to apply: URL (first load only) wins, else the default.
    let field: string | null = null;
    let scale: ColorScale | null = null;
    let palette: string | null = null;

    if (wasInitialLoad && initialColorByRef.current) {
      field = initialColorByRef.current;
      scale = initialColorScaleRef.current;
      palette = initialPaletteRef.current;
    } else {
      const resolved = resolveDefaultColorScheme(collections?.[selectedCollection]?.defaultColorScheme);
      if (resolved) {
        field = resolved.field;
        scale = resolved.scale;
        palette = resolved.palette;
      }
    }

    if (!field) return;
    applyColorScheme(field, scale, palette, colorFieldOptions);
  }, [selectedCollection, loadedCollection, collections, colorFieldOptions]);

  // Apply a mid-session preset's colour block once its collection has loaded.
  // Declared after the default-scheme effect so it runs later in the same
  // commit and wins over the collection default (preset > collection default).
  useEffect(() => {
    if (!pendingPreset || loadedCollection !== pendingPreset.collection || colorFieldOptions.length === 0) return;
    const { color } = pendingPreset;
    setPendingPreset(null);
    if (!color) return;
    applyColorScheme(color.colorBy, color.scale ?? null, color.palette ?? null, colorFieldOptions);
  }, [pendingPreset, loadedCollection, colorFieldOptions]);

  // Welcome-dialog actions: apply a curated preset / start the guided tour.
  // Deliberately does NOT write `?preset=` to the URL — the param describes a
  // link-entry state; interactive use keeps the URL on the regular
  // collection/colour params only.
  const applyPresetLive = useCallback((presetId: string) => {
    const preset = getPreset(presetId);
    if (!preset || !collections?.[preset.collection]) return;
    setIntroOpen(false);
    setSelectedCollection(preset.collection);
    runPresetStoreOps(preset);
    setPendingPreset(preset);
  }, [collections]);

  // Collections present in the manifest — gates the dialog's preset buttons.
  const availableCollections = useMemo(
    () => (collections ? new Set(Object.keys(collections)) : null),
    [collections],
  );

  const startTour = useCallback(() => {
    setIntroOpen(false);
    // Land on the tour collection immediately so step 1's map is already the
    // one the tour narrates (step 3 re-applies as a safety net).
    applyPresetLive(TOUR_PRESET_ID);
    setTourRequested(true);
  }, [applyPresetLive]);

  // Imperative surface for the tour's prepare hooks. The runtime MUST be
  // identity-stable AND always-fresh: react-joyride deep-compares steps with
  // function source-text equality, so rebuilt `before` closures are treated
  // as unchanged and the runtime captured at the tour's first render would be
  // used for the whole tour. Every capture-prone piece therefore delegates
  // through a ref reassigned each render (same pattern as screenshotHandlerRef).
  const loadedCollectionRef = useRef(loadedCollection);
  loadedCollectionRef.current = loadedCollection;
  const applyPresetLiveRef = useRef(applyPresetLive);
  applyPresetLiveRef.current = applyPresetLive;
  const runSearchRef = useRef(wrappedHandleSemanticSearch);
  runSearchRef.current = wrappedHandleSemanticSearch;
  const resetSearchRef = useRef(resetSearch);
  resetSearchRef.current = resetSearch;
  const topicSearchRef = useRef(topicSearch);
  topicSearchRef.current = topicSearch;
  const collectionTopicsRef = useRef(selectedCollectionTopics);
  collectionTopicsRef.current = selectedCollectionTopics;
  const tourRuntime = useMemo<TourRuntime>(() => ({
    applyPreset: (presetId: string) => applyPresetLiveRef.current(presetId),
    runSearch: async (query: string) => {
      await runSearchRef.current(query);
    },
    clearSearch: () => resetSearchRef.current(),
    // Isolation = exactly one selected topic; DashboardPanel derives the
    // muting from `selectedTopicIds` when colouring by topic_label.
    isolateFirstTopic: () => {
      const topic = (collectionTopicsRef.current ?? []).find(
        (t) => t.topicId >= 0 && t.label,
      );
      if (!topic) return null;
      topicSearchRef.current.clearAll();
      topicSearchRef.current.toggleTopic(topic.topicId);
      return topic.label;
    },
    clearTopicSelection: () => topicSearchRef.current.clearAll(),
    setActivePanel,
    setShowLabels: (value: boolean) =>
      useVisualizationStore.getState().setFlag('showLabels', value),
    getLoadedCollection: () => loadedCollectionRef.current ?? null,
    getColorByField: () => useVisualizationStore.getState().colorByField,
  }), []);

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
              onOpenIntro={IS_DEMO ? () => setIntroOpen(true) : undefined}
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
                  onPromptHighlightSubmit={saeFeatureInfo ? promptHighlight.submit : undefined}
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
                  colorFieldOptions={mergedColorFieldOptions}
                  collectionName={selectedCollection}
                  probes={probes}
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
        <DemoIntro
          open={introOpen}
          onOpenChange={setIntroOpen}
          onStartTour={startTour}
          onApplyPreset={applyPresetLive}
          availableCollections={availableCollections}
        />
        {tourRequested && (
          <TourController runtime={tourRuntime} onDone={() => setTourRequested(false)} />
        )}
      </SidebarInset>
    </SidebarProvider>
  );
}
