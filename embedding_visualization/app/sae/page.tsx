'use client';

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery, useLazyQuery, useApolloClient } from '@apollo/client/react';
import { Sparkles, Sun, Moon, X } from 'lucide-react';
import { useTheme } from 'next-themes';
import { toast } from 'sonner';
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
} from '@/lib/types/types';
import { FeatureHeader } from './components/FeatureHeader';
import { FeatureDetailCard } from './components/FeatureDetailCard';
import { ActivationExamples } from './components/ActivationExamples';
import {
  FeatureSearchResults,
  type SemanticFeatureResult,
  type SelectedFeatureRef,
} from './components/FeatureSearchResults';
import { FeatureStatistics } from './components/FeatureStatistics';
import { SimilarFeatures } from './components/SimilarFeatures';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { PageNav } from '@/app/components/PageNav';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { Slider } from '@/lib/ui-primitives/slider';
import { RUN_PROMPT_ACTIVATIONS } from '@/lib/graphql/mutations';
import type { PromptActivationsResult } from '@/lib/graphql/mutations';
import { SAE_TO_COLLECTION, getSemanticCollectionName, getSemanticCollections, parseSaeId } from '@/lib/utils/saeCollections';
import { ensureModelLoaded } from '@/lib/utils/modelLoader';
import { ChatPanel } from './components/ChatInterface';
import { SteeringIdenticon } from './components/ChatInterface/SteeringIdenticon';
import { useModelIdentityStore } from '@/lib/stores/useModelIdentityStore';
import { PromptTokenActivations, type SelectedTokenInfo } from './components/PromptTokenActivations';
import { useChatSessions } from '@/lib/hooks/useChatSessions';
import { useSaeSelection } from './hooks/useSaeSelection';
import { attachSaeIdentity, poolPromptFeatures, MAX_POOLED_ROWS } from './utils/promptPooling';
import { serializeSaesParam } from './utils/saeSelection';
import type { ChatMessage } from '@/lib/types/types';

/** Shape of a single collection's fan-out semantic search result. */
interface FanoutResult {
  modelId: string;
  saeId: string;
  results: Array<{
    document: string | null;
    metadata: Record<string, unknown>;
    similarity: number;
  }>;
}

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
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-background text-muted-foreground">
        Loading...
      </div>
    }>
      <FeaturesPageContent />
    </Suspense>
  );
}

function FeaturesPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const apolloClient = useApolloClient();

  // URL params — multi (?model=&saes=), legacy single (?modelId=&saeId=),
  // and the old dimension format (?model=&layer=&hookType=&width=).
  // Snapshot on mount only; the URL-sync effect owns the URL afterwards.
  const urlParams = useMemo(() => ({
    saes: searchParams.get('saes'),
    model: searchParams.get('model'),
    modelId: searchParams.get('modelId'),
    saeId: searchParams.get('saeId'),
    layer: searchParams.get('layer'),
    hookType: searchParams.get('hookType'),
    width: searchParams.get('width'),
    featureIndex: searchParams.get('featureIndex'),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), []);

  // The feature open in the detail pane — independent of the SAE selection,
  // so clicking search results never rewrites the selection or the results.
  const [selectedFeature, setSelectedFeature] = useState<SelectedFeatureRef | null>(() =>
    urlParams.modelId && urlParams.saeId && urlParams.featureIndex != null
      ? {
          modelId: urlParams.modelId,
          saeId: urlParams.saeId,
          featureIndex: parseInt(urlParams.featureIndex, 10),
        }
      : null,
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'text' | 'semantic' | 'prompt'>('text');

  // Prompt search state — single call to runPromptActivations gives both ranked list and token strip
  const [skipChatTemplate, setSkipChatTemplate] = useState(false);
  const [promptActivations, setPromptActivations] = useState<PromptActivationsResult | null>(null);
  const [promptSearchLoading, setPromptSearchLoading] = useState(false);
  const [promptSearchError, setPromptSearchError] = useState<string | null>(null);
  const [promptPooling, setPromptPooling] = useState<'max' | 'mean' | 'last'>('max');
  const [promptMaxDensity, setPromptMaxDensity] = useState<number>(0.01);
  const [activationFilterMode, setActivationFilterMode] = useState<'NONE' | 'NEURONPEDIA' | 'COVERAGE_BOS' | 'COVERAGE_NO_BOS'>('COVERAGE_BOS');
  const [selectedTokenInfo, setSelectedTokenInfo] = useState<SelectedTokenInfo | null>(null);
  const [hoveredActivationValue, setHoveredActivationValue] = useState<number | null>(null);
  const [chatOpen, setChatOpen] = useState(false);

  const floatingButtonFeatures = useModelIdentityStore((s) => s.steeringConfig.features);
  const [chatWidth, setChatWidth] = useState(448); // 28rem
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  // Fan-out semantic search state
  const [mergedSemanticResults, setMergedSemanticResults] = useState<SemanticFeatureResult[]>([]);
  const [semanticFanoutLoading, setSemanticFanoutLoading] = useState(false);

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

  // ---------- Chat sessions ----------

  const {
    sessions: chatSessions,
    loading: chatSessionsLoading,
    activeSessionId,
    createSession,
    loadSession,
    saveMessage,
    deleteSession,
    setActiveSessionId,
  } = useChatSessions();

  const [loadedMessages, setLoadedMessages] = useState<ChatMessage[] | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);
  activeSessionIdRef.current = activeSessionId;

  const handleUserMessageSent = useCallback(
    async (message: ChatMessage) => {
      let sessionId = activeSessionIdRef.current;
      const snapshot = useModelIdentityStore.getState().steeringConfig;
      if (!sessionId) {
        sessionId = await createSession(snapshot, message.content);
      }
      saveMessage(sessionId, message, snapshot);
    },
    [createSession, saveMessage]
  );

  const handleAssistantMessageComplete = useCallback(
    (message: ChatMessage) => {
      const sessionId = activeSessionIdRef.current;
      if (sessionId) {
        const snapshot = useModelIdentityStore.getState().steeringConfig;
        saveMessage(sessionId, message, snapshot);
      }
    },
    [saveMessage]
  );

  const handleSelectSession = useCallback(
    async (id: string) => {
      try {
        const { messages, config } = await loadSession(id);
        setLoadedMessages(messages);
        useModelIdentityStore.getState().setSteeringConfig(config);
      } catch {
        toast.error('Failed to load session');
      }
    },
    [loadSession]
  );

  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    // Empty array (NOT null) is the new-chat/clear signal. ChatPanel's
    // loadedMessages effect distinguishes it from a real session load by length
    // (length 0 → clear only; length > 0 → load + drop compare mode). Compare
    // mode's toggle relies on this — don't change [] to a sentinel/null here.
    setLoadedMessages([]);
  }, [setActiveSessionId]);

  // ---------- Queries ----------

  const { data: modelsData, loading: modelsLoading } = useQuery<{ saeModels: SaeModelInfo[] }>(
    GET_SAE_MODELS,
  );
  const models = useMemo(() => modelsData?.saeModels ?? [], [modelsData]);

  // Selection: one model + a multi-select over its SAEs (defaults to all)
  const {
    modelId,
    saeIds,
    setModel,
    setSaeIds,
    selectSingle,
    modelOptions,
    saeOptions,
    pairs,
    isSingleSae,
    singleSaeId,
    totalFeatureCount,
  } = useSaeSelection(models, urlParams);

  // Bridge selection into the Zustand store (single source of truth for chat).
  // The model is always known; saeId only when exactly one SAE is selected.
  useEffect(() => {
    useModelIdentityStore.getState().setIdentity(modelId, isSingleSae ? singleSaeId : null);
  }, [modelId, isSingleSae, singleSaeId]);

  // Handle model selection from the chat input — narrows to that single SAE
  const handleSelectModel = useCallback((newModelId: string, newSaeId: string) => {
    selectSingle(newModelId, newSaeId);
  }, [selectSingle]);

  // Semantic collections available for the current selection (fan-out targets)
  const semanticCollections = useMemo(() => getSemanticCollections(pairs), [pairs]);
  const hasAnySemanticCollection = semanticCollections.length > 0;

  // Semantic collection for the feature open in the detail pane
  const detailSemanticCollection = useMemo(
    () => selectedFeature
      ? getSemanticCollectionName(`${selectedFeature.modelId}::${selectedFeature.saeId}`)
      : null,
    [selectedFeature],
  );

  const [fetchFeature, { data: featureData, loading: featureLoading }] = useLazyQuery<{
    saeFeature: SaeFeature | null;
  }>(GET_SAE_FEATURE);

  const [fetchActivations, { data: activationsData, loading: activationsLoading }] = useLazyQuery<{
    saeActivations: SaeActivation[];
  }>(GET_SAE_ACTIVATIONS);

  const [fetchSearch, { data: searchData, loading: searchLoading }] = useLazyQuery<{
    saeFeatureSearch: SaeFeatureSearchResult[];
  }>(SEARCH_SAE_FEATURES);

  // Densities (for histogram, fetched once per detail model/sae)
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

  // Semantic results come from the fan-out (also for a single SAE — one target)
  const semanticSearchResults = mergedSemanticResults;

  // Prompt layers tagged with their SAE identity (resolved from the selection)
  const attachedPromptLayers = useMemo(
    () => promptActivations && modelId
      ? attachSaeIdentity(promptActivations.layers, modelId, pairs)
      : [],
    [promptActivations, modelId, pairs],
  );

  // Ranked feature list pooled across all hooked SAEs
  const promptSearchAsSemanticResults: SemanticFeatureResult[] = useMemo(
    () => poolPromptFeatures(attachedPromptLayers, promptPooling, promptMaxDensity),
    [attachedPromptLayers, promptPooling, promptMaxDensity],
  );

  // Clear stale fan-out / prompt results when the selection changes
  useEffect(() => {
    setMergedSemanticResults([]);
    setPromptActivations(null);
    setPromptSearchError(null);
    setSelectedTokenInfo(null);
  }, [pairs]);

  // ---------- Effects ----------

  const detailModelId = selectedFeature?.modelId ?? null;
  const detailSaeId = selectedFeature?.saeId ?? null;
  const detailFeatureIndex = selectedFeature?.featureIndex ?? null;

  // Fetch feature + activations when the detail target changes
  useEffect(() => {
    if (detailModelId && detailSaeId && detailFeatureIndex != null) {
      fetchFeature({ variables: { modelId: detailModelId, saeId: detailSaeId, featureIndex: detailFeatureIndex } });
      fetchActivations({ variables: { modelId: detailModelId, saeId: detailSaeId, featureIndex: detailFeatureIndex, limit: 20 } });
    }
  }, [detailModelId, detailSaeId, detailFeatureIndex, fetchFeature, fetchActivations]);

  // Fetch densities once per detail model/sae (histogram context)
  useEffect(() => {
    if (detailModelId && detailSaeId) {
      fetchDensities({ variables: { modelId: detailModelId, saeId: detailSaeId } });
    }
  }, [detailModelId, detailSaeId, fetchDensities]);

  // Sync URL — legacy single-SAE params for cross-link compat, multi otherwise
  useEffect(() => {
    if (!modelId) return; // selection not initialized yet
    const params = new URLSearchParams();
    if (isSingleSae && singleSaeId) {
      params.set('modelId', modelId);
      params.set('saeId', singleSaeId);
      if (
        selectedFeature &&
        selectedFeature.modelId === modelId &&
        selectedFeature.saeId === singleSaeId
      ) {
        params.set('featureIndex', selectedFeature.featureIndex.toString());
      }
    } else {
      params.set('model', modelId);
      params.set('saes', serializeSaesParam(saeIds));
    }
    const newSearch = `?${params.toString()}`;
    if (newSearch !== window.location.search) {
      router.replace(newSearch, { scroll: false });
    }
  }, [modelId, saeIds, isSingleSae, singleSaeId, selectedFeature, router]);

  // ---------- Handlers ----------

  // Index input / prev-next browsing (single-SAE mode only)
  const handleFeatureIndexChange = useCallback((index: number) => {
    if (modelId && singleSaeId) {
      setSelectedFeature({ modelId, saeId: singleSaeId, featureIndex: index });
    }
  }, [modelId, singleSaeId]);

  const handleSearch = useCallback(async () => {
    const q = searchQuery.trim();
    if (!q) return;

    if (searchMode === 'prompt') {
      // Prompt activation search: ONE runPromptActivations call hooks every
      // selected SAE (same model) in a single forward pass.
      if (!modelId || pairs.length === 0) return;
      if (promptSearchLoading) return;
      setPromptSearchLoading(true);
      setPromptSearchError(null);
      setSelectedTokenInfo(null);
      try {
        const { checkpoint: storeCheckpt } = useModelIdentityStore.getState();
        const loadErr = await ensureModelLoaded(storeCheckpt ?? undefined);
        if (loadErr) {
          setPromptSearchError(loadErr);
          setPromptSearchLoading(false);
          return;
        }
        const saes = pairs.map((p) => {
          const parsed = parseSaeId(p.saeId);
          return { layer: parsed.layerIndex, width: parsed.width };
        });
        const { data } = await apolloClient.mutate<{ runPromptActivations: PromptActivationsResult }>({
          mutation: RUN_PROMPT_ACTIVATIONS,
          variables: {
            input: {
              prompt: q,
              saes,
              topK: 0,
              modelId,
              saeId: isSingleSae ? singleSaeId : null,
              skipChatTemplate,
              filterMode: activationFilterMode,
            },
          },
        });
        const result = data?.runPromptActivations;
        if (result?.error) {
          setPromptSearchError(result.error);
          setPromptSearchLoading(false);
          return;
        }
        setPromptActivations(result ?? null);
      } catch (err) {
        setPromptSearchError(err instanceof Error ? err.message : 'Prompt inference failed');
      } finally {
        setPromptSearchLoading(false);
      }
    } else if (searchMode === 'semantic') {
      // Fan-out semantic search across the selected SAEs' embedded collections
      if (semanticCollections.length === 0) return;

      setSemanticFanoutLoading(true);
      try {
        const promises = semanticCollections.map(({ modelId: mId, saeId: sId, collectionName }): Promise<FanoutResult> =>
          apolloClient.query<{ semanticSearch: FanoutResult['results'] }>({
            query: SEMANTIC_SEARCH,
            variables: { collectionName, query: q, nResults: 50 },
          }).then(({ data }) => ({
            modelId: mId,
            saeId: sId,
            results: (data?.semanticSearch ?? []) as FanoutResult['results'],
          })),
        );

        const allResults = await Promise.allSettled(promises);
        const merged: SemanticFeatureResult[] = [];
        for (const r of allResults) {
          if (r.status !== 'fulfilled') continue;
          const { modelId: mId, saeId: sId, results } = r.value;
          for (const item of results) {
            merged.push({
              featureIndex: Number(item.metadata?.index ?? 0),
              label: item.document ?? null,
              density: (item.metadata?.density as number) ?? null,
              similarity: item.similarity,
              modelId: mId,
              saeId: sId,
            });
          }
        }
        merged.sort((a, b) => b.similarity - a.similarity);
        setMergedSemanticResults(merged.slice(0, 50));
      } finally {
        setSemanticFanoutLoading(false);
      }
    } else {
      // Text search across the selected SAEs (model always set → no cross-model
      // ambiguity for saeIds shared between models)
      if (!modelId || saeIds.length === 0) return;
      fetchSearch({
        variables: { modelId, saeIds, query: q, limit: 50 },
      });
    }
  }, [
    searchQuery, searchMode, modelId, pairs, saeIds, isSingleSae, singleSaeId,
    semanticCollections, fetchSearch, apolloClient, promptSearchLoading,
    skipChatTemplate, activationFilterMode,
  ]);

  // Open a feature in the detail pane — never rewrites the SAE selection
  const handleSearchSelect = useCallback((index: number, resultModelId?: string, resultSaeId?: string) => {
    const targetModelId = resultModelId ?? (isSingleSae ? modelId : null);
    const targetSaeId = resultSaeId ?? (isSingleSae ? singleSaeId : null);
    if (targetModelId && targetSaeId) {
      setSelectedFeature({ modelId: targetModelId, saeId: targetSaeId, featureIndex: index });
    }
  }, [isSingleSae, modelId, singleSaeId]);

  const handleRequestQuantiles = useCallback(() => {
    if (detailModelId && detailSaeId && detailFeatureIndex != null) {
      fetchQuantiles({
        variables: {
          modelId: detailModelId,
          saeId: detailSaeId,
          featureIndex: detailFeatureIndex,
          nQuantiles: 5,
          perQuantileLimit: 5,
        },
      });
    }
  }, [detailModelId, detailSaeId, detailFeatureIndex, fetchQuantiles]);

  // Max feature index for navigation bounds (single SAE only)
  const maxFeatureIndex = isSingleSae ? totalFeatureCount : undefined;

  // Collection link for cross-navigation (single SAE only)
  const collectionLink = isSingleSae && modelId && singleSaeId
    ? SAE_TO_COLLECTION[`${modelId}::${singleSaeId}`] ?? null
    : null;

  // Header index input reflects the open feature when it belongs to the single selection
  const headerFeatureIndex = isSingleSae &&
    selectedFeature &&
    selectedFeature.modelId === modelId &&
    selectedFeature.saeId === singleSaeId
    ? selectedFeature.featureIndex
    : null;

  // Active search results depend on mode
  const isSemanticSearch = searchMode === 'semantic';
  const isPromptSearch = searchMode === 'prompt';
  const activeSearchLoading = isPromptSearch
    ? promptSearchLoading
    : isSemanticSearch
      ? semanticFanoutLoading
      : searchLoading;
  const hasActiveResults = isPromptSearch
    ? promptSearchAsSemanticResults.length > 0
    : isSemanticSearch
      ? semanticSearchResults.length > 0
      : searchResults.length > 0;
  const activeResultCount = isPromptSearch
    ? promptSearchAsSemanticResults.length
    : isSemanticSearch
      ? semanticSearchResults.length
      : searchResults.length;

  // Show SAE badge in results when multiple SAEs are selected
  const showSaeBadge = !isSingleSae;

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
          <PageNav variant="solid" size="sm" />
          <h1 className="font-semibold text-sm">SAE Feature Explorer</h1>
          <ModeToggle />
        </header>

        <main className="flex-1 overflow-hidden flex flex-col">
          <div className="max-w-7xl mx-auto px-4 pt-4 pb-2 shrink-0 w-full">
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
              <FeatureHeader
                  modelId={modelId}
                  modelOptions={modelOptions}
                  onModelChange={setModel}
                  saeOptions={saeOptions}
                  selectedSaeIds={saeIds}
                  onSaeIdsChange={setSaeIds}
                  isSingleSae={isSingleSae}
                  featureIndex={headerFeatureIndex}
                  onFeatureIndexChange={handleFeatureIndexChange}
                  searchQuery={searchQuery}
                  onSearchQueryChange={setSearchQuery}
                  onSearch={handleSearch}
                  maxFeatureIndex={maxFeatureIndex}
                  collectionLink={collectionLink}
                  searchMode={searchMode}
                  onSearchModeChange={setSearchMode}
                  hasSemanticSearch={hasAnySemanticCollection}
                  hasPromptSearch={pairs.length > 0}
                />
            )}
          </div>

          {/* Grid fills remaining viewport height — each column scrolls independently */}
          {!modelsLoading && models.length > 0 && (
            <div className="flex-1 min-h-0 max-w-7xl mx-auto px-4 pb-4 w-full overflow-y-auto lg:overflow-hidden">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:h-full">
                {/* Left: Search results */}
                <div className="lg:col-span-1 flex flex-col min-h-0">
                  {/* Token strip (prompt mode only) */}
                  {isPromptSearch && promptActivations && (
                    <div className="shrink-0 mb-2">
                      <PromptTokenActivations
                        layers={attachedPromptLayers}
                        tokenStrings={promptActivations.tokenStrings}
                        selectedTokenIdx={selectedTokenInfo?.tokenIdx ?? null}
                        onTokenSelect={setSelectedTokenInfo}
                        highlightedFeatureIndex={detailFeatureIndex}
                        highlightedFeatureSaeId={detailSaeId}
                        highlightedFeatureLabel={feature?.label}
                        onClearHighlight={() => setSelectedFeature(null)}
                      />
                    </div>
                  )}

                  {/* Header (+ reset from token view back to whole-prompt pooling) */}
                  <div className="flex items-center gap-2 shrink-0 mb-1">
                    <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      {isPromptSearch && selectedTokenInfo
                        ? `Token "${selectedTokenInfo.token}" features (${selectedTokenInfo.features.length})`
                        : hasActiveResults
                          ? `${isPromptSearch ? 'Prompt' : isSemanticSearch ? 'Semantic' : 'Search'} Results (${
                              isPromptSearch && activeResultCount === MAX_POOLED_ROWS
                                ? `top ${MAX_POOLED_ROWS}`
                                : activeResultCount
                            })`
                          : 'Search Features'}
                    </h3>
                    {isPromptSearch && selectedTokenInfo && (
                      <button
                        onClick={() => setSelectedTokenInfo(null)}
                        className="flex items-center gap-0.5 text-[10px] text-muted-foreground hover:text-foreground rounded px-1 py-0.5 hover:bg-muted transition-colors"
                        title="Back to pooled results for the whole prompt"
                      >
                        <X className="h-3 w-3" />
                        All tokens
                      </button>
                    )}
                  </div>

                  {isPromptSearch && promptSearchError && (
                    <p className="text-xs text-destructive shrink-0">{promptSearchError}</p>
                  )}

                  {/* Skip chat template toggle (prompt mode only) */}
                  {isPromptSearch && (
                    <label className="flex items-center gap-1.5 shrink-0 mb-1 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={skipChatTemplate}
                        onChange={(e) => setSkipChatTemplate(e.target.checked)}
                        className="h-3 w-3 rounded border-border accent-primary"
                      />
                      <span className="text-[10px] text-muted-foreground select-none">
                        Raw tokens (skip chat template)
                      </span>
                    </label>
                  )}

                  {/* Activation filter mode (prompt mode only) */}
                  {isPromptSearch && (
                    <div className="space-y-1 shrink-0 mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground shrink-0">Filter:</span>
                        <ToggleGroup
                          type="single"
                          value={activationFilterMode}
                          onValueChange={(v) => {
                            if (v) {
                              setActivationFilterMode(v as 'NONE' | 'NEURONPEDIA' | 'COVERAGE_BOS' | 'COVERAGE_NO_BOS');
                              setPromptActivations(null);
                              setSelectedTokenInfo(null);
                            }
                          }}
                          variant="outline"
                          className="flex-1"
                        >
                          <ToggleGroupItem value="NONE" className="text-[10px] h-6 px-1.5 flex-1">None</ToggleGroupItem>
                          <ToggleGroupItem value="NEURONPEDIA" className="text-[10px] h-6 px-1.5 flex-1">Top-50</ToggleGroupItem>
                          <ToggleGroupItem value="COVERAGE_BOS" className="text-[10px] h-6 px-1.5 flex-1">Coverage</ToggleGroupItem>
                          <ToggleGroupItem value="COVERAGE_NO_BOS" className="text-[10px] h-6 px-1.5 flex-1">Strict</ToggleGroupItem>
                        </ToggleGroup>
                      </div>
                      <p className="text-[9px] text-muted-foreground leading-tight">
                        {activationFilterMode === 'NONE'
                          ? 'All nonzero features, no filtering'
                          : activationFilterMode === 'NEURONPEDIA'
                          ? 'Top 50 features per token (Neuronpedia-style)'
                          : activationFilterMode === 'COVERAGE_BOS'
                          ? 'Removes features firing on >80% of all positions'
                          : 'Removes features firing on >80% of prompt tokens only'}
                      </p>
                    </div>
                  )}

                  {/* Prompt pooling controls (only when showing pooled results, not token features) */}
                  {isPromptSearch && hasActiveResults && !selectedTokenInfo && (
                    <div className="space-y-2 border rounded-md p-2 bg-muted/30 shrink-0 mb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground shrink-0">Pool:</span>
                        <ToggleGroup
                          type="single"
                          value={promptPooling}
                          onValueChange={(v) => v && setPromptPooling(v as 'max' | 'mean' | 'last')}
                          variant="outline"
                          className="flex-1"
                        >
                          <ToggleGroupItem value="max" className="text-[10px] h-6 px-2 flex-1">Max</ToggleGroupItem>
                          <ToggleGroupItem value="mean" className="text-[10px] h-6 px-2 flex-1">Mean</ToggleGroupItem>
                          <ToggleGroupItem value="last" className="text-[10px] h-6 px-2 flex-1">Last</ToggleGroupItem>
                        </ToggleGroup>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-muted-foreground shrink-0">Density ≤</span>
                        <Slider
                          value={[promptMaxDensity]}
                          onValueChange={([v]) => setPromptMaxDensity(v)}
                          min={0.0001}
                          max={0.1}
                          step={0.0001}
                          className="flex-1"
                        />
                        <span className="text-[10px] font-mono text-muted-foreground w-12 text-right">
                          {promptMaxDensity < 0.001 ? promptMaxDensity.toExponential(0) : promptMaxDensity.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Scrollable results area */}
                  <div className="flex-1 min-h-0 overflow-y-auto">
                    {activeSearchLoading ? (
                      <div className="flex items-center justify-center gap-2 py-4">
                        <Spinner className="h-4 w-4" />
                        {isPromptSearch && (
                          <span className="text-xs text-muted-foreground">Running inference...</span>
                        )}
                      </div>
                    ) : isPromptSearch && selectedTokenInfo ? (
                      /* Token-level feature list (replaces pooled results when token is selected) */
                      <FeatureSearchResults
                        results={[]}
                        onSelect={handleSearchSelect}
                        selectedFeature={selectedFeature}
                        mode="prompt"
                        semanticResults={selectedTokenInfo.features.map((f) => ({
                          featureIndex: f.index,
                          label: f.label || null,
                          density: f.density,
                          similarity: f.activation,
                          modelId: selectedTokenInfo.modelId,
                          saeId: selectedTokenInfo.saeId,
                        }))}
                      />
                    ) : hasActiveResults ? (
                      <FeatureSearchResults
                        results={searchResults}
                        onSelect={handleSearchSelect}
                        selectedFeature={selectedFeature}
                        mode={searchMode}
                        semanticResults={
                          isPromptSearch ? promptSearchAsSemanticResults
                            : isSemanticSearch ? semanticSearchResults
                              : undefined
                        }
                        showSaeBadge={showSaeBadge}
                      />
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        {pairs.length === 0
                          ? 'Select at least one SAE to search.'
                          : !isSingleSae
                            ? `Search across ${pairs.length} SAEs — click a result to open it in the detail pane.`
                            : 'Search by label or browse with the arrow buttons.'}
                      </p>
                    )}
                  </div>
                </div>

                {/* Right: Feature detail + statistics + similar + activations */}
                <div className="lg:col-span-2 overflow-y-auto space-y-4">
                  {selectedFeature == null ? (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      {pairs.length > 1
                        ? `${pairs.length} SAEs selected. Use the search to find features across them — clicking a result opens it here without changing your selection.`
                        : pairs.length === 0
                          ? 'No SAEs selected.'
                          : 'Select a feature to view details.'}
                    </div>
                  ) : featureLoading ? (
                    <div className="flex justify-center py-8">
                      <Spinner className="h-5 w-5" />
                    </div>
                  ) : feature ? (
                    <>
                      <div className="border rounded-lg p-4 bg-card">
                        <FeatureDetailCard
                          feature={feature}
                          onLabelUpdated={() => {
                            if (detailModelId && detailSaeId && detailFeatureIndex != null) {
                              fetchFeature({
                                variables: {
                                  modelId: detailModelId,
                                  saeId: detailSaeId,
                                  featureIndex: detailFeatureIndex,
                                },
                              });
                            }
                          }}
                        />
                      </div>

                      <FeatureStatistics
                        feature={feature}
                        activations={activations}
                        allDensities={allDensities}
                        densitiesLoading={densitiesLoading}
                        hoveredActivationValue={hoveredActivationValue}
                      />

                      {detailSemanticCollection && (
                        <SimilarFeatures
                          collectionName={detailSemanticCollection}
                          featureIndex={feature.featureIndex}
                          featureLabel={feature.label}
                          onSelectFeature={(index: number) => {
                            if (detailModelId && detailSaeId) {
                              setSelectedFeature({ modelId: detailModelId, saeId: detailSaeId, featureIndex: index });
                            }
                          }}
                          selectedFeature={selectedFeature}
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
                  ) : (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      Feature #{selectedFeature.featureIndex} not found.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
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
              currentFeature={feature}
              onClose={closeChat}
              sessions={chatSessions}
              sessionsLoading={chatSessionsLoading}
              activeSessionId={activeSessionId}
              onSelectSession={handleSelectSession}
              onDeleteSession={deleteSession}
              onNewChat={handleNewChat}
              onUserMessageSent={handleUserMessageSent}
              onAssistantMessageComplete={handleAssistantMessageComplete}
              loadedMessages={loadedMessages}
              onSelectModel={handleSelectModel}
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
          <SteeringIdenticon
            features={floatingButtonFeatures}
            size={32}
            fallback={<Sparkles className="size-4" />}
            crossfadeOnChange
          />
          <span className="sr-only">Open steered chat</span>
        </Button>
      )}
    </div>
  );
}
