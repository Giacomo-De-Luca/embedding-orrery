'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { useQuery, useLazyQuery, useApolloClient } from '@apollo/client/react';
import Link from 'next/link';
import { ArrowLeft, Sparkles, Sun, Moon } from 'lucide-react';
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
import { FeatureSearchResults, type SemanticFeatureResult } from './components/FeatureSearchResults';
import { FeatureStatistics } from './components/FeatureStatistics';
import { SimilarFeatures } from './components/SimilarFeatures';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Separator } from '@/lib/ui-primitives/separator';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { Slider } from '@/lib/ui-primitives/slider';
import { RUN_PROMPT_ACTIVATIONS } from '@/lib/graphql/mutations';
import type { PromptActivationsResult } from '@/lib/graphql/mutations';
import { SAE_TO_COLLECTION, getSemanticCollectionName, getSemanticCollections, parseSaeId } from '@/lib/utils/saeCollections';
import { ensureModelLoaded } from '@/lib/utils/modelLoader';
import { ChatPanel } from './components/ChatInterface';
import { useModelIdentityStore } from '@/lib/stores/useModelIdentityStore';
import { PromptTokenActivations, type SelectedTokenInfo } from './components/PromptTokenActivations';
import { useChatSessions } from '@/lib/hooks/useChatSessions';
import { useSaeSelectors } from './hooks/useSaeSelectors';
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
  const searchParams = useSearchParams();
  const router = useRouter();
  const apolloClient = useApolloClient();

  // URL state — supports both legacy (?modelId=&saeId=) and multi-SAE (?model=&layer=&hookType=&width=)
  const modelIdParam = searchParams.get('modelId');
  const saeIdParam = searchParams.get('saeId');
  const featureParam = searchParams.get('featureIndex');
  const modelParam = searchParams.get('model');
  const layerParam = searchParams.get('layer');
  const hookTypeParam = searchParams.get('hookType');
  const widthParam = searchParams.get('width');

  // Local state
  const [featureIndex, setFeatureIndex] = useState<number | null>(
    featureParam != null ? parseInt(featureParam, 10) : null,
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
    setLoadedMessages([]);
  }, [setActiveSessionId]);

  // ---------- Queries ----------

  const { data: modelsData, loading: modelsLoading } = useQuery<{ saeModels: SaeModelInfo[] }>(
    GET_SAE_MODELS,
  );
  const models = useMemo(() => modelsData?.saeModels ?? [], [modelsData]);

  // Build initial selectors from URL params (legacy or multi-SAE format)
  const initialSelectors = useMemo(() => {
    if (modelIdParam && saeIdParam) {
      // Legacy format: reverse-parse saeId into individual selectors
      const parsed = parseSaeId(saeIdParam);
      return {
        model: modelIdParam,
        layer: String(parsed.layerIndex),
        hookType: parsed.hookType as string,
        width: parsed.width,
      };
    }
    // Multi-SAE format: read individual selector params (null = "All")
    return {
      model: modelParam,
      layer: layerParam,
      hookType: hookTypeParam,
      width: widthParam,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cascading SAE selectors
  const {
    selectors,
    setModel,
    setLayer,
    setHookType,
    setWidth,
    modelOptions,
    layerOptions,
    hookTypeOptions,
    widthOptions,
    resolvedSaePairs,
    isSingleSae,
    singleModelId,
    singleSaeId,
    totalFeatureCount,
  } = useSaeSelectors(models, initialSelectors);

  // Aliases reading from the selector hook (for queries, URL sync, etc.)
  const modelId = singleModelId;
  const saeId = singleSaeId;

  // Bridge useSaeSelectors output into the Zustand store (single source of truth).
  // Always propagate the model selector value so the chat knows which model to
  // load, even when multiple SAEs are resolved. saeId is only set when exactly
  // one SAE is resolved (needed for label lookups / prompt search).
  useEffect(() => {
    const effectiveModelId = singleModelId ?? selectors.model ?? null;
    const effectiveSaeId = isSingleSae ? singleSaeId : null;
    useModelIdentityStore.getState().setIdentity(effectiveModelId, effectiveSaeId);
  }, [isSingleSae, singleModelId, singleSaeId, selectors.model]);

  // Handle model selection from the chat input — updates selectors only; bridge effect syncs store
  const handleSelectModel = useCallback((newModelId: string, newSaeId: string) => {
    const parsed = parseSaeId(newSaeId);
    setModel(newModelId);
    setLayer(String(parsed.layerIndex));
    setHookType(parsed.hookType);
    setWidth(parsed.width);
  }, [setModel, setLayer, setHookType, setWidth]);

  // Auto-select first model when no URL params and models load
  useEffect(() => {
    if (!modelIdParam && !saeIdParam && models.length > 0 && resolvedSaePairs.length === 0) {
      const first = models[0];
      const parsed = parseSaeId(first.saeId);
      setModel(first.modelId);
      setLayer(String(parsed.layerIndex));
      setHookType(parsed.hookType);
      setWidth(parsed.width);
    }
  }, [models, modelIdParam, saeIdParam, resolvedSaePairs.length, setModel, setLayer, setHookType, setWidth]);

  // Semantic collection: single-SAE mode uses direct lookup, multi-SAE checks all resolved
  const semanticCollectionName = useMemo(
    () => isSingleSae && modelId && saeId ? getSemanticCollectionName(`${modelId}::${saeId}`) : null,
    [isSingleSae, modelId, saeId],
  );
  const hasAnySemanticCollection = useMemo(
    () => isSingleSae
      ? !!semanticCollectionName
      : getSemanticCollections(resolvedSaePairs).length > 0,
    [isSingleSae, semanticCollectionName, resolvedSaePairs],
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

  // Semantic search (single-collection)
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

  // Semantic search results mapped to display format (single-collection)
  const singleSemanticResults: SemanticFeatureResult[] = useMemo(() => {
    if (!semanticSearchData?.semanticSearch) return [];
    return semanticSearchData.semanticSearch.map((r) => ({
      featureIndex: Number(r.metadata?.index ?? 0),
      label: r.document ?? null,
      density: (r.metadata?.density as number) ?? null,
      similarity: r.similarity,
    }));
  }, [semanticSearchData]);

  // Active semantic results: use fan-out results when multi-SAE, single otherwise
  const semanticSearchResults = !isSingleSae && mergedSemanticResults.length > 0
    ? mergedSemanticResults
    : singleSemanticResults;

  // Derive ranked feature list from per-token activations using selected pooling strategy
  const promptSearchAsSemanticResults: SemanticFeatureResult[] = useMemo(() => {
    if (!promptActivations || promptActivations.layers.length === 0) return [];

    const featureMap = new Map<number, { activation: number; label: string; density: number | null }>();

    for (const layer of promptActivations.layers) {
      const tokens = layer.tokens;
      if (tokens.length === 0) continue;

      if (promptPooling === 'last') {
        // Last token only
        const lastToken = tokens[tokens.length - 1];
        for (const feat of lastToken.features) {
          featureMap.set(feat.index, { activation: feat.activation, label: feat.label, density: feat.density });
        }
      } else {
        // Accumulate per-feature across all tokens
        const accumulator = new Map<number, { sum: number; count: number; max: number; label: string; density: number | null }>();
        for (const token of tokens) {
          for (const feat of token.features) {
            const existing = accumulator.get(feat.index);
            if (existing) {
              existing.sum += feat.activation;
              existing.count += 1;
              if (feat.activation > existing.max) existing.max = feat.activation;
            } else {
              accumulator.set(feat.index, {
                sum: feat.activation, count: 1, max: feat.activation,
                label: feat.label, density: feat.density,
              });
            }
          }
        }
        for (const [idx, { sum, count, max, label, density }] of accumulator) {
          const activation = promptPooling === 'mean' ? sum / count : max;
          featureMap.set(idx, { activation, label, density });
        }
      }
    }

    // Filter by density threshold (exclude ultra-common features)
    const filtered = [...featureMap.entries()].filter(([, { density }]) => {
      if (density === null) return true;
      return density <= promptMaxDensity;
    });

    // Sort by activation descending
    const sorted = filtered
      .sort(([, a], [, b]) => b.activation - a.activation);
    if (sorted.length === 0) return [];
    const maxAct = sorted[0][1].activation;
    return sorted.map(([featureIndex, { activation, label, density }]) => ({
      featureIndex,
      label: label || null,
      density,
      similarity: maxAct > 0 ? activation / maxAct : 1,
    }));
  }, [promptActivations, promptPooling, promptMaxDensity]);

  // Clear stale fan-out / prompt results when selectors change
  useEffect(() => {
    setMergedSemanticResults([]);
    setPromptActivations(null);
    setPromptSearchError(null);
    setSelectedTokenInfo(null);
  }, [resolvedSaePairs]);

  // ---------- Effects ----------

  // Fetch feature + activations when index or model changes (single SAE only)
  useEffect(() => {
    if (modelId && saeId && featureIndex != null) {
      fetchFeature({ variables: { modelId, saeId, featureIndex } });
      fetchActivations({ variables: { modelId, saeId, featureIndex, limit: 20 } });
    }
  }, [modelId, saeId, featureIndex, fetchFeature, fetchActivations]);

  // Fetch densities once when model/sae changes (single SAE only)
  useEffect(() => {
    if (modelId && saeId) {
      fetchDensities({ variables: { modelId, saeId } });
    }
  }, [modelId, saeId, fetchDensities]);

  // Sync URL
  useEffect(() => {
    const params = new URLSearchParams();
    if (isSingleSae && modelId && saeId) {
      params.set('modelId', modelId);
      params.set('saeId', saeId);
      if (featureIndex != null) params.set('featureIndex', featureIndex.toString());
    } else {
      // Multi-SAE mode: encode individual selectors
      if (selectors.model) params.set('model', selectors.model);
      if (selectors.layer) params.set('layer', selectors.layer);
      if (selectors.hookType) params.set('hookType', selectors.hookType);
      if (selectors.width) params.set('width', selectors.width);
    }
    const newSearch = `?${params.toString()}`;
    if (newSearch !== window.location.search) {
      router.replace(newSearch, { scroll: false });
    }
  }, [isSingleSae, modelId, saeId, featureIndex, selectors, router]);

  // ---------- Handlers ----------

  const handleFeatureIndexChange = useCallback((index: number) => {
    setFeatureIndex(index);
  }, []);

  const handleSearch = useCallback(async () => {
    const q = searchQuery.trim();
    if (!q) return;

    if (searchMode === 'prompt') {
      // Prompt activation search via runPromptActivations (gives both ranked list + token strip)
      if (!isSingleSae || !modelId || !saeId) return;
      if (promptSearchLoading) return;
      setPromptSearchLoading(true);
      setPromptSearchError(null);
      try {
        const { checkpoint: storeCheckpt, parsedSae } = useModelIdentityStore.getState();
        const loadErr = await ensureModelLoaded(storeCheckpt ?? undefined);
        if (loadErr) {
          setPromptSearchError(loadErr);
          setPromptSearchLoading(false);
          return;
        }
        const { data } = await apolloClient.mutate<{ runPromptActivations: PromptActivationsResult }>({
          mutation: RUN_PROMPT_ACTIVATIONS,
          variables: {
            input: {
              prompt: q,
              layers: parsedSae ? [parsedSae.layerIndex] : [],
              width: parsedSae?.width ?? '16k',
              topK: 0,
              modelId: modelId,
              saeId: saeId,
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
      if (isSingleSae && semanticCollectionName) {
        // Single SAE semantic search
        fetchSemanticSearch({
          variables: { collectionName: semanticCollectionName, query: q, nResults: 50 },
        });
      } else if (!isSingleSae) {
        // Fan-out semantic search across resolved SAEs
        const collections = getSemanticCollections(resolvedSaePairs);
        if (collections.length === 0) return;

        setSemanticFanoutLoading(true);
        try {
          const promises = collections.map(({ modelId: mId, saeId: sId, collectionName }): Promise<FanoutResult> =>
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
      }
    } else {
      // Text search
      if (isSingleSae && modelId && saeId) {
        fetchSearch({ variables: { modelId, saeId, query: q, limit: 50 } });
      } else {
        // Cross-SAE text search
        const saeIds = resolvedSaePairs.map((p) => p.saeId);
        const selectedModel = selectors.model; // null if "All models"
        fetchSearch({
          variables: {
            modelId: selectedModel,
            saeIds: saeIds.length > 0 ? saeIds : undefined,
            query: q,
            limit: 50,
          },
        });
      }
    }
  }, [
    searchQuery, searchMode, isSingleSae, modelId, saeId,
    semanticCollectionName, resolvedSaePairs, selectors.model,
    fetchSearch, fetchSemanticSearch, apolloClient, promptSearchLoading,
    skipChatTemplate, activationFilterMode,
  ]);

  const handleSearchSelect = useCallback((index: number, resultModelId?: string, resultSaeId?: string) => {
    if (resultModelId && resultSaeId && !isSingleSae) {
      // Cross-SAE result: drill into that specific SAE
      const parsed = parseSaeId(resultSaeId);
      setModel(resultModelId);
      setLayer(String(parsed.layerIndex));
      setHookType(parsed.hookType);
      setWidth(parsed.width);
    }
    setFeatureIndex(index);
  }, [isSingleSae, setModel, setLayer, setHookType, setWidth]);

  const handleRequestQuantiles = useCallback(() => {
    if (modelId && saeId && featureIndex != null) {
      fetchQuantiles({
        variables: { modelId, saeId, featureIndex, nQuantiles: 5, perQuantileLimit: 5 },
      });
    }
  }, [modelId, saeId, featureIndex, fetchQuantiles]);

  // Max feature index for navigation bounds (single SAE only)
  const maxFeatureIndex = isSingleSae ? totalFeatureCount : undefined;

  // Collection link for cross-navigation (single SAE only)
  const collectionLink = isSingleSae && modelId && saeId
    ? SAE_TO_COLLECTION[`${modelId}::${saeId}`] ?? null
    : null;

  // Active search results depend on mode
  const isSemanticSearch = searchMode === 'semantic';
  const isPromptSearch = searchMode === 'prompt';
  const activeSearchLoading = isPromptSearch
    ? promptSearchLoading
    : isSemanticSearch
      ? (isSingleSae ? semanticSearchLoading : semanticFanoutLoading)
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

  // Show SAE badge in results when multi-SAE
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
                  selectors={selectors}
                  modelOptions={modelOptions}
                  layerOptions={layerOptions}
                  hookTypeOptions={hookTypeOptions}
                  widthOptions={widthOptions}
                  onModelChange={setModel}
                  onLayerChange={setLayer}
                  onHookTypeChange={setHookType}
                  onWidthChange={setWidth}
                  resolvedCount={resolvedSaePairs.length}
                  isSingleSae={isSingleSae}
                  featureIndex={isSingleSae ? featureIndex : null}
                  onFeatureIndexChange={handleFeatureIndexChange}
                  searchQuery={searchQuery}
                  onSearchQueryChange={setSearchQuery}
                  onSearch={handleSearch}
                  maxFeatureIndex={maxFeatureIndex}
                  collectionLink={collectionLink}
                  searchMode={searchMode}
                  onSearchModeChange={setSearchMode}
                  hasSemanticSearch={hasAnySemanticCollection}
                  hasPromptSearch={isSingleSae}
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
                        layers={promptActivations.layers}
                        tokenStrings={promptActivations.tokenStrings}
                        onTokenSelect={setSelectedTokenInfo}
                        highlightedFeatureIndex={featureIndex}
                        highlightedFeatureLabel={feature?.label}
                        onClearHighlight={() => setFeatureIndex(null)}
                      />
                    </div>
                  )}

                  {/* Header */}
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wide shrink-0 mb-1">
                    {isPromptSearch && selectedTokenInfo
                      ? `Token "${selectedTokenInfo.token}" features (${selectedTokenInfo.features.length})`
                      : hasActiveResults
                        ? `${isPromptSearch ? 'Prompt' : isSemanticSearch ? 'Semantic' : 'Search'} Results (${activeResultCount})`
                        : 'Search Features'}
                  </h3>

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
                        selectedIndex={featureIndex}
                        mode="prompt"
                        semanticResults={selectedTokenInfo.features.map((f) => ({
                          featureIndex: f.index,
                          label: f.label || null,
                          density: f.density,
                          similarity: f.activation,
                        }))}
                      />
                    ) : hasActiveResults ? (
                      <FeatureSearchResults
                        results={searchResults}
                        onSelect={handleSearchSelect}
                        selectedIndex={featureIndex}
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
                        {!isSingleSae
                          ? `Search across ${resolvedSaePairs.length} SAEs, or select a single SAE to browse features.`
                          : 'Search by label or browse with the arrow buttons.'}
                      </p>
                    )}
                  </div>
                </div>

                {/* Right: Feature detail + statistics + similar + activations */}
                <div className="lg:col-span-2 overflow-y-auto space-y-4">
                  {!isSingleSae ? (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      {resolvedSaePairs.length > 1
                        ? `${resolvedSaePairs.length} SAEs selected. Use the search to find features across them, or narrow the selectors to browse a single SAE.`
                        : 'No SAEs match the current selection.'}
                    </div>
                  ) : featureLoading ? (
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
          <Sparkles className="size-4" />
          <span className="sr-only">Open steered chat</span>
        </Button>
      )}
    </div>
  );
}
