import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { apolloClient } from '@/lib/utils/apollo-client';
import { HAS_DOCUMENT_ACTIVATIONS, SEARCH_SAE_FEATURES, SEARCH_DOCUMENTS_BY_FEATURE_INDICES } from '@/lib/graphql/queries';
import type { DocumentActivationResult } from '@/lib/graphql/mutations';
import type { HighlightMap } from '@/lib/types/types';

type FeatureSearchStatus = 'idle' | 'searching' | 'error';

/** Mirrors the backend DocumentRankingMode GraphQL enum. */
export type DocumentRankingMode = 'MAX' | 'SUM' | 'SCALED_SUM' | 'MATCHING_FEATURES';

/** How many ranked documents to fetch (drives the scatter-plot highlight map). */
const FETCH_LIMIT = 100;

interface SaeInfo {
  modelId: string;
  saeId: string;
}

export interface SelectedFeature {
  featureIndex: number;
  label: string | null;
  density: number | null;
}

export interface UseDocumentFeatureSearchReturn {
  // Autocomplete suggestions
  suggestions: SelectedFeature[];
  suggestionsLoading: boolean;
  searchFeatures: (query: string) => void;

  // Selection
  selectedFeatures: SelectedFeature[];
  addFeature: (feature: SelectedFeature) => void;
  removeFeature: (featureIndex: number) => void;
  clearFeatures: () => void;

  // Ranking
  rankingMode: DocumentRankingMode;
  setRankingMode: (mode: DocumentRankingMode) => void;

  // Document results
  highlightMap: HighlightMap | undefined;
  results: DocumentActivationResult[];
  totalResults: number;
  status: FeatureSearchStatus;
  error: string | null;
  hasActivations: boolean | null;
}

export function useDocumentFeatureSearch(
  collectionName: string | null,
  saeInfo: SaeInfo | null,
): UseDocumentFeatureSearchReturn {
  // Autocomplete state
  const [suggestions, setSuggestions] = useState<SelectedFeature[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const suggestionsRequestId = useRef(0);
  const debounceTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Selection state
  const [selectedFeatures, setSelectedFeatures] = useState<SelectedFeature[]>([]);
  const [rankingMode, setRankingMode] = useState<DocumentRankingMode>('SCALED_SUM');

  // Document results state
  const [results, setResults] = useState<DocumentActivationResult[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [status, setStatus] = useState<FeatureSearchStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [hasActivations, setHasActivations] = useState<boolean | null>(null);
  const docRequestId = useRef(0);

  // Check if document activations exist
  useEffect(() => {
    if (!collectionName || !saeInfo) {
      setHasActivations(null);
      return;
    }

    let cancelled = false;
    apolloClient
      .query<{ hasDocumentActivations: boolean }>({
        query: HAS_DOCUMENT_ACTIVATIONS,
        variables: { collectionName },
        fetchPolicy: 'network-only',
      })
      .then(({ data }) => {
        if (!cancelled) setHasActivations(data?.hasDocumentActivations ?? false);
      })
      .catch(() => {
        if (!cancelled) setHasActivations(false);
      });

    return () => { cancelled = true; };
  }, [collectionName, saeInfo?.modelId, saeInfo?.saeId]);

  // Build highlight map from results
  const highlightMap = useMemo((): HighlightMap | undefined => {
    if (results.length === 0) return undefined;
    const map: HighlightMap = new Map();
    const maxScore = results[0].score;
    for (const r of results) {
      if (r.rowIndex != null && maxScore > 0) {
        map.set(r.rowIndex, r.score / maxScore);
      }
    }
    return map.size > 0 ? map : undefined;
  }, [results]);

  // Debounced feature label search (autocomplete)
  const searchFeatures = useCallback(
    (query: string) => {
      if (!saeInfo) return;

      clearTimeout(debounceTimer.current);

      if (!query.trim()) {
        setSuggestions([]);
        setSuggestionsLoading(false);
        return;
      }

      setSuggestionsLoading(true);

      debounceTimer.current = setTimeout(async () => {
        const currentId = ++suggestionsRequestId.current;
        try {
          const { data } = await apolloClient.query<{
            saeFeatureSearch: Array<{
              feature: { featureIndex: number; label: string | null; density: number | null };
            }>;
          }>({
            query: SEARCH_SAE_FEATURES,
            variables: {
              modelId: saeInfo.modelId,
              saeId: saeInfo.saeId,
              query: query.trim(),
              limit: 30,
            },
            fetchPolicy: 'network-only',
          });

          if (suggestionsRequestId.current !== currentId) return;

          const features: SelectedFeature[] = (data?.saeFeatureSearch ?? []).map((r) => ({
            featureIndex: r.feature.featureIndex,
            label: r.feature.label,
            density: r.feature.density,
          }));
          setSuggestions(features);
        } catch {
          if (suggestionsRequestId.current !== currentId) return;
          setSuggestions([]);
        } finally {
          if (suggestionsRequestId.current === currentId) {
            setSuggestionsLoading(false);
          }
        }
      }, 300);
    },
    [saeInfo],
  );

  // Search documents by selected feature indices
  const searchDocuments = useCallback(
    async (features: SelectedFeature[]) => {
      // Bump the request id even when clearing, so a clear invalidates any
      // in-flight request (e.g. the query fired just before a collection
      // switch resets the selection) instead of letting its late response
      // repopulate the emptied results.
      const currentId = ++docRequestId.current;

      if (!collectionName || features.length === 0) {
        setResults([]);
        setTotalResults(0);
        setStatus('idle');
        setError(null);
        return;
      }

      setStatus('searching');
      setError(null);

      try {
        const { data } = await apolloClient.query<{
          searchDocumentsByFeatureIndices: {
            results: DocumentActivationResult[];
            totalResults: number;
          };
        }>({
          query: SEARCH_DOCUMENTS_BY_FEATURE_INDICES,
          variables: {
            collectionName,
            featureIndices: features.map((f) => f.featureIndex),
            limit: FETCH_LIMIT,
            ranking: rankingMode,
          },
          fetchPolicy: 'network-only',
        });

        if (docRequestId.current !== currentId) return;

        const response = data?.searchDocumentsByFeatureIndices;
        setResults(response?.results ?? []);
        setTotalResults(response?.totalResults ?? 0);
        setStatus('idle');
      } catch (err) {
        if (docRequestId.current !== currentId) return;
        setError(err instanceof Error ? err.message : 'Document search failed');
        setStatus('error');
      }
    },
    [collectionName, rankingMode],
  );

  // Trigger document search whenever the selection or ranking mode changes.
  // The request-id counter in searchDocuments handles overlapping calls.
  useEffect(() => {
    searchDocuments(selectedFeatures);
  }, [selectedFeatures, searchDocuments]);

  const addFeature = useCallback(
    (feature: SelectedFeature) => {
      setSelectedFeatures((prev) => {
        if (prev.some((f) => f.featureIndex === feature.featureIndex)) return prev;
        return [...prev, feature];
      });
    },
    [],
  );

  const removeFeature = useCallback(
    (featureIndex: number) => {
      setSelectedFeatures((prev) => prev.filter((f) => f.featureIndex !== featureIndex));
    },
    [],
  );

  const clearFeatures = useCallback(() => {
    setSelectedFeatures([]);
    setResults([]);
    setTotalResults(0);
    setStatus('idle');
    setError(null);
    setSuggestions([]);
  }, []);

  // Reset on collection change
  useEffect(() => {
    clearFeatures();
  }, [collectionName, clearFeatures]);

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => clearTimeout(debounceTimer.current);
  }, []);

  return {
    suggestions,
    suggestionsLoading,
    searchFeatures,
    selectedFeatures,
    addFeature,
    removeFeature,
    clearFeatures,
    rankingMode,
    setRankingMode,
    highlightMap,
    results,
    totalResults,
    status,
    error,
    hasActivations,
  };
}
