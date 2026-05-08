import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { apolloClient } from '@/lib/utils/apollo-client';
import { RUN_PROMPT_HIGHLIGHT } from '@/lib/graphql/mutations';
import type { PromptHighlightFeature, PromptHighlightResult } from '@/lib/graphql/mutations';
import { SAE_FEATURE_INDEX_FIELD, parseSaeId } from '@/lib/utils/saeCollections';
import { ensureModelLoaded } from '@/lib/utils/modelLoader';
import type { HighlightMap, SemanticSearchResult } from '@/lib/types/types';

type PromptHighlightStatus = 'idle' | 'loading_model' | 'running' | 'error';

export const MAX_HIGHLIGHTED_FEATURES = 20;

interface SaeInfo {
  modelId: string;
  saeId: string;
}

/** A resolved feature with its point index in the scatter plot. */
export interface ResolvedFeature {
  featureIndex: number;
  activation: number;
  pointIndex: number;
}

interface UsePromptHighlightReturn {
  highlightMap: HighlightMap | undefined;
  /** Top features after density filter + top-N, with point indices resolved. */
  topFeatures: ResolvedFeature[];
  status: PromptHighlightStatus;
  error: string | null;
  activePrompt: string | null;
  submit: (prompt: string) => void;
  clear: () => void;
}

export function usePromptHighlight(
  saeInfo: SaeInfo | null,
  itemMetadata: Record<string, unknown>[],
  maxDensity: number | null,
): UsePromptHighlightReturn {
  // Raw mutation result — all nonzero features, unfiltered
  const [allFeatures, setAllFeatures] = useState<PromptHighlightFeature[] | null>(null);
  const [status, setStatus] = useState<PromptHighlightStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [activePrompt, setActivePrompt] = useState<string | null>(null);
  const requestIdRef = useRef(0);
  const statusRef = useRef(status);
  statusRef.current = status;

  // Build featureIndex -> pointIndex lookup once when metadata changes
  const featureIndexToPointIndex = useMemo(() => {
    if (!itemMetadata || itemMetadata.length === 0) return null;
    const map = new Map<number, number>();
    for (let i = 0; i < itemMetadata.length; i++) {
      const idx = itemMetadata[i]?.[SAE_FEATURE_INDEX_FIELD];
      if (idx !== undefined && idx !== null) {
        map.set(Number(idx), i);
      }
    }
    return map.size > 0 ? map : null;
  }, [itemMetadata]);

  // Derive filtered top features + highlight map from raw results + maxDensity.
  // Re-runs instantly when maxDensity changes (no new inference needed).
  const { highlightMap, topFeatures } = useMemo(() => {
    if (!allFeatures || allFeatures.length === 0 || !featureIndexToPointIndex) {
      return { highlightMap: undefined, topFeatures: [] as ResolvedFeature[] };
    }

    // Resolve point indices and apply density filter
    const resolved: (ResolvedFeature & { density?: number })[] = [];
    for (const feat of allFeatures) {
      const pointIndex = featureIndexToPointIndex.get(feat.featureIndex);
      if (pointIndex === undefined) continue;
      if (maxDensity !== null) {
        const density = itemMetadata[pointIndex]?.density;
        if (typeof density === 'number' && density > maxDensity) continue;
      }
      resolved.push({ featureIndex: feat.featureIndex, activation: feat.activation, pointIndex });
    }

    // Sort by activation descending, take top N
    resolved.sort((a, b) => b.activation - a.activation);
    const top = resolved.slice(0, MAX_HIGHLIGHTED_FEATURES);

    if (top.length === 0) {
      return { highlightMap: undefined, topFeatures: [] as ResolvedFeature[] };
    }

    // Normalize to 0-1 and build HighlightMap
    const maxAct = top[0].activation;
    const map: HighlightMap = new Map();
    for (const feat of top) {
      map.set(feat.pointIndex, maxAct > 0 ? feat.activation / maxAct : 1);
    }

    return {
      highlightMap: map.size > 0 ? map : undefined,
      topFeatures: top,
    };
  }, [allFeatures, featureIndexToPointIndex, itemMetadata, maxDensity]);

  const clear = useCallback(() => {
    setAllFeatures(null);
    setStatus('idle');
    setError(null);
    setActivePrompt(null);
  }, []);

  // Reset when collection changes
  useEffect(() => {
    clear();
  }, [saeInfo?.modelId, saeInfo?.saeId, clear]);

  const submit = useCallback(
    (prompt: string) => {
      if (!saeInfo || !featureIndexToPointIndex) return;
      if (statusRef.current === 'loading_model' || statusRef.current === 'running') return;

      const currentRequestId = ++requestIdRef.current;

      const run = async () => {
        setStatus('loading_model');
        setError(null);

        try {
          const loadError = await ensureModelLoaded();
          if (requestIdRef.current !== currentRequestId) return;
          if (loadError) {
            setError(loadError);
            setStatus('error');
            return;
          }
        } catch (err) {
          if (requestIdRef.current !== currentRequestId) return;
          setError(err instanceof Error ? err.message : 'Failed to load model');
          setStatus('error');
          return;
        }

        setStatus('running');
        try {
          const parsed = parseSaeId(saeInfo.saeId);
          const { data } = await apolloClient.mutate<{
            runPromptHighlight: PromptHighlightResult;
          }>({
            mutation: RUN_PROMPT_HIGHLIGHT,
            variables: {
              input: {
                prompt,
                layer: parsed.layerIndex,
                width: parsed.width,
                hookType: parsed.hookType,
              },
            },
          });

          if (requestIdRef.current !== currentRequestId) return;

          const result = data?.runPromptHighlight;
          if (result?.error) {
            setError(result.error);
            setStatus('error');
            return;
          }

          // Store full result — filtering/top-N happens in useMemo
          setAllFeatures(result?.features ?? []);
          setActivePrompt(prompt);
          setStatus('idle');
        } catch (err) {
          if (requestIdRef.current !== currentRequestId) return;
          setError(err instanceof Error ? err.message : 'SAE inference failed');
          setStatus('error');
        }
      };

      run();
    },
    [saeInfo, featureIndexToPointIndex],
  );

  return { highlightMap, topFeatures, status, error, activePrompt, submit, clear };
}

// ── Utility: build table results from resolved features ──────────────

/** Convert resolved features into SemanticSearchResult[] for the results table. */
export function buildPromptHighlightResults(
  topFeatures: ResolvedFeature[],
  ids: string[],
  documents: string[],
  itemMetadata: Record<string, unknown>[],
  labelField: string | null,
): SemanticSearchResult[] {
  if (topFeatures.length === 0) return [];
  const maxAct = topFeatures[0].activation;
  return topFeatures.map((feat) => {
    const meta = itemMetadata[feat.pointIndex] ?? {};
    const label = labelField && meta[labelField] != null
      ? String(meta[labelField])
      : String(meta['label'] ?? `Feature #${feat.featureIndex}`);
    return {
      id: ids[feat.pointIndex] ?? String(feat.featureIndex),
      label,
      document: documents[feat.pointIndex] ?? '',
      category: '',
      similarity: maxAct > 0 ? feat.activation / maxAct : 1,
      distance: 0,
      metadata: { ...meta, activation: feat.activation },
    };
  });
}
