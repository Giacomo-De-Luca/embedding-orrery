'use client';

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import type { SaeModelInfo } from '@/lib/types/types';
import { parseSaeId, type ParsedSaeId } from '@/lib/utils/saeCollections';
import { resolveSelectionFromParams, type SelectionUrlParams } from '../utils/saeSelection';

// ── Types ────────────────────────────────────────────────────────

export interface SaePair {
  modelId: string;
  saeId: string;
}

export interface SaeOption {
  saeId: string;
  parsed: ParsedSaeId;
  featureCount: number;
}

export interface UseSaeSelectionReturn {
  /** The selected model (single-select — model load is the expensive op). */
  modelId: string | null;
  /** Selected SAEs of that model (multi-select). */
  saeIds: string[];
  /** Switch model; selects ALL of the new model's SAEs. */
  setModel: (modelId: string) => void;
  setSaeIds: (saeIds: string[]) => void;
  /** Narrow to exactly one SAE (chat model picker). */
  selectSingle: (modelId: string, saeId: string) => void;

  modelOptions: string[];
  /** SAE options for the selected model, sorted by layer then width. */
  saeOptions: SaeOption[];

  /** The selected (modelId, saeId) pairs. */
  pairs: SaePair[];
  isSingleSae: boolean;
  /** Non-null only when isSingleSae. */
  singleSaeId: string | null;
  /** Sum of feature counts across the selected SAEs. */
  totalFeatureCount: number;
}

// ── Hook ─────────────────────────────────────────────────────────

/**
 * SAE selection state for the /sae page: one model + a subset of its SAEs.
 *
 * Initializes once when GET_SAE_MODELS arrives — from the URL params when
 * present (multi `model`+`saes`, legacy `modelId`+`saeId`, or the old
 * dimension format), otherwise defaulting to the first model with ALL of
 * its SAEs selected.
 */
export function useSaeSelection(
  models: SaeModelInfo[],
  urlParams: SelectionUrlParams,
): UseSaeSelectionReturn {
  const [modelId, setModelId] = useState<string | null>(null);
  const [saeIds, setSaeIdsState] = useState<string[]>([]);
  const initializedRef = useRef(false);

  useEffect(() => {
    if (initializedRef.current || models.length === 0) return;
    initializedRef.current = true;
    const resolved = resolveSelectionFromParams(urlParams, models);
    if (resolved) {
      setModelId(resolved.modelId);
      setSaeIdsState(resolved.saeIds);
    } else {
      const first = models[0].modelId;
      setModelId(first);
      setSaeIdsState(models.filter((m) => m.modelId === first).map((m) => m.saeId));
    }
  }, [models, urlParams]);

  const modelOptions = useMemo(
    () => [...new Set(models.map((m) => m.modelId))].sort(),
    [models],
  );

  const saeOptions: SaeOption[] = useMemo(
    () =>
      models
        .filter((m) => m.modelId === modelId)
        .map((m) => ({
          saeId: m.saeId,
          parsed: parseSaeId(m.saeId),
          featureCount: m.featureCount,
        }))
        .sort(
          (a, b) =>
            a.parsed.layerIndex - b.parsed.layerIndex ||
            parseInt(a.parsed.width, 10) - parseInt(b.parsed.width, 10),
        ),
    [models, modelId],
  );

  // Clamp to available options (guards stale URL ids / model switches)
  const validSaeIds = useMemo(() => {
    const available = new Set(saeOptions.map((o) => o.saeId));
    return saeIds.filter((id) => available.has(id));
  }, [saeIds, saeOptions]);

  const setModel = useCallback(
    (newModelId: string) => {
      setModelId(newModelId);
      setSaeIdsState(models.filter((m) => m.modelId === newModelId).map((m) => m.saeId));
    },
    [models],
  );

  const setSaeIds = useCallback((ids: string[]) => setSaeIdsState(ids), []);

  const selectSingle = useCallback((newModelId: string, saeId: string) => {
    setModelId(newModelId);
    setSaeIdsState([saeId]);
  }, []);

  const pairs: SaePair[] = useMemo(
    () => (modelId ? validSaeIds.map((saeId) => ({ modelId, saeId })) : []),
    [modelId, validSaeIds],
  );

  const isSingleSae = pairs.length === 1;
  const singleSaeId = isSingleSae ? pairs[0].saeId : null;

  const totalFeatureCount = useMemo(() => {
    const counts = new Map(saeOptions.map((o) => [o.saeId, o.featureCount]));
    return validSaeIds.reduce((sum, id) => sum + (counts.get(id) ?? 0), 0);
  }, [saeOptions, validSaeIds]);

  return {
    modelId,
    saeIds: validSaeIds,
    setModel,
    setSaeIds,
    selectSingle,
    modelOptions,
    saeOptions,
    pairs,
    isSingleSae,
    singleSaeId,
    totalFeatureCount,
  };
}
