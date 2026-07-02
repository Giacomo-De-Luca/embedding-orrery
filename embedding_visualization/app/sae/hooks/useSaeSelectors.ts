'use client';

import { useState, useMemo, useCallback } from 'react';
import type { SaeModelInfo } from '@/lib/types/types';
import { parseSaeId, type HookType } from '@/lib/utils/saeCollections';

// ── Types ────────────────────────────────────────────────────────

export interface SaeSelectors {
  model: string | null;    // null = "All"
  layer: string | null;    // null = "All"
  hookType: string | null; // null = "All"
  width: string | null;    // null = "All"
}

export interface ParsedSaeEntry {
  modelId: string;
  saeId: string;
  layerIndex: number;
  hookType: HookType;
  width: string;
  featureCount: number;
  activationCount: number;
}

export interface SaePair {
  modelId: string;
  saeId: string;
}

export interface UseSaeSelectorsReturn {
  selectors: SaeSelectors;
  setModel: (v: string | null) => void;
  setLayer: (v: string | null) => void;
  setHookType: (v: string | null) => void;
  setWidth: (v: string | null) => void;

  /** Options available for each selector, filtered by the other selectors. */
  modelOptions: string[];
  layerOptions: string[];
  hookTypeOptions: string[];
  widthOptions: string[];

  /** The (modelId, saeId) pairs matching all active selectors. */
  resolvedSaePairs: SaePair[];
  /** True when exactly one SAE is resolved. */
  isSingleSae: boolean;
  /** Non-null only when isSingleSae. */
  singleModelId: string | null;
  /** Non-null only when isSingleSae. */
  singleSaeId: string | null;
  /** Feature count of the single SAE, or sum across resolved SAEs. */
  totalFeatureCount: number;
}

// ── Helpers ──────────────────────────────────────────────────────

function unique<T>(arr: T[]): T[] {
  return [...new Set(arr)];
}

/** Filter entries matching all *other* selectors (excluding the given dimension). */
function filterEntries(
  entries: ParsedSaeEntry[],
  selectors: SaeSelectors,
  exclude?: keyof SaeSelectors,
): ParsedSaeEntry[] {
  return entries.filter((e) => {
    if (exclude !== 'model' && selectors.model !== null && e.modelId !== selectors.model) return false;
    if (exclude !== 'layer' && selectors.layer !== null && e.layerIndex !== Number(selectors.layer)) return false;
    if (exclude !== 'hookType' && selectors.hookType !== null && e.hookType !== selectors.hookType) return false;
    if (exclude !== 'width' && selectors.width !== null && e.width !== selectors.width) return false;
    return true;
  });
}

// ── Hook ─────────────────────────────────────────────────────────

/**
 * Cascading SAE selector state. Parses saeIds from GET_SAE_MODELS into
 * four independent dimensions (model, layer, hook, width), each with an
 * "All" option. Derives available options and resolved SAE pairs.
 *
 * @param models — result of GET_SAE_MODELS query
 * @param initialSelectors — optional initial selector state (from URL params)
 */
export function useSaeSelectors(
  models: SaeModelInfo[],
  initialSelectors?: Partial<SaeSelectors>,
): UseSaeSelectorsReturn {
  // Compute initial state once on mount
  const initState = useMemo((): SaeSelectors => ({
    model: initialSelectors?.model ?? null,
    layer: initialSelectors?.layer ?? null,
    hookType: initialSelectors?.hookType ?? null,
    width: initialSelectors?.width ?? null,
    // Only compute on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), []);

  const [selectors, setSelectors] = useState<SaeSelectors>(initState);

  // Parse all model entries once
  const parsedEntries: ParsedSaeEntry[] = useMemo(
    () =>
      models.map((m) => {
        const parsed = parseSaeId(m.saeId);
        return {
          modelId: m.modelId,
          saeId: m.saeId,
          layerIndex: parsed.layerIndex,
          hookType: parsed.hookType,
          width: parsed.width,
          featureCount: m.featureCount,
          activationCount: m.activationCount,
        };
      }),
    [models],
  );

  // Derive options: each selector's options come from entries matching all *other* selectors
  const modelOptions = useMemo(
    () => unique(filterEntries(parsedEntries, selectors, 'model').map((e) => e.modelId)).sort(),
    [parsedEntries, selectors],
  );
  const layerOptions = useMemo(
    () => unique(filterEntries(parsedEntries, selectors, 'layer').map((e) => String(e.layerIndex))).sort((a, b) => Number(a) - Number(b)),
    [parsedEntries, selectors],
  );
  const hookTypeOptions: string[] = useMemo(
    () => unique(filterEntries(parsedEntries, selectors, 'hookType').map((e) => e.hookType as string)).sort(),
    [parsedEntries, selectors],
  );
  const widthOptions = useMemo(
    () => unique(filterEntries(parsedEntries, selectors, 'width').map((e) => e.width))
      .sort((a, b) => parseInt(a, 10) - parseInt(b, 10)),
    [parsedEntries, selectors],
  );

  // Auto-fallback: if a selector's value is no longer in its options, reset to null
  const effectiveSelectors = useMemo((): SaeSelectors => {
    return {
      model: selectors.model !== null && modelOptions.includes(selectors.model) ? selectors.model : null,
      layer: selectors.layer !== null && layerOptions.includes(selectors.layer) ? selectors.layer : null,
      hookType: selectors.hookType !== null && hookTypeOptions.includes(selectors.hookType) ? selectors.hookType : null,
      width: selectors.width !== null && widthOptions.includes(selectors.width) ? selectors.width : null,
    };
  }, [selectors, modelOptions, layerOptions, hookTypeOptions, widthOptions]);

  // Resolved pairs: entries matching all effective selectors
  const resolvedSaePairs: SaePair[] = useMemo(
    () =>
      filterEntries(parsedEntries, effectiveSelectors).map((e) => ({
        modelId: e.modelId,
        saeId: e.saeId,
      })),
    [parsedEntries, effectiveSelectors],
  );

  const isSingleSae = resolvedSaePairs.length === 1;
  const singleModelId = isSingleSae ? resolvedSaePairs[0].modelId : null;
  const singleSaeId = isSingleSae ? resolvedSaePairs[0].saeId : null;

  const totalFeatureCount = useMemo(
    () => {
      const resolved = filterEntries(parsedEntries, effectiveSelectors);
      return resolved.reduce((sum, e) => sum + e.featureCount, 0);
    },
    [parsedEntries, effectiveSelectors],
  );

  // Setters
  const setModel = useCallback((v: string | null) => setSelectors((s) => ({ ...s, model: v })), []);
  const setLayer = useCallback((v: string | null) => setSelectors((s) => ({ ...s, layer: v })), []);
  const setHookType = useCallback((v: string | null) => setSelectors((s) => ({ ...s, hookType: v })), []);
  const setWidth = useCallback((v: string | null) => setSelectors((s) => ({ ...s, width: v })), []);

  return {
    selectors: effectiveSelectors,
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
  };
}
