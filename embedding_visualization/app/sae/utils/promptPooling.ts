/**
 * Pure helpers for the prompt-activation search: attach SAE identity to the
 * per-layer results returned by `runPromptActivations`, then pool per-token
 * activations into one ranked feature list across all hooked SAEs.
 *
 * Features are keyed by `saeId::featureIndex`, so the same feature index
 * appearing in two SAEs (different layers or widths) stays distinct.
 */

import type { LayerActivationsResult } from '@/lib/graphql/mutations';
import { buildSaeId, parseSaeId } from '@/lib/utils/saeCollections';
import type { SemanticFeatureResult } from '../components/FeatureSearchResults';

export type PromptPooling = 'max' | 'mean' | 'last';

/** Ranked-list cap — topK=0 prompt runs return every nonzero feature per
 * token, which across several SAEs can reach thousands of rows; the table
 * is unvirtualized and only the top of the ranking is meaningful. */
export const MAX_POOLED_ROWS = 200;

export interface SaeLayerActivations extends LayerActivationsResult {
  modelId: string;
  saeId: string;
}

export interface PooledFeatureRow extends SemanticFeatureResult {
  activation: number;
  modelId: string;
  saeId: string;
}

/**
 * Tag each layer result with its (modelId, saeId). The saeId is resolved by
 * matching (layer, width) back to the selection pairs — exact DB ids — with
 * the derived gemmascope id as fallback. The prompt path hooks RESID_POST
 * only, so (layer, width) identifies the SAE.
 */
export function attachSaeIdentity(
  layers: LayerActivationsResult[],
  modelId: string,
  pairs: Array<{ modelId: string; saeId: string }>,
): SaeLayerActivations[] {
  const pairByKey = new Map<string, string>();
  for (const pair of pairs) {
    const parsed = parseSaeId(pair.saeId);
    pairByKey.set(`${parsed.layerIndex}::${parsed.width}`, pair.saeId);
  }
  return layers.map((layer) => ({
    ...layer,
    modelId,
    saeId:
      pairByKey.get(`${layer.layer}::${layer.width}`) ??
      buildSaeId(layer.layer, 'RESID_POST', layer.width),
  }));
}

/**
 * Pool per-token activations into one globally ranked feature list.
 *
 * - `max` / `mean`: pooled over the tokens where the feature fired.
 * - `last`: only the final token's features.
 * - Features with density above `maxDensity` are dropped (null passes).
 * - Rows sort by raw pooled activation descending; `similarity` is the
 *   activation normalized to the top row (for the shared results table).
 */
export function poolPromptFeatures(
  layers: SaeLayerActivations[],
  pooling: PromptPooling,
  maxDensity: number,
): PooledFeatureRow[] {
  interface Entry {
    featureIndex: number;
    activation: number;
    label: string;
    density: number | null;
    modelId: string;
    saeId: string;
  }
  const featureMap = new Map<string, Entry>();

  for (const layer of layers) {
    const tokens = layer.tokens;
    if (tokens.length === 0) continue;

    if (pooling === 'last') {
      const lastToken = tokens[tokens.length - 1];
      for (const feat of lastToken.features) {
        featureMap.set(`${layer.saeId}::${feat.index}`, {
          featureIndex: feat.index,
          activation: feat.activation,
          label: feat.label,
          density: feat.density,
          modelId: layer.modelId,
          saeId: layer.saeId,
        });
      }
    } else {
      // Accumulate per-feature across all tokens of this layer
      const accumulator = new Map<
        number,
        { sum: number; count: number; max: number; label: string; density: number | null }
      >();
      for (const token of tokens) {
        for (const feat of token.features) {
          const existing = accumulator.get(feat.index);
          if (existing) {
            existing.sum += feat.activation;
            existing.count += 1;
            if (feat.activation > existing.max) existing.max = feat.activation;
          } else {
            accumulator.set(feat.index, {
              sum: feat.activation,
              count: 1,
              max: feat.activation,
              label: feat.label,
              density: feat.density,
            });
          }
        }
      }
      for (const [idx, { sum, count, max, label, density }] of accumulator) {
        featureMap.set(`${layer.saeId}::${idx}`, {
          featureIndex: idx,
          activation: pooling === 'mean' ? sum / count : max,
          label,
          density,
          modelId: layer.modelId,
          saeId: layer.saeId,
        });
      }
    }
  }

  // Filter by density threshold (exclude ultra-common features)
  const filtered = [...featureMap.values()].filter(
    ({ density }) => density === null || density <= maxDensity,
  );

  filtered.sort((a, b) => b.activation - a.activation);
  if (filtered.length === 0) return [];
  const maxAct = filtered[0].activation;
  return filtered.slice(0, MAX_POOLED_ROWS).map((entry) => ({
    featureIndex: entry.featureIndex,
    label: entry.label || null,
    density: entry.density,
    activation: entry.activation,
    similarity: maxAct > 0 ? entry.activation / maxAct : 1,
    modelId: entry.modelId,
    saeId: entry.saeId,
  }));
}
