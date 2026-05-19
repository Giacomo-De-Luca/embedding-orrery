import type { SteeringFeature } from '../types/types';

/**
 * Deterministic hash string for a steering bundle, used as the seed for
 * jdenticon identicon rendering. Returns null when no features are
 * configured — callers should render their Sparkles fallback in that case.
 *
 * Sorting makes the hash order-independent: "poetry" + "philosophy" yields
 * the same identicon as "philosophy" + "poetry". Missing labels fall back
 * to `idx_<n>` so raw (unlabeled) features still hash deterministically.
 */
export function steeringIdenticonHash(features: SteeringFeature[]): string | null {
  if (features.length === 0) return null;
  return features
    .map((f) => f.label?.trim() || `idx_${f.featureIndex}`)
    .sort()
    .join('_');
}
