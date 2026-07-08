/**
 * Steering-strength hint from per-layer residual-stream norms.
 *
 * Additive steering adds `strength · v` to the residual stream `h` at a layer;
 * how disruptive that is depends on the dimensionless ratio
 *
 *     rho = |strength| · ‖v‖ / ‖h_L‖
 *
 * `‖h_L‖` (the residual-stream norm at layer L) grows with depth, so a fixed
 * coefficient is far stronger at an early layer than a late one. The
 * `residualNorms.json` table — generated offline by
 * `interpretability_backend/scripts/profile_residual_norms.py` — supplies
 * `‖h_L‖` per model/layer. Gemma-scope SAE decoder rows are unit-norm, so for
 * SAE features `‖v‖ = 1` and this table is a complete hint; direction presets
 * carry their own `vecNorm` under `directions`.
 *
 * The table ships empty until the profiler is run on a machine with enough
 * VRAM for the target model; every lookup degrades to `undefined`/`null` so
 * the UI simply hides the hint when data is absent.
 *
 * Pure math (`computeRho`, `suggestedStrength`, `strengthBand`) is separated
 * from the JSON lookup and unit-tested without the asset.
 */

import rawTable from './residualNorms.json';

export interface LayerNorm {
  layer: number;
  median: number;
  p25: number;
  p75: number;
  mean: number;
  count: number;
}

export interface DirectionNorm {
  layer: number;
  vecNorm: number;
}

export interface ModelResidualNorms {
  checkpoint: string;
  dModel: number;
  nLayers: number;
  promptCount: number;
  droppedBos: boolean;
  generatedAt: string;
  layers: LayerNorm[];
  directions: Record<string, DirectionNorm>;
}

export type ResidualNormsTable = Record<string, ModelResidualNorms>;

const TABLE = rawTable as ResidualNormsTable;

// rho band thresholds — model-agnostic because rho is already normalized.
export const RHO_SUBTLE_MAX = 0.08;
export const RHO_STRONG_MIN = 0.25;
// Target rho used when suggesting a starting strength (a moderate nudge).
export const RHO_RECOMMENDED = 0.15;

export type StrengthBand = 'subtle' | 'medium' | 'strong';

/** rho = |strength| · ‖v‖ / ‖h_L‖. Returns 0 for a non-positive residual norm. */
export function computeRho(strength: number, residualNorm: number, vecNorm = 1): number {
  if (!(residualNorm > 0)) return 0;
  return (Math.abs(strength) * vecNorm) / residualNorm;
}

/** Inverse of computeRho: the strength that yields `rhoTarget`. */
export function suggestedStrength(rhoTarget: number, residualNorm: number, vecNorm = 1): number {
  if (!(vecNorm > 0)) return 0;
  return (rhoTarget * residualNorm) / vecNorm;
}

export function strengthBand(rho: number): StrengthBand {
  if (rho < RHO_SUBTLE_MAX) return 'subtle';
  if (rho >= RHO_STRONG_MIN) return 'strong';
  return 'medium';
}

/** Snap a raw strength to the slider grid, then clamp to its range. */
export function snapStrengthToSlider(
  raw: number,
  bounds: { min: number; max: number; step: number },
): number {
  const snapped = Math.round(raw / bounds.step) * bounds.step;
  return Math.min(bounds.max, Math.max(bounds.min, snapped));
}

export function modelResidualNorms(
  modelId: string | null | undefined,
  table: ResidualNormsTable = TABLE,
): ModelResidualNorms | undefined {
  if (!modelId) return undefined;
  return table[modelId];
}

/** Median `‖h_L‖` for a model/layer, or undefined if unavailable. */
export function layerMedianNorm(
  modelId: string | null | undefined,
  layerIndex: number,
  table: ResidualNormsTable = TABLE,
): number | undefined {
  const model = modelResidualNorms(modelId, table);
  return model?.layers.find((l) => l.layer === layerIndex)?.median;
}

export function directionNorm(
  modelId: string | null | undefined,
  name: string,
  table: ResidualNormsTable = TABLE,
): DirectionNorm | undefined {
  return modelResidualNorms(modelId, table)?.directions[name];
}

export interface SteeringStrengthHint {
  /** |strength|·‖v‖ / ‖h_L‖ for the current strength. */
  rho: number;
  band: StrengthBand;
  /** Residual-stream norm ‖h_L‖ at the applied layer. */
  residualNorm: number;
  /** Direction norm ‖v‖ (1 for SAE features). */
  vecNorm: number;
  /** Layer the op is applied at (direction layer is authoritative from the table). */
  layer: number;
  /** Raw strength for a moderate (RHO_RECOMMENDED) nudge — round to the slider step before use. */
  suggestedStrength: number;
}

/**
 * Compute the strength hint for one steering row, or `null` when the residual
 * norms for this model/layer are unavailable (so the caller hides the hint).
 *
 * SAE features use `layerIndex` and `‖v‖ = 1`. Direction presets look up the
 * registered `{layer, vecNorm}` (the direction's application layer, which is
 * authoritative server-side, may differ from the row's display `layerIndex`).
 */
export function steeringHint(
  params: {
    modelId: string | null | undefined;
    layerIndex: number;
    strength: number;
    directionName?: string | null;
  },
  table: ResidualNormsTable = TABLE,
): SteeringStrengthHint | null {
  const { modelId, layerIndex, strength, directionName } = params;
  if (!modelId) return null;

  let layer = layerIndex;
  let vecNorm = 1;
  if (directionName) {
    const dir = directionNorm(modelId, directionName, table);
    if (!dir) return null;
    layer = dir.layer;
    vecNorm = dir.vecNorm;
  }

  const residualNorm = layerMedianNorm(modelId, layer, table);
  if (residualNorm === undefined) return null;

  const rho = computeRho(strength, residualNorm, vecNorm);
  return {
    rho,
    band: strengthBand(rho),
    residualNorm,
    vecNorm,
    layer,
    suggestedStrength: suggestedStrength(RHO_RECOMMENDED, residualNorm, vecNorm),
  };
}
