/**
 * Probe field utilities.
 *
 * Probe scores arrive from the backend as parallel arrays keyed by item id
 * (probeScores query). These helpers merge them into a fresh itemMetadata
 * array (so points rebuild — plots read point.metadata by reference) and
 * build the derived ColorFieldOption entries for the Color By dropdown.
 * Probe fields are deliberately NOT baked into the persisted field_analysis
 * cache; they live in this separate client-side layer.
 */

import type { ColorFieldOption } from './fieldAnalysis';

export interface ProbeInfo {
  targetField: string;
  kind: string;
  scoreField: string;
  residualField: string | null;
  metrics: Record<string, number | null> | null;
  nTrain: number;
  nVal: number;
  createdAt: string;
}

export interface ProbeScoresData {
  itemIds: string[];
  scores: number[];
  residuals: (number | null)[] | null;
}

export interface ProbeWithScores {
  probe: ProbeInfo;
  scores: ProbeScoresData;
}

/**
 * Merge per-item probe scores/residuals into item metadata.
 *
 * Returns a fresh array of fresh objects (originals untouched). Score item
 * ids missing from `ids` are skipped; null residuals are left absent so the
 * numeric coloring path treats them as missing values.
 */
export function mergeProbeScores(
  itemMetadata: Record<string, unknown>[],
  ids: string[],
  probesWithScores: ProbeWithScores[],
): Record<string, unknown>[] {
  const merged = itemMetadata.map((meta) => ({ ...meta }));
  const idToIndex = new Map(ids.map((id, i) => [id, i]));

  for (const { probe, scores } of probesWithScores) {
    const { itemIds, scores: values, residuals } = scores;
    for (let j = 0; j < itemIds.length; j++) {
      const idx = idToIndex.get(itemIds[j]);
      if (idx === undefined) continue;
      merged[idx][probe.scoreField] = values[j];
      if (probe.residualField && residuals) {
        const r = residuals[j];
        if (r !== null && r !== undefined) {
          merged[idx][probe.residualField] = r;
        }
      }
    }
  }
  return merged;
}

function numericRange(values: number[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

/**
 * Build Color By dropdown options for trained probes.
 *
 * One sequential numeric option per probe score; a residual option is added
 * only when the probe has a residual field with at least one non-null value.
 * (ColorFieldOption's recommendedScale cannot express 'diverging'; the
 * residual button in ProbeSection applies the diverging scale explicitly.)
 */
export function buildProbeFieldOptions(
  probesWithScores: ProbeWithScores[],
): ColorFieldOption[] {
  const options: ColorFieldOption[] = [];
  for (const { probe, scores } of probesWithScores) {
    if (scores.scores.length === 0) continue;
    const { min, max } = numericRange(scores.scores);
    options.push({
      field: probe.scoreField,
      displayName: `${probe.targetField} · ${probe.kind} score`,
      valueType: 'numeric',
      uniqueCount: scores.scores.length,
      recommendedScale: 'sequential',
      min,
      max,
    });

    const residualValues = (scores.residuals ?? []).filter(
      (r): r is number => r !== null && r !== undefined,
    );
    if (probe.residualField && residualValues.length > 0) {
      const residualRange = numericRange(residualValues);
      options.push({
        field: probe.residualField,
        displayName: `${probe.targetField} · ${probe.kind} residual`,
        valueType: 'numeric',
        uniqueCount: residualValues.length,
        recommendedScale: 'sequential',
        min: residualRange.min,
        max: residualRange.max,
      });
    }
  }
  return options;
}
