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
  /** Binary categorical targets only: applied value->0/1 mapping (e.g. {safe: 0, unsafe: 1}). */
  targetMapping?: Record<string, number> | null;
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

/** Derived |residual| field name, or null when the probe has no residuals. */
export function probeAbsErrorField(probe: ProbeInfo): string | null {
  return probe.residualField ? probe.residualField.replace(/_residual$/, '_abserr') : null;
}

/** Derived confusion-category field name (logreg probes only). */
export function probeConfusionField(probe: ProbeInfo): string | null {
  return probe.kind === 'logreg' ? probe.scoreField.replace(/_score$/, '_confusion') : null;
}

/**
 * Build a metadata → binary class (0/1) resolver for a logreg probe's target,
 * mirroring the backend's mapping: the recorded targetMapping for text
 * targets, otherwise the larger of the field's two distinct numeric values
 * maps to 1 (`_binarize_for_logreg`). Returns null when the target is not
 * resolvable as binary (confusion categories are then unavailable).
 */
export function buildBinaryActualResolver(
  itemMetadata: Record<string, unknown>[],
  targetField: string,
  targetMapping: Record<string, number> | null,
): ((meta: Record<string, unknown>) => 0 | 1 | null) | null {
  if (targetMapping) {
    return (meta) => {
      const v = meta[targetField];
      if (v === null || v === undefined) return null;
      const mapped = targetMapping[String(v)];
      if (mapped === undefined) return null;
      return mapped >= 0.5 ? 1 : 0;
    };
  }
  const distinct = new Set<number>();
  for (const meta of itemMetadata) {
    const n = coerceFiniteNumber(meta[targetField]);
    if (n !== null) {
      distinct.add(n);
      if (distinct.size > 2) return null;
    }
  }
  if (distinct.size !== 2) return null;
  const [low, high] = [...distinct].sort((a, b) => a - b);
  return (meta) => {
    const n = coerceFiniteNumber(meta[targetField]);
    if (n === high) return 1;
    if (n === low) return 0;
    return null;
  };
}

/**
 * Numeric coercion matching the backend's DuckDB TRY_CAST: numbers pass
 * through, numeric strings ("3", "7.5") coerce, everything else is null.
 */
function coerceFiniteNumber(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function confusionCategory(predPositive: boolean, actual: 0 | 1): string {
  if (predPositive) return actual === 1 ? 'TP' : 'FP';
  return actual === 0 ? 'TN' : 'FN';
}

/**
 * Merge per-item probe scores/residuals into item metadata.
 *
 * Returns a fresh array of fresh objects (originals untouched). Score item
 * ids missing from `ids` are skipped; null residuals are left absent so the
 * numeric coloring path treats them as missing values. Two derived error
 * fields are added alongside: |residual| (kinds with residuals) and the
 * TP/TN/FP/FN confusion category (logreg, 0.5 threshold, where the actual
 * class is resolvable).
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
    const absErrorField = probeAbsErrorField(probe);
    const confusionField = probeConfusionField(probe);
    const resolveActual = confusionField
      ? buildBinaryActualResolver(itemMetadata, probe.targetField, probe.targetMapping ?? null)
      : null;
    for (let j = 0; j < itemIds.length; j++) {
      const idx = idToIndex.get(itemIds[j]);
      if (idx === undefined) continue;
      merged[idx][probe.scoreField] = values[j];
      if (probe.residualField && residuals) {
        const r = residuals[j];
        if (r !== null && r !== undefined) {
          merged[idx][probe.residualField] = r;
          if (absErrorField) merged[idx][absErrorField] = Math.abs(r);
        }
      }
      if (confusionField && resolveActual) {
        const actual = resolveActual(itemMetadata[idx]);
        if (actual !== null) {
          merged[idx][confusionField] = confusionCategory(values[j] >= 0.5, actual);
        }
      }
    }
  }
  return merged;
}

/**
 * Whether a Color By field option is a valid probe target.
 *
 * Numeric fields (except probe-derived ones) train directly; string fields
 * with exactly two distinct values train as binary 0/1 targets (the backend
 * maps them alphabetically and reports the mapping on the probe).
 */
export function isProbeTargetOption(option: ColorFieldOption): boolean {
  if (option.field.startsWith('probe_')) return false;
  if (option.valueType === 'numeric') return true;
  return option.valueType === 'string' && option.uniqueCount === 2;
}

/**
 * Next target-field selection for the probe form when colorByField changes.
 *
 * The form defaults to following the active color field. When a probe is
 * fitted the map auto-recolors to that probe's `probe_*` score field; if the
 * user was in follow mode (no explicit pick) the followed field would then
 * resolve to nothing and disable the Fit button. In that case we pin the
 * field that was just probed (`lastResolvedTarget`) so another kind can be
 * fitted on it. Coloring by any real field clears back to follow mode.
 */
export function resolveProbeTargetSelection(
  colorByField: string | null,
  prevSelected: string | null,
  lastResolvedTarget: string | null,
): string | null {
  if (colorByField?.startsWith('probe_')) {
    return prevSelected ?? lastResolvedTarget;
  }
  return null;
}

/** Human-readable "safe → 0 · unsafe → 1" line for a probe's target mapping. */
export function formatTargetMapping(
  mapping: Record<string, number> | null | undefined,
): string | null {
  if (!mapping) return null;
  return Object.entries(mapping)
    .sort((a, b) => a[1] - b[1])
    .map(([value, num]) => `${value} → ${num}`)
    .join(' · ');
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
 * One sequential numeric option per probe score; residual and |error|
 * options are added only when the probe has a residual field with at least
 * one non-null value; a categorical confusion option is added for logreg
 * probes whose actual class is resolvable from `itemMetadata`.
 * (ColorFieldOption's recommendedScale cannot express 'diverging'; the
 * residual button in ProbeSection applies the diverging scale explicitly.)
 */
export function buildProbeFieldOptions(
  probesWithScores: ProbeWithScores[],
  itemMetadata?: Record<string, unknown>[],
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
      const absErrorField = probeAbsErrorField(probe);
      if (absErrorField) {
        const absRange = numericRange(residualValues.map(Math.abs));
        options.push({
          field: absErrorField,
          displayName: `${probe.targetField} · ${probe.kind} |error|`,
          valueType: 'numeric',
          uniqueCount: residualValues.length,
          recommendedScale: 'sequential',
          min: absRange.min,
          max: absRange.max,
        });
      }
    }

    const confusionField = probeConfusionField(probe);
    if (
      confusionField &&
      itemMetadata &&
      buildBinaryActualResolver(itemMetadata, probe.targetField, probe.targetMapping ?? null)
    ) {
      options.push({
        field: confusionField,
        displayName: `${probe.targetField} · ${probe.kind} confusion`,
        valueType: 'string',
        uniqueCount: 4,
        recommendedScale: 'categorical',
      });
    }
  }
  return options;
}
