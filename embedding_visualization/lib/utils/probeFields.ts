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

import { CategoryKey } from './categoryKey';
import { CORRECTNESS_LABELS, type CategoryColorAlias } from './categoryColors';
import type { ColorFieldOption } from './fieldAnalysis';
import { isClassificationKind, probeKindLabel } from './probeParams';
import { MAX_CATEGORICAL_TARGET_VALUES, canonicalProbeLabel } from './probeTargets';

export type ProbeTask = 'regression' | 'binary' | 'multiclass';

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
  /** Derived server-side for probes trained before multiclass support. */
  task?: ProbeTask;
  /** Classifiers: kept classes in class-index order (null for legacy binary probes). */
  classes?: string[] | null;
  /** Classifiers: classes excluded by the min-class threshold -> item count. */
  excludedClasses?: Record<string, number> | null;
  minClassFraction?: number | null;
}

export interface ProbeScoresData {
  itemIds: string[];
  scores: number[];
  residuals: (number | null)[] | null;
  /** Multiclass probes: predicted class index (into ProbeInfo.classes). */
  predicted?: (number | null)[] | null;
}

/** One class's per-item probabilities (probeClassProbabilities query). */
export interface ProbeClassProbabilitiesData {
  itemIds: string[];
  probabilities: (number | null)[];
}

export interface ProbeWithScores {
  probe: ProbeInfo;
  scores: ProbeScoresData;
  /** Lazily loaded class probabilities, keyed by class index. */
  classProbabilities?: Record<number, ProbeClassProbabilitiesData>;
}

/** The probe's task, inferred for responses that predate the `task` field. */
export function probeTask(probe: ProbeInfo): ProbeTask {
  if (probe.task) return probe.task;
  return isClassificationKind(probe.kind) ? 'binary' : 'regression';
}

/**
 * Names of every metadata field derived from one probe. All are built from
 * the backend's score field (whose target part is already sanitized), so
 * matching is by exact name — never by prefix, which would confuse probes on
 * similarly named fields.
 */
export class ProbeFieldNames {
  private readonly base: string;
  private readonly task: ProbeTask;

  constructor(private readonly probe: ProbeInfo) {
    this.base = probe.scoreField.replace(/_score$/, '');
    this.task = probeTask(probe);
  }

  /** Regression prediction, binary P(class 1), or multiclass confidence. */
  get score(): string {
    return this.probe.scoreField;
  }

  get residual(): string | null {
    return this.probe.residualField;
  }

  /** |residual| (kinds with residuals). */
  get absError(): string | null {
    return this.probe.residualField ? `${this.base}_abserr` : null;
  }

  /** TP/TN/FP/FN at threshold 0.5 (binary classifiers). */
  get confusion(): string | null {
    return this.task === 'binary' ? `${this.base}_confusion` : null;
  }

  /** Predicted class label (multiclass). */
  get predicted(): string | null {
    return this.task === 'multiclass' ? `${this.base}_pred` : null;
  }

  /** Correct / Wrong / Excluded against the actual class (multiclass). */
  get correct(): string | null {
    return this.task === 'multiclass' ? `${this.base}_correct` : null;
  }

  /** P(class `index`) field (multiclass, loaded on demand). */
  classProbability(index: number): string {
    return `${this.base}_p${index}`;
  }

  /** Class index of one of this probe's P(class) fields, else null. */
  classIndexOf(field: string): number | null {
    if (this.task !== 'multiclass' || !field.startsWith(`${this.base}_p`)) return null;
    const suffix = field.slice(this.base.length + 2);
    return /^\d+$/.test(suffix) ? Number(suffix) : null;
  }

  /** The fixed derived fields (P(class) fields are matched by classIndexOf). */
  get fixed(): string[] {
    return [
      this.score,
      this.residual,
      this.absError,
      this.confusion,
      this.predicted,
      this.correct,
    ].filter((f): f is string => f !== null);
  }

  matches(field: string): boolean {
    return this.fixed.includes(field) || this.classIndexOf(field) !== null;
  }
}

/** A recolor queued until its field's values have loaded. */
export interface PendingRecolor {
  field: string;
  scale: 'categorical' | 'sequential';
  /** colorByField when the recolor was requested. */
  fromField: string | null;
}

/**
 * What to do with a pending recolor: apply it once its field is available,
 * keep waiting, or drop it because the user picked another color field in
 * the meantime (a late-landing fetch must not override that choice).
 */
export function pendingRecolorAction(
  pending: PendingRecolor,
  colorByField: string | null,
  availableFields: readonly { field: string }[],
): 'apply' | 'wait' | 'drop' {
  if (colorByField !== pending.fromField) return 'drop';
  return availableFields.some((o) => o.field === pending.field) ? 'apply' : 'wait';
}

/** The probe + class index behind a P(class) colour field, if any. */
export function parseProbeClassField(
  field: string,
  probes: readonly ProbeInfo[],
): { probe: ProbeInfo; classIndex: number } | null {
  for (const probe of probes) {
    const classIndex = new ProbeFieldNames(probe).classIndexOf(field);
    if (classIndex !== null && classIndex < (probe.classes?.length ?? 0)) {
      return { probe, classIndex };
    }
  }
  return null;
}

/**
 * Build a metadata → binary class (0/1) resolver for a legacy logreg probe's
 * target, mirroring the backend's old mapping: the recorded targetMapping for
 * text targets, otherwise the larger of the field's two distinct numeric
 * values maps to 1. Returns null when the target is not resolvable as binary
 * (confusion categories are then unavailable).
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
 * Metadata → actual class index of a classifier probe's target, or null when
 * unlabelled / outside the kept classes. Uses the classes the backend recorded
 * (every probe trained since multiclass support); legacy binary probes fall
 * back to {@link buildBinaryActualResolver}.
 */
export function buildClassActualResolver(
  probe: ProbeInfo,
  itemMetadata: Record<string, unknown>[],
): ((meta: Record<string, unknown>) => number | null) | null {
  if (probe.classes && probe.classes.length > 0) {
    const index = new Map(probe.classes.map((label, i) => [label, i]));
    return (meta) => {
      const label = canonicalProbeLabel(meta[probe.targetField]);
      return label === null ? null : (index.get(label) ?? null);
    };
  }
  return buildBinaryActualResolver(itemMetadata, probe.targetField, probe.targetMapping ?? null);
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

function confusionCategory(predPositive: boolean, actual: number): string {
  if (predPositive) return actual === 1 ? 'TP' : 'FP';
  return actual === 0 ? 'TN' : 'FN';
}

/** Correct / Wrong against a kept class; Excluded for any other label. */
function correctnessCategory(predicted: string, label: string, classes: Set<string>): string {
  if (!classes.has(label)) return CORRECTNESS_LABELS.excluded;
  return predicted === label ? CORRECTNESS_LABELS.correct : CORRECTNESS_LABELS.wrong;
}

/**
 * Merge per-item probe outputs into item metadata.
 *
 * Returns a fresh array of fresh objects (originals untouched). Score item
 * ids missing from `ids` are skipped; null values are left absent so the
 * coloring paths treat them as missing. Derived fields per probe task:
 * regression → |residual|; binary → TP/TN/FP/FN confusion at 0.5;
 * multiclass → predicted label, Correct/Wrong/Excluded against the actual
 * class, and any loaded P(class) fields.
 */
export function mergeProbeScores(
  itemMetadata: Record<string, unknown>[],
  ids: string[],
  probesWithScores: ProbeWithScores[],
): Record<string, unknown>[] {
  const merged = itemMetadata.map((meta) => ({ ...meta }));
  const idToIndex = new Map(ids.map((id, i) => [id, i]));

  for (const { probe, scores, classProbabilities } of probesWithScores) {
    const names = new ProbeFieldNames(probe);
    const { itemIds, scores: values, residuals, predicted } = scores;
    const resolveActual = names.confusion ? buildClassActualResolver(probe, itemMetadata) : null;
    const classes = probe.classes ?? [];
    const classSet = new Set(classes);

    for (let j = 0; j < itemIds.length; j++) {
      const idx = idToIndex.get(itemIds[j]);
      if (idx === undefined) continue;
      const row = merged[idx];
      row[names.score] = values[j];
      if (names.residual && residuals) {
        const r = residuals[j];
        if (r !== null && r !== undefined) {
          row[names.residual] = r;
          if (names.absError) row[names.absError] = Math.abs(r);
        }
      }
      if (names.confusion && resolveActual) {
        const actual = resolveActual(itemMetadata[idx]);
        if (actual !== null) row[names.confusion] = confusionCategory(values[j] >= 0.5, actual);
      }
      const p = predicted?.[j];
      if (names.predicted && names.correct && p !== null && p !== undefined && classes[p]) {
        row[names.predicted] = classes[p];
        const label = canonicalProbeLabel(itemMetadata[idx][probe.targetField]);
        if (label !== null) row[names.correct] = correctnessCategory(classes[p], label, classSet);
      }
    }

    for (const [classIndex, data] of Object.entries(classProbabilities ?? {})) {
      const field = names.classProbability(Number(classIndex));
      data.itemIds.forEach((id, i) => {
        const idx = idToIndex.get(id);
        const value = data.probabilities[i];
        if (idx !== undefined && value !== null && value !== undefined) merged[idx][field] = value;
      });
    }
  }
  return merged;
}

/**
 * Whether a Color By field option is a valid probe target.
 *
 * Numeric fields (except probe-derived ones) train directly; text fields
 * with 2–100 distinct values train as class labels (two values also as 0/1
 * regression targets). Which kinds apply is decided per target by
 * ProbeTargetProfile.
 */
export function isProbeTargetOption(option: ColorFieldOption): boolean {
  if (option.field.startsWith('probe_')) return false;
  if (option.valueType === 'numeric') return true;
  return (
    option.valueType === 'string' &&
    option.uniqueCount >= 2 &&
    option.uniqueCount <= MAX_CATEGORICAL_TARGET_VALUES
  );
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

function sequentialOption(field: string, displayName: string, values: number[]): ColorFieldOption {
  return {
    field,
    displayName,
    valueType: 'numeric',
    uniqueCount: values.length,
    recommendedScale: 'sequential',
    ...numericRange(values),
  };
}

function categoricalOption(field: string, displayName: string, uniqueCount: number): ColorFieldOption {
  return { field, displayName, valueType: 'string', uniqueCount, recommendedScale: 'categorical' };
}

/**
 * Build Color By dropdown options for trained probes.
 *
 * One sequential option per probe score (a multiclass probe's score is its
 * confidence); residual and |error| options when the probe has non-null
 * residuals; a confusion option for binary classifiers whose actual class is
 * resolvable from `itemMetadata`; predicted-class and correctness options
 * for multiclass probes, plus a P(class) option per loaded class.
 * (ColorFieldOption's recommendedScale cannot express 'diverging'; the
 * residual button in ProbeSection applies the diverging scale explicitly.)
 */
export function buildProbeFieldOptions(
  probesWithScores: ProbeWithScores[],
  itemMetadata?: Record<string, unknown>[],
): ColorFieldOption[] {
  const options: ColorFieldOption[] = [];
  for (const { probe, scores, classProbabilities } of probesWithScores) {
    if (scores.scores.length === 0) continue;
    const names = new ProbeFieldNames(probe);
    const prefix = `${probe.targetField} · ${probeKindLabel(probe.kind)}`;
    const scoreLabel = probeTask(probe) === 'multiclass' ? 'confidence' : 'score';
    options.push(sequentialOption(names.score, `${prefix} ${scoreLabel}`, scores.scores));

    const residualValues = (scores.residuals ?? []).filter(
      (r): r is number => r !== null && r !== undefined,
    );
    if (names.residual && residualValues.length > 0) {
      options.push(sequentialOption(names.residual, `${prefix} residual`, residualValues));
      if (names.absError) {
        options.push(
          sequentialOption(names.absError, `${prefix} |error|`, residualValues.map(Math.abs)),
        );
      }
    }

    if (names.confusion && itemMetadata && buildClassActualResolver(probe, itemMetadata)) {
      options.push(categoricalOption(names.confusion, `${prefix} confusion`, 4));
    }

    const classes = probe.classes ?? [];
    if (names.predicted && names.correct && scores.predicted) {
      options.push(categoricalOption(names.predicted, `${prefix} predicted`, classes.length));
      if (itemMetadata) options.push(categoricalOption(names.correct, `${prefix} correct`, 3));
    }
    for (const [classIndex, data] of Object.entries(classProbabilities ?? {})) {
      const values = data.probabilities.filter((v): v is number => v !== null && v !== undefined);
      if (values.length === 0) continue;
      const index = Number(classIndex);
      options.push(
        sequentialOption(
          names.classProbability(index),
          `${prefix} P(${classes[index] ?? index})`,
          values,
        ),
      );
    }
  }
  return options;
}

/** Sorted category keys of a field, exactly as the colour pipeline groups them. */
function categoryDomain(itemMetadata: Record<string, unknown>[], field: string): string[] {
  return CategoryKey.sort(new Set(itemMetadata.map((meta) => CategoryKey.of(meta[field]))));
}

/**
 * Colour aliases for multiclass predicted-class fields: each one takes the
 * palette slots of its target field's full value domain, so a predicted
 * label is drawn in the same colour as that label in the target field — even
 * when some labels are never predicted.
 */
export function buildProbeColorAliases(
  probes: readonly ProbeInfo[],
  itemMetadata: Record<string, unknown>[],
): Record<string, CategoryColorAlias> {
  const aliases: Record<string, CategoryColorAlias> = {};
  const domains = new Map<string, string[]>();
  for (const probe of probes) {
    const predicted = new ProbeFieldNames(probe).predicted;
    if (!predicted) continue;
    const field = probe.targetField;
    if (!domains.has(field)) domains.set(field, categoryDomain(itemMetadata, field));
    aliases[predicted] = { sourceField: field, domain: domains.get(field)! };
  }
  return aliases;
}
