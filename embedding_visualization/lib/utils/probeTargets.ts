/**
 * Probe target profiling: which probe kinds a metadata field can train.
 *
 * A field is profiled from the loaded item metadata (the persisted
 * field_analysis cache carries no integer/label information). Classifier
 * kinds treat values as class labels; the label and min-class threshold
 * rules mirror the backend's `services/probe_targets.py` — keep in sync.
 */
import { CategoryKey } from './categoryKey';
import { isClassificationKind, type ProbeKind, type TrainedProbeKind } from './probeParams';

/** A kept class needs max(floor, ceil(fraction × labelled items)) items. */
export const MIN_CLASS_COUNT_FLOOR = 5;
/** Upper bound on kept classes (after the min-class threshold). */
export const MAX_PROBE_CLASSES = 50;
/** Text fields with more distinct values are not offered as class targets. */
export const MAX_CATEGORICAL_TARGET_VALUES = 100;

// Strict decimal syntax — Number() would also accept "0x1A", " ", "1e3 ".
const NUMERIC_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
const THRESHOLD_EPSILON = 1e-9;

/**
 * Class label of a metadata value, or null when unlabelled. Matches the
 * backend's canonical labels: numbers render as String(n) (3.0 → "3"),
 * strings stay verbatim, empty/absent/non-scalar values are unlabelled.
 */
export function canonicalProbeLabel(raw: unknown): string | null {
  if (CategoryKey.isMissing(raw) || typeof raw === 'object') return null;
  return String(raw);
}

export function classCountThreshold(nLabelled: number, minClassFraction: number): number {
  return Math.max(
    MIN_CLASS_COUNT_FLOOR,
    Math.ceil(minClassFraction * nLabelled - THRESHOLD_EPSILON),
  );
}

export interface ClassPreview {
  threshold: number;
  kept: number;
  excluded: { label: string; count: number }[];
  /** More than MAX_PROBE_CLASSES classes survive the threshold. */
  tooMany: boolean;
  /** Fewer than two classes survive the threshold. */
  tooFew: boolean;
}

/** Label counts of one metadata field and what they allow a probe to train. */
export class ProbeTargetProfile {
  private constructor(
    readonly counts: ReadonlyMap<string, number>,
    readonly numeric: boolean,
    readonly integral: boolean,
  ) {}

  static fromMetadata(
    itemMetadata: readonly Record<string, unknown>[],
    field: string,
  ): ProbeTargetProfile {
    const counts = new Map<string, number>();
    for (const meta of itemMetadata) {
      const label = canonicalProbeLabel(meta[field]);
      if (label !== null) counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    const labels = [...counts.keys()];
    const numeric = labels.length > 0 && labels.every((l) => NUMERIC_RE.test(l));
    const integral = numeric && labels.every((l) => Number.isInteger(Number(l)));
    return new ProbeTargetProfile(counts, numeric, integral);
  }

  get nDistinct(): number {
    return this.counts.size;
  }

  get nLabelled(): number {
    let total = 0;
    for (const n of this.counts.values()) total += n;
    return total;
  }

  /** Regression kinds: numeric fields, or exactly two values (trained as 0/1). */
  get supportsRegression(): boolean {
    return this.nDistinct === 2 || (this.numeric && this.nDistinct > 2);
  }

  /** Classifier kinds: two values, or 3+ text / integer labels (capped). */
  get supportsClassification(): boolean {
    if (this.nDistinct === 2) return true;
    if (this.nDistinct < 2 || this.nDistinct > MAX_CATEGORICAL_TARGET_VALUES) return false;
    return !this.numeric || this.integral;
  }

  /** Only classifiers can train it (text labels with 3+ values). */
  get requiresClassification(): boolean {
    return this.supportsClassification && !this.supportsRegression;
  }

  /** Classes kept/excluded at a min-class fraction (mirrors the backend). */
  preview(minClassFraction: number): ClassPreview {
    const threshold = classCountThreshold(this.nLabelled, minClassFraction);
    const entries = [...this.counts.entries()];
    const excluded =
      entries.length === 2 // binary targets are never thresholded
        ? []
        : entries.filter(([, n]) => n < threshold).map(([label, count]) => ({ label, count }));
    const kept = entries.length - excluded.length;
    return { threshold, kept, excluded, tooMany: kept > MAX_PROBE_CLASSES, tooFew: kept < 2 };
  }
}

/** Whether `kind` can train on the profiled target (no target: everything). */
export function isProbeKindAvailable(kind: ProbeKind, profile: ProbeTargetProfile | null): boolean {
  if (!profile) return true;
  if (kind === 'mlp') return profile.supportsRegression || profile.supportsClassification;
  if (isClassificationKind(kind)) return profile.supportsClassification;
  return profile.supportsRegression;
}

/** `kind` if the target supports it, else ridge (regression) or logistic. */
export function fallbackProbeKind(kind: ProbeKind, profile: ProbeTargetProfile | null): ProbeKind {
  if (isProbeKindAvailable(kind, profile)) return kind;
  return profile?.supportsRegression ? 'ridge' : 'logreg';
}

/** The MLP's Classifier switch: locked by the target, or left to the user. */
export function mlpClassifierMode(
  profile: ProbeTargetProfile | null,
): 'forced-on' | 'forced-off' | 'free' {
  if (!profile) return 'free';
  if (profile.requiresClassification) return 'forced-on';
  if (!profile.supportsClassification) return 'forced-off';
  return 'free';
}

/** The kind sent to the backend: the MLP trains as `mlp_clf` when classifying. */
export function resolveProbeKind(
  kind: ProbeKind,
  mlpClassifier: boolean,
  profile: ProbeTargetProfile | null,
): TrainedProbeKind {
  if (kind !== 'mlp') return kind;
  const mode = mlpClassifierMode(profile);
  if (mode === 'forced-on') return 'mlp_clf';
  if (mode === 'forced-off') return 'mlp';
  return mlpClassifier ? 'mlp_clf' : 'mlp';
}
