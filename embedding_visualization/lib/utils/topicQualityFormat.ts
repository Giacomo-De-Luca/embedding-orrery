/**
 * Pure helpers for the topic-quality scoring UI (TopicQualitySection).
 *
 * The backend returns a free-form metrics blob (see
 * `backend/services/topic_quality_service.py`); stored scores arrive keyed by
 * level: `{ topic: {...}, subtopic: {...} }`. These helpers turn that blob into
 * pill options, badge entries, and interpretation tooltips.
 */

export type QualityLevel = 'topic' | 'subtopic';

/** Loose shape of one level's metrics blob from the backend. */
export type QualityMetricsBlob = Record<string, unknown>;

export interface MetricOption {
  /** Backend selection name (must match evaluation METRIC_NAMES). */
  name: string;
  /** Short pill label. */
  label: string;
  /** Cost hint appended to the pill (e.g. "slow"). */
  hint?: string;
  /** Always-disabled pills (DBCV needs the live fitted model). */
  disabled?: boolean;
  /** Native tooltip for the pill. */
  title: string;
}

export const METRIC_OPTIONS: MetricOption[] = [
  {
    name: 'silhouette',
    label: 'Silhouette',
    title:
      'Cluster separation in the space the clustering ran in (-1..1, higher better).\n' +
      '> 0.5 strong · 0.25–0.5 reasonable · ~0 overlapping · < 0 mixed.',
  },
  {
    name: 'diversity',
    label: 'Diversity',
    title:
      'Unique words ÷ total words across topics\' top keywords (0–1).\n' +
      'Higher = topics use distinct vocabulary; low values suggest redundant topics.',
  },
  {
    name: 'coherence_cv',
    label: 'C_v',
    hint: 'slow',
    title:
      'C_v keyword coherence (0–1, higher better) — best correlation with human judgment.\n' +
      '> 0.55 good · 0.4–0.55 acceptable · < 0.4 weak. Slow on large collections.',
  },
  {
    name: 'coherence_umass',
    label: 'U_Mass',
    title:
      'U_Mass keyword co-occurrence (≤ 0, closer to 0 better).\n' +
      'Only meaningful relative to other runs on the same collection.',
  },
  {
    name: 'dbcv',
    label: 'DBCV',
    disabled: true,
    title:
      'Density-based cluster validity — the HDBSCAN-native metric.\n' +
      'Requires the live fitted model, so it is unavailable when scoring stored topics.',
  },
];

export const DEFAULT_SELECTED_METRICS = ['silhouette', 'diversity', 'coherence_cv', 'coherence_umass'];

/** Metric result keys → short badge labels (order = display order). */
const BADGE_KEYS: [string, string][] = [
  ['dbcv', 'DBCV'],
  ['silhouette_cluster_space', 'Sil'],
  ['topic_diversity', 'Div'],
  ['coherence_cv', 'C_v'],
  ['coherence_umass', 'U_Mass'],
];

export const fmtMetric = (value: unknown): string =>
  typeof value === 'number' && Number.isFinite(value)
    ? Math.abs(value) >= 10
      ? value.toFixed(1)
      : value.toFixed(2)
    : '—';

export interface QualityBadgeEntry {
  key: string;
  text: string; // e.g. "Sil 0.47"
}

/** Badge entries for the metric values present (and non-null) in a blob. */
export function qualityBadges(metrics: QualityMetricsBlob | null | undefined): QualityBadgeEntry[] {
  if (!metrics) return [];
  return BADGE_KEYS.filter(([key]) => metrics[key] !== undefined && metrics[key] !== null).map(
    ([key, label]) => ({ key, text: `${label} ${fmtMetric(metrics[key])}` })
  );
}

/** Multi-line native tooltip: every metric with its interpretation + meta. */
export function qualityTitle(metrics: QualityMetricsBlob | null | undefined): string {
  if (!metrics) return 'No quality metrics computed yet';
  const lines: string[] = [];
  const push = (key: string, label: string, guide: string) => {
    if (metrics[key] !== undefined) lines.push(`${label}: ${fmtMetric(metrics[key])}  ${guide}`);
  };
  push('dbcv', 'DBCV', '(density validity, -1..1, higher better)');
  push('silhouette_cluster_space', 'Silhouette', '(cluster space, -1..1; >0.5 strong, ~0 overlapping)');
  push('topic_diversity', 'Diversity', '(unique keyword ratio, 0-1, higher better)');
  push('coherence_cv', 'Coherence C_v', '(0-1; >0.55 good, <0.4 weak)');
  push('coherence_umass', 'Coherence U_Mass', '(<=0, closer to 0 better; relative measure)');

  const meta: string[] = [];
  if (typeof metrics.num_clusters_evaluated === 'number') {
    meta.push(`${metrics.num_clusters_evaluated} clusters`);
  }
  if (typeof metrics.cluster_space === 'string') meta.push(`space: ${metrics.cluster_space}`);
  if (metrics.sampled === true) meta.push('silhouette subsampled');
  if (typeof metrics.computed_at === 'string') meta.push(`computed ${metrics.computed_at}`);
  if (meta.length) lines.push(meta.join(' · '));
  return lines.join('\n') || 'No quality metrics computed yet';
}

/** Extract one level's metrics from the stored `{topic, subtopic}` blob. */
export function pickLevelMetrics(
  stored: unknown,
  level: QualityLevel
): QualityMetricsBlob | null {
  if (!stored || typeof stored !== 'object') return null;
  const blob = (stored as Record<string, unknown>)[level];
  if (!blob || typeof blob !== 'object') return null;
  return blob as QualityMetricsBlob;
}
