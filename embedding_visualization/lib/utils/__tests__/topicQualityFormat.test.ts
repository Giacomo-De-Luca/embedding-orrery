import { describe, expect, it } from 'vitest';
import {
  DEFAULT_SELECTED_METRICS,
  METRIC_OPTIONS,
  fmtMetric,
  pickLevelMetrics,
  qualityBadges,
  qualityTitle,
} from '../topicQualityFormat';

const SAMPLE = {
  dbcv: null,
  silhouette_cluster_space: 0.4709,
  topic_diversity: 0.875,
  coherence_cv: 0.3202,
  coherence_umass: -14.1212,
  num_clusters_evaluated: 16,
  sampled: false,
  cluster_space: 'projection/umap_3d',
  computed_at: '2026-07-06T13:00:17+00:00',
};

describe('fmtMetric', () => {
  it('formats small numbers with 2 decimals and large with 1', () => {
    expect(fmtMetric(0.4709)).toBe('0.47');
    expect(fmtMetric(-14.1212)).toBe('-14.1');
  });

  it('falls back to em dash for non-numbers', () => {
    expect(fmtMetric(null)).toBe('—');
    expect(fmtMetric(undefined)).toBe('—');
    expect(fmtMetric('x')).toBe('—');
    expect(fmtMetric(NaN)).toBe('—');
  });
});

describe('qualityBadges', () => {
  it('emits one badge per present non-null metric, in display order', () => {
    const badges = qualityBadges(SAMPLE);
    // dbcv is null -> skipped
    expect(badges.map((b) => b.text)).toEqual([
      'Sil 0.47',
      'Div 0.88',
      'C_v 0.32',
      'U_Mass -14.1',
    ]);
  });

  it('omits metrics that were not computed at all', () => {
    const badges = qualityBadges({ silhouette_cluster_space: 0.5 });
    expect(badges).toHaveLength(1);
    expect(badges[0].key).toBe('silhouette_cluster_space');
  });

  it('handles null/undefined blobs', () => {
    expect(qualityBadges(null)).toEqual([]);
    expect(qualityBadges(undefined)).toEqual([]);
  });
});

describe('qualityTitle', () => {
  it('includes interpretation guidance and meta line', () => {
    const title = qualityTitle(SAMPLE);
    expect(title).toContain('Silhouette: 0.47');
    expect(title).toContain('16 clusters');
    expect(title).toContain('space: projection/umap_3d');
    expect(title).not.toContain('subsampled'); // sampled=false
  });

  it('mentions subsampling when sampled', () => {
    expect(qualityTitle({ ...SAMPLE, sampled: true })).toContain('subsampled');
  });
});

describe('pickLevelMetrics', () => {
  const stored = { topic: { topic_diversity: 0.9 }, subtopic: { topic_diversity: 0.7 } };

  it('extracts the requested level', () => {
    expect(pickLevelMetrics(stored, 'topic')).toEqual({ topic_diversity: 0.9 });
    expect(pickLevelMetrics(stored, 'subtopic')).toEqual({ topic_diversity: 0.7 });
  });

  it('returns null for missing levels or malformed blobs', () => {
    expect(pickLevelMetrics({ topic: { a: 1 } }, 'subtopic')).toBeNull();
    expect(pickLevelMetrics(null, 'topic')).toBeNull();
    expect(pickLevelMetrics('junk', 'topic')).toBeNull();
    expect(pickLevelMetrics({ topic: 'junk' }, 'topic')).toBeNull();
  });
});

describe('metric options', () => {
  it('default selection excludes the disabled DBCV pill', () => {
    const disabled = METRIC_OPTIONS.filter((o) => o.disabled).map((o) => o.name);
    expect(disabled).toEqual(['dbcv']);
    for (const name of DEFAULT_SELECTED_METRICS) {
      expect(disabled).not.toContain(name);
    }
  });

  it('option names match the backend metric-selection names', () => {
    expect(METRIC_OPTIONS.map((o) => o.name).sort()).toEqual([
      'coherence_cv',
      'coherence_umass',
      'dbcv',
      'diversity',
      'silhouette',
    ]);
  });
});
