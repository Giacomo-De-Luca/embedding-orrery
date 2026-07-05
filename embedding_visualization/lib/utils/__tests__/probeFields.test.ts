/**
 * Tests for probe field utilities: merging per-item probe scores into item
 * metadata (fresh objects, id-keyed) and building ColorFieldOption entries
 * for the Color By dropdown.
 */
import { describe, it, expect } from 'vitest';

import {
  mergeProbeScores,
  buildProbeFieldOptions,
  type ProbeWithScores,
} from '../probeFields';

const ridgeProbe = (overrides: Partial<ProbeWithScores['probe']> = {}) => ({
  targetField: 'rating',
  kind: 'ridge',
  scoreField: 'probe_rating_ridge_score',
  residualField: 'probe_rating_ridge_residual',
  metrics: { val_r2: 0.8, val_spearman: 0.9 },
  nTrain: 80,
  nVal: 20,
  createdAt: '2026-07-05 12:00:00',
  ...overrides,
});

const massmeanProbe = () =>
  ridgeProbe({
    kind: 'massmean',
    scoreField: 'probe_rating_massmean_score',
    residualField: null,
    metrics: { val_spearman: 0.7 },
  });

describe('mergeProbeScores', () => {
  const ids = ['a', 'b', 'c'];
  const itemMetadata = [{ x: 1 }, { x: 2 }, { x: 3 }];

  it('returns a fresh array with fresh per-item objects', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe(),
        scores: { itemIds: ['a'], scores: [1.5], residuals: [0.1] },
      },
    ];
    const merged = mergeProbeScores(itemMetadata, ids, probes);
    expect(merged).not.toBe(itemMetadata);
    expect(merged[0]).not.toBe(itemMetadata[0]);
    expect(itemMetadata[0]).toEqual({ x: 1 }); // originals untouched
  });

  it('merges score and residual under the probe field names', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe(),
        scores: {
          itemIds: ['a', 'b', 'c'],
          scores: [1.5, 2.5, 3.5],
          residuals: [0.1, -0.2, null],
        },
      },
    ];
    const merged = mergeProbeScores(itemMetadata, ids, probes);
    expect(merged[0].probe_rating_ridge_score).toBe(1.5);
    expect(merged[1].probe_rating_ridge_score).toBe(2.5);
    expect(merged[0].probe_rating_ridge_residual).toBe(0.1);
    expect(merged[1].probe_rating_ridge_residual).toBe(-0.2);
    // null residual -> key absent, not null (numeric color path skips missing)
    expect('probe_rating_ridge_residual' in merged[2]).toBe(false);
  });

  it('skips score item ids that are not in the collection ids', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe(),
        scores: { itemIds: ['ghost', 'b'], scores: [9.9, 2.5], residuals: null },
      },
    ];
    const merged = mergeProbeScores(itemMetadata, ids, probes);
    expect(merged[1].probe_rating_ridge_score).toBe(2.5);
    expect(Object.values(merged[0])).not.toContain(9.9);
  });

  it('writes no residual keys when residualField is null (massmean)', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: massmeanProbe(),
        scores: { itemIds: ['a'], scores: [0.5], residuals: null },
      },
    ];
    const merged = mergeProbeScores(itemMetadata, ids, probes);
    expect(merged[0].probe_rating_massmean_score).toBe(0.5);
    expect(
      Object.keys(merged[0]).filter((k) => k.includes('residual')),
    ).toEqual([]);
  });

  it('merges multiple probes into the same items', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe(),
        scores: { itemIds: ['a'], scores: [1.0], residuals: [0.0] },
      },
      {
        probe: massmeanProbe(),
        scores: { itemIds: ['a'], scores: [-2.0], residuals: null },
      },
    ];
    const merged = mergeProbeScores(itemMetadata, ids, probes);
    expect(merged[0].probe_rating_ridge_score).toBe(1.0);
    expect(merged[0].probe_rating_massmean_score).toBe(-2.0);
  });
});

describe('buildProbeFieldOptions', () => {
  it('builds a sequential numeric score option with min/max', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe({ residualField: null }),
        scores: { itemIds: ['a', 'b', 'c'], scores: [2.0, -1.0, 5.0], residuals: null },
      },
    ];
    const options = buildProbeFieldOptions(probes);
    expect(options).toHaveLength(1);
    expect(options[0]).toMatchObject({
      field: 'probe_rating_ridge_score',
      displayName: 'rating · ridge score',
      valueType: 'numeric',
      recommendedScale: 'sequential',
      min: -1.0,
      max: 5.0,
    });
    expect(options[0].uniqueCount).toBe(3);
  });

  it('adds a residual option only when non-null residuals exist', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: ridgeProbe(),
        scores: {
          itemIds: ['a', 'b'],
          scores: [1.0, 2.0],
          residuals: [0.5, null],
        },
      },
    ];
    const options = buildProbeFieldOptions(probes);
    expect(options).toHaveLength(2);
    const residual = options[1];
    expect(residual.field).toBe('probe_rating_ridge_residual');
    expect(residual.displayName).toBe('rating · ridge residual');
    expect(residual.min).toBe(0.5);
    expect(residual.max).toBe(0.5);
  });

  it('omits the residual option for massmean or all-null residuals', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: massmeanProbe(),
        scores: { itemIds: ['a'], scores: [0.1], residuals: null },
      },
      {
        probe: ridgeProbe(),
        scores: { itemIds: ['a'], scores: [0.1], residuals: [null] },
      },
    ];
    const options = buildProbeFieldOptions(probes);
    expect(options.map((o) => o.field)).toEqual([
      'probe_rating_massmean_score',
      'probe_rating_ridge_score',
    ]);
  });

  it('returns an empty list for no probes', () => {
    expect(buildProbeFieldOptions([])).toEqual([]);
  });
});
