/**
 * Tests for probe field utilities: merging per-item probe scores into item
 * metadata (fresh objects, id-keyed) and building ColorFieldOption entries
 * for the Color By dropdown.
 */
import { describe, it, expect } from 'vitest';

import {
  mergeProbeScores,
  buildProbeFieldOptions,
  buildBinaryActualResolver,
  isProbeTargetOption,
  formatTargetMapping,
  probeAbsErrorField,
  probeConfusionField,
  resolveProbeTargetSelection,
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

const logregProbe = (overrides: Partial<ProbeWithScores['probe']> = {}) =>
  ridgeProbe({
    targetField: 'label',
    kind: 'logreg',
    scoreField: 'probe_label_logreg_score',
    residualField: null,
    metrics: { val_auc: 0.9 },
    ...overrides,
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
    // score + residual + derived |error|
    expect(options).toHaveLength(3);
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

describe('derived error field names', () => {
  it('derives the |error| field from the residual field', () => {
    expect(probeAbsErrorField(ridgeProbe())).toBe('probe_rating_ridge_abserr');
    expect(probeAbsErrorField(massmeanProbe())).toBeNull();
  });

  it('derives the confusion field for logreg only', () => {
    expect(probeConfusionField(logregProbe())).toBe('probe_label_logreg_confusion');
    expect(probeConfusionField(ridgeProbe())).toBeNull();
  });
});

describe('buildBinaryActualResolver', () => {
  it('uses the target mapping for text targets', () => {
    const resolve = buildBinaryActualResolver(
      [{ label: 'safe' }, { label: 'unsafe' }],
      'label',
      { safe: 0, unsafe: 1 },
    );
    expect(resolve).not.toBeNull();
    expect(resolve!({ label: 'unsafe' })).toBe(1);
    expect(resolve!({ label: 'safe' })).toBe(0);
    expect(resolve!({ label: 'other' })).toBeNull();
    expect(resolve!({})).toBeNull();
  });

  it('maps the larger of two numeric values to 1 (backend convention)', () => {
    const resolve = buildBinaryActualResolver(
      [{ label: 3 }, { label: 7 }, { label: 3 }],
      'label',
      null,
    );
    expect(resolve).not.toBeNull();
    expect(resolve!({ label: 7 })).toBe(1);
    expect(resolve!({ label: 3 })).toBe(0);
    expect(resolve!({ label: null })).toBeNull();
  });

  it('coerces string-encoded numeric targets (backend TRY_CAST parity)', () => {
    // Backend trains "3"/"7" string columns via TRY_CAST with no targetMapping;
    // the client mirror must resolve them the same way.
    const resolve = buildBinaryActualResolver(
      [{ label: '3' }, { label: '7' }],
      'label',
      null,
    );
    expect(resolve).not.toBeNull();
    expect(resolve!({ label: '7' })).toBe(1);
    expect(resolve!({ label: '3' })).toBe(0);
    expect(resolve!({ label: '' })).toBeNull();
  });

  it('returns null for non-binary numeric targets', () => {
    expect(
      buildBinaryActualResolver([{ x: 1 }, { x: 2 }, { x: 3 }], 'x', null),
    ).toBeNull();
    expect(buildBinaryActualResolver([{ x: 1 }], 'x', null)).toBeNull();
  });
});

describe('mergeProbeScores derived error fields', () => {
  it('adds |residual| under the abserr field, skipping null residuals', () => {
    const merged = mergeProbeScores(
      [{ x: 1 }, { x: 2 }, { x: 3 }],
      ['a', 'b', 'c'],
      [
        {
          probe: ridgeProbe(),
          scores: {
            itemIds: ['a', 'b', 'c'],
            scores: [1, 2, 3],
            residuals: [0.5, -0.2, null],
          },
        },
      ],
    );
    expect(merged[0].probe_rating_ridge_abserr).toBe(0.5);
    expect(merged[1].probe_rating_ridge_abserr).toBe(0.2);
    expect('probe_rating_ridge_abserr' in merged[2]).toBe(false);
  });

  it('adds confusion categories for logreg against the actual class', () => {
    const merged = mergeProbeScores(
      [{ label: 1 }, { label: 0 }, { label: 0 }, { label: 1 }, { label: null }],
      ['a', 'b', 'c', 'd', 'e'],
      [
        {
          probe: logregProbe(),
          scores: {
            itemIds: ['a', 'b', 'c', 'd', 'e'],
            scores: [0.9, 0.2, 0.7, 0.1, 0.8],
            residuals: null,
          },
        },
      ],
    );
    expect(merged[0].probe_label_logreg_confusion).toBe('TP');
    expect(merged[1].probe_label_logreg_confusion).toBe('TN');
    expect(merged[2].probe_label_logreg_confusion).toBe('FP');
    expect(merged[3].probe_label_logreg_confusion).toBe('FN');
    // Missing target -> no confusion key.
    expect('probe_label_logreg_confusion' in merged[4]).toBe(false);
  });

  it('uses the target mapping for text-target confusion', () => {
    const merged = mergeProbeScores(
      [{ label: 'unsafe' }, { label: 'safe' }],
      ['a', 'b'],
      [
        {
          probe: logregProbe({ targetMapping: { safe: 0, unsafe: 1 } }),
          scores: { itemIds: ['a', 'b'], scores: [0.9, 0.6], residuals: null },
        },
      ],
    );
    expect(merged[0].probe_label_logreg_confusion).toBe('TP');
    expect(merged[1].probe_label_logreg_confusion).toBe('FP');
  });
});

describe('buildProbeFieldOptions derived error options', () => {
  it('adds an |error| option alongside the residual option', () => {
    const options = buildProbeFieldOptions([
      {
        probe: ridgeProbe(),
        scores: {
          itemIds: ['a', 'b'],
          scores: [1, 2],
          residuals: [0.5, -0.2],
        },
      },
    ]);
    const abserr = options.find((o) => o.field === 'probe_rating_ridge_abserr');
    expect(abserr).toMatchObject({
      displayName: 'rating · ridge |error|',
      valueType: 'numeric',
      recommendedScale: 'sequential',
      min: 0.2,
      max: 0.5,
    });
  });

  it('adds a categorical confusion option when the target is resolvable', () => {
    const options = buildProbeFieldOptions(
      [
        {
          probe: logregProbe(),
          scores: { itemIds: ['a', 'b'], scores: [0.9, 0.2], residuals: null },
        },
      ],
      [{ label: 1 }, { label: 0 }],
    );
    const confusion = options.find((o) => o.field === 'probe_label_logreg_confusion');
    expect(confusion).toMatchObject({
      displayName: 'label · logreg confusion',
      valueType: 'string',
      recommendedScale: 'categorical',
    });
  });

  it('omits the confusion option without metadata or a binary target', () => {
    const probes: ProbeWithScores[] = [
      {
        probe: logregProbe(),
        scores: { itemIds: ['a'], scores: [0.9], residuals: null },
      },
    ];
    // No metadata passed (backwards compatible), then a non-binary target.
    expect(
      buildProbeFieldOptions(probes).some((o) => o.field.endsWith('_confusion')),
    ).toBe(false);
    expect(
      buildProbeFieldOptions(probes, [{ label: 1 }, { label: 2 }, { label: 3 }]).some(
        (o) => o.field.endsWith('_confusion'),
      ),
    ).toBe(false);
  });
});

describe('isProbeTargetOption', () => {
  const numericOpt = {
    field: 'rating',
    displayName: 'rating',
    valueType: 'numeric' as const,
    uniqueCount: 120,
    recommendedScale: 'sequential' as const,
  };

  it('accepts numeric fields', () => {
    expect(isProbeTargetOption(numericOpt)).toBe(true);
  });

  it('rejects probe-derived fields', () => {
    expect(
      isProbeTargetOption({ ...numericOpt, field: 'probe_rating_ridge_score' }),
    ).toBe(false);
  });

  it('accepts binary categorical fields (exactly 2 values)', () => {
    expect(
      isProbeTargetOption({
        ...numericOpt,
        field: 'safety',
        valueType: 'string',
        uniqueCount: 2,
        recommendedScale: 'categorical',
      }),
    ).toBe(true);
  });

  it('rejects non-binary categorical fields', () => {
    expect(
      isProbeTargetOption({
        ...numericOpt,
        field: 'venue',
        valueType: 'string',
        uniqueCount: 7,
        recommendedScale: 'categorical',
      }),
    ).toBe(false);
    expect(
      isProbeTargetOption({
        ...numericOpt,
        field: 'constant',
        valueType: 'string',
        uniqueCount: 1,
        recommendedScale: 'categorical',
      }),
    ).toBe(false);
  });
});

describe('formatTargetMapping', () => {
  it('renders values sorted by mapped number', () => {
    expect(formatTargetMapping({ unsafe: 1, safe: 0 })).toBe('safe → 0 · unsafe → 1');
  });

  it('returns null for missing mapping', () => {
    expect(formatTargetMapping(null)).toBeNull();
    expect(formatTargetMapping(undefined)).toBeNull();
  });
});


describe('resolveProbeTargetSelection', () => {
  it('clears the explicit selection when coloring by a real field (follow mode)', () => {
    expect(resolveProbeTargetSelection('rating', 'rating', 'rating')).toBeNull();
    expect(resolveProbeTargetSelection(null, 'x', 'x')).toBeNull();
  });

  it('pins the last target when auto-recolor lands on a probe field in follow mode', () => {
    expect(
      resolveProbeTargetSelection('probe_safety_massmean_score', null, 'safety'),
    ).toBe('safety');
  });

  it('keeps an explicit selection over the last target', () => {
    expect(
      resolveProbeTargetSelection('probe_safety_massmean_score', 'other', 'safety'),
    ).toBe('other');
  });
});
