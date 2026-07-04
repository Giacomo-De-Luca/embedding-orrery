import { describe, it, expect } from 'vitest';
import type { LayerActivationsResult, ActiveFeatureResult } from '@/lib/graphql/mutations';
import {
  attachSaeIdentity,
  poolPromptFeatures,
  MAX_POOLED_ROWS,
  type SaeLayerActivations,
} from '../promptPooling';

function feat(
  index: number,
  activation: number,
  label = `feature ${index}`,
  density: number | null = 0.001,
): ActiveFeatureResult {
  return { index, activation, label, density };
}

function tok(position: number, ...features: ActiveFeatureResult[]) {
  return { token: `t${position}`, position, features };
}

function layer(
  layerIdx: number,
  width: string,
  tokens: ReturnType<typeof tok>[],
): LayerActivationsResult {
  return { layer: layerIdx, width, tokens };
}

function saeLayer(
  layerIdx: number,
  width: string,
  tokens: ReturnType<typeof tok>[],
): SaeLayerActivations {
  return attachSaeIdentity([layer(layerIdx, width, tokens)], 'gemma-3-4b-it', [])[0];
}

describe('attachSaeIdentity', () => {
  it('resolves the exact saeId from the selection pairs by (layer, width)', () => {
    const pairs = [
      { modelId: 'gemma-3-4b-it', saeId: '9-gemmascope-2-res-16k' },
      { modelId: 'gemma-3-4b-it', saeId: '9-gemmascope-2-res-65k' },
    ];
    const result = attachSaeIdentity(
      [layer(9, '16k', []), layer(9, '65k', [])],
      'gemma-3-4b-it',
      pairs,
    );
    expect(result[0].saeId).toBe('9-gemmascope-2-res-16k');
    expect(result[1].saeId).toBe('9-gemmascope-2-res-65k');
    expect(result[0].modelId).toBe('gemma-3-4b-it');
  });

  it('falls back to the derived gemmascope saeId when no pair matches', () => {
    const result = attachSaeIdentity([layer(22, '16k', [])], 'gemma-3-4b-it', []);
    expect(result[0].saeId).toBe('22-gemmascope-2-res-16k');
  });
});

describe('poolPromptFeatures', () => {
  it('max pooling takes the peak activation across tokens', () => {
    const layers = [saeLayer(9, '16k', [tok(0, feat(5, 2.0)), tok(1, feat(5, 7.0))])];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows).toHaveLength(1);
    expect(rows[0].activation).toBe(7.0);
  });

  it('mean pooling averages over tokens where the feature fired', () => {
    // fires on 2 of 3 tokens: (2 + 6) / 2 = 4
    const layers = [
      saeLayer(9, '16k', [tok(0, feat(5, 2.0)), tok(1), tok(2, feat(5, 6.0))]),
    ];
    const rows = poolPromptFeatures(layers, 'mean', 0.01);
    expect(rows[0].activation).toBe(4.0);
  });

  it('last pooling keeps only the final token features', () => {
    const layers = [
      saeLayer(9, '16k', [tok(0, feat(5, 9.0)), tok(1, feat(6, 3.0))]),
    ];
    const rows = poolPromptFeatures(layers, 'last', 0.01);
    expect(rows).toHaveLength(1);
    expect(rows[0].featureIndex).toBe(6);
    expect(rows[0].activation).toBe(3.0);
  });

  it('keeps the same feature index distinct across layers (no collision)', () => {
    const layers = [
      saeLayer(9, '16k', [tok(0, feat(5, 2.0))]),
      saeLayer(22, '16k', [tok(0, feat(5, 8.0))]),
    ];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows).toHaveLength(2);
    expect(new Set(rows.map((r) => r.saeId)).size).toBe(2);
  });

  it('keeps the same feature index distinct across widths at one layer', () => {
    const layers = [
      saeLayer(9, '16k', [tok(0, feat(5, 2.0))]),
      saeLayer(9, '65k', [tok(0, feat(5, 8.0))]),
    ];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows).toHaveLength(2);
    expect(rows.map((r) => r.saeId).sort()).toEqual([
      '9-gemmascope-2-res-16k',
      '9-gemmascope-2-res-65k',
    ]);
  });

  it('filters features above the density threshold, keeps null densities', () => {
    const layers = [
      saeLayer(9, '16k', [
        tok(0, feat(1, 5.0, 'dense', 0.5), feat(2, 4.0, 'sparse', 0.001), feat(3, 3.0, 'unknown', null)),
      ]),
    ];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows.map((r) => r.featureIndex).sort()).toEqual([2, 3]);
  });

  it('sorts globally by activation and normalizes similarity to the top row', () => {
    const layers = [
      saeLayer(9, '16k', [tok(0, feat(1, 2.0))]),
      saeLayer(22, '16k', [tok(0, feat(2, 8.0))]),
    ];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows.map((r) => r.featureIndex)).toEqual([2, 1]);
    expect(rows[0].similarity).toBe(1);
    expect(rows[1].similarity).toBeCloseTo(0.25);
  });

  it('returns empty for no layers or no tokens', () => {
    expect(poolPromptFeatures([], 'max', 0.01)).toEqual([]);
    expect(poolPromptFeatures([saeLayer(9, '16k', [])], 'max', 0.01)).toEqual([]);
  });

  it('caps the ranked list at MAX_POOLED_ROWS, keeping the top rows', () => {
    const features = Array.from({ length: MAX_POOLED_ROWS + 50 }, (_, i) =>
      feat(i, MAX_POOLED_ROWS + 50 - i),
    );
    const layers = [saeLayer(9, '16k', [tok(0, ...features)])];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows).toHaveLength(MAX_POOLED_ROWS);
    expect(rows[0].featureIndex).toBe(0); // highest activation survives
    expect(rows[rows.length - 1].activation).toBeGreaterThan(50); // tail dropped
  });

  it('single-layer output matches the pre-change page behavior', () => {
    // Regression guard: one layer, max pooling — mirrors the old page memo.
    const layers = [
      saeLayer(9, '16k', [
        tok(0, feat(10, 3.0, 'a', 0.002), feat(11, 1.0, 'b', 0.003)),
        tok(1, feat(10, 5.0, 'a', 0.002)),
      ]),
    ];
    const rows = poolPromptFeatures(layers, 'max', 0.01);
    expect(rows.map((r) => ({ i: r.featureIndex, a: r.activation, s: r.similarity }))).toEqual([
      { i: 10, a: 5.0, s: 1 },
      { i: 11, a: 1.0, s: 0.2 },
    ]);
    expect(rows[0].label).toBe('a');
    expect(rows[0].density).toBe(0.002);
  });
});
