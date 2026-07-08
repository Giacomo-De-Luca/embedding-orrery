/**
 * Tests for the steering-strength hint: rho math, suggested-strength inverse,
 * band thresholds, and the table-lookup hint with graceful fallback. Uses an
 * injected mock table so it never depends on the (empty-until-generated)
 * residualNorms.json asset.
 */
import { describe, it, expect } from 'vitest';

import {
  computeRho,
  suggestedStrength,
  strengthBand,
  snapStrengthToSlider,
  steeringHint,
  layerMedianNorm,
  directionNorm,
  RHO_RECOMMENDED,
  RHO_SUBTLE_MAX,
  RHO_STRONG_MIN,
  type ResidualNormsTable,
} from '../steeringStrengthHint';

const MODEL = 'gemma-3-4b-it';

const TABLE: ResidualNormsTable = {
  [MODEL]: {
    checkpoint: 'google/gemma-3-4b-it',
    dModel: 2560,
    nLayers: 34,
    promptCount: 15,
    droppedBos: true,
    generatedAt: '2026-07-08T00:00:00',
    layers: [
      { layer: 9, median: 1000, p25: 900, p75: 1100, mean: 1000, count: 100 },
      { layer: 22, median: 4000, p25: 3800, p75: 4200, mean: 4000, count: 100 },
    ],
    directions: {
      refusal: { layer: 22, vecNorm: 200 },
    },
  },
};

describe('computeRho', () => {
  it('is |strength|·vecNorm / residualNorm', () => {
    expect(computeRho(1000, 1000)).toBe(1);
    expect(computeRho(500, 1000)).toBe(0.5);
    expect(computeRho(-500, 1000)).toBe(0.5); // magnitude only
    expect(computeRho(3, 1000, 200)).toBeCloseTo(0.6); // direction: 3·200/1000
  });

  it('guards a non-positive residual norm', () => {
    expect(computeRho(500, 0)).toBe(0);
  });
});

describe('suggestedStrength', () => {
  it('inverts computeRho', () => {
    const s = suggestedStrength(0.15, 1000); // 150
    expect(s).toBe(150);
    expect(computeRho(s, 1000)).toBeCloseTo(0.15);
  });

  it('accounts for vecNorm (directions)', () => {
    expect(suggestedStrength(0.6, 1000, 200)).toBeCloseTo(3);
  });
});

describe('strengthBand', () => {
  it('thresholds subtle/medium/strong', () => {
    expect(strengthBand(RHO_SUBTLE_MAX - 0.001)).toBe('subtle');
    expect(strengthBand(RHO_SUBTLE_MAX)).toBe('medium');
    expect(strengthBand(0.15)).toBe('medium');
    expect(strengthBand(RHO_STRONG_MIN)).toBe('strong');
    expect(strengthBand(1.0)).toBe('strong');
  });
});

describe('snapStrengthToSlider', () => {
  const sae = { min: -2000, max: 2000, step: 50 };

  it('snaps to the nearest step', () => {
    expect(snapStrengthToSlider(137, sae)).toBe(150);
    expect(snapStrengthToSlider(124, sae)).toBe(100);
  });

  it('clamps above the slider max (deep-layer overshoot)', () => {
    expect(snapStrengthToSlider(13333 * 0.15, sae)).toBe(2000); // ~2000 median × rho would exceed max
    expect(snapStrengthToSlider(50000, sae)).toBe(2000);
  });

  it('clamps below the slider min', () => {
    expect(snapStrengthToSlider(-9999, sae)).toBe(-2000);
  });

  it('handles the direction slider grid', () => {
    expect(snapStrengthToSlider(3.04, { min: -5, max: 5, step: 0.1 })).toBeCloseTo(3.0);
    expect(snapStrengthToSlider(7, { min: -5, max: 5, step: 0.1 })).toBe(5);
  });
});

describe('lookups', () => {
  it('reads layer median from the injected table', () => {
    expect(layerMedianNorm(MODEL, 9, TABLE)).toBe(1000);
    expect(layerMedianNorm(MODEL, 999, TABLE)).toBeUndefined();
    expect(layerMedianNorm('unknown', 9, TABLE)).toBeUndefined();
  });

  it('reads direction norm', () => {
    expect(directionNorm(MODEL, 'refusal', TABLE)).toEqual({ layer: 22, vecNorm: 200 });
    expect(directionNorm(MODEL, 'missing', TABLE)).toBeUndefined();
  });
});

describe('steeringHint', () => {
  it('computes an SAE-feature hint (vecNorm = 1)', () => {
    const hint = steeringHint({ modelId: MODEL, layerIndex: 9, strength: 800 }, TABLE);
    expect(hint).not.toBeNull();
    expect(hint!.vecNorm).toBe(1);
    expect(hint!.residualNorm).toBe(1000);
    expect(hint!.rho).toBeCloseTo(0.8);
    expect(hint!.band).toBe('strong');
    expect(hint!.suggestedStrength).toBeCloseTo(RHO_RECOMMENDED * 1000);
  });

  it('reads the same strength as a lower fraction at a deeper layer', () => {
    const shallow = steeringHint({ modelId: MODEL, layerIndex: 9, strength: 800 }, TABLE)!;
    const deep = steeringHint({ modelId: MODEL, layerIndex: 22, strength: 800 }, TABLE)!;
    expect(deep.rho).toBeLessThan(shallow.rho); // 0.2 < 0.8 — the whole point
  });

  it('uses the direction layer + vecNorm (not the row layerIndex)', () => {
    // Row shows layerIndex 14 for display; the refusal direction applies at 22.
    const hint = steeringHint(
      { modelId: MODEL, layerIndex: 14, strength: 3, directionName: 'refusal' },
      TABLE,
    )!;
    expect(hint.layer).toBe(22);
    expect(hint.vecNorm).toBe(200);
    expect(hint.residualNorm).toBe(4000);
    expect(hint.rho).toBeCloseTo(0.15); // 3·200/4000
  });

  it('returns null on missing model / layer / direction data', () => {
    expect(steeringHint({ modelId: null, layerIndex: 9, strength: 800 }, TABLE)).toBeNull();
    expect(steeringHint({ modelId: MODEL, layerIndex: 999, strength: 800 }, TABLE)).toBeNull();
    expect(
      steeringHint({ modelId: MODEL, layerIndex: 9, strength: 3, directionName: 'missing' }, TABLE),
    ).toBeNull();
  });
});
