/**
 * Tests for the density-haze math: per-cluster kernel/weight/exposure rules and
 * the GPU buffer builder. Pure — no Three.js, no DOM.
 */

import { describe, it, expect } from 'vitest';
import {
  clusterHazeParams,
  computeExposure,
  pointSizeCompensation,
  buildDensityHazeBuffers,
  reweightDensityHazeBuffers,
  median,
  DEFAULT_DENSITY_HAZE_TUNING,
  MIN_HAZE_CLUSTER_POINTS,
  SPRITE_SIGMA_SPAN,
  CLAMP_COMPENSATION_GLSL,
  type DensityHazeTuning,
} from '../densityHaze/densityHazeMath';
import type { ClusterData } from '../clusterGeometry';
import type { Point3D } from '../../types/types';

const RMS2D_FACTOR = Math.sqrt(2 / 3);

const tuning = (overrides: Partial<DensityHazeTuning> = {}): DensityHazeTuning => ({
  ...DEFAULT_DENSITY_HAZE_TUNING,
  ...overrides,
});

/** Build a ClusterData with `n` points spread isotropically with the given std. */
function cluster(n: number, std: number, color: string, origin = 0): ClusterData {
  const points: Point3D[] = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    points.push({
      id: `${color}-${i}`,
      index: i,
      x: origin + std * Math.cos(t),
      y: origin + std * Math.sin(t),
      z: origin + std * Math.sin(2 * t),
    } as Point3D);
  }
  return {
    points,
    centroid: { x: origin, y: origin, z: origin },
    std: { x: std, y: std, z: std },
    color,
  };
}

describe('median', () => {
  it('handles odd and even lengths and ignores input order', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([7])).toBe(7);
  });

  it('returns NaN for an empty array', () => {
    expect(Number.isNaN(median([]))).toBe(true);
  });
});

describe('clusterHazeParams', () => {
  it('derives the kernel sigma from the projected rms radius', () => {
    const p = clusterHazeParams({ n: 100, rms3d: 3 }, tuning({ kernelFrac: 0.8 }));
    expect(p.sigma).toBeCloseTo(0.8 * 3 * RMS2D_FACTOR, 10);
  });

  it('expected peak is n·2k²/(1+2k²) for kernelFrac k (sigma is k × rms radius = k·√2 × per-axis std)', () => {
    const k = 0.8;
    const p = clusterHazeParams({ n: 100, rms3d: 3 }, tuning({ kernelFrac: k }));
    expect(p.peak).toBeCloseTo((100 * 2 * k * k) / (1 + 2 * k * k), 10);
  });

  it('uses the per-axis std in the convolution: sigma equal to the per-axis std halves the peak', () => {
    const axisStd = 2;
    const rms3d = axisStd * Math.sqrt(3);
    const kernelFrac = axisStd / (rms3d * RMS2D_FACTOR);
    const p = clusterHazeParams({ n: 100, rms3d }, tuning({ kernelFrac }));
    expect(p.sigma).toBeCloseTo(axisStd, 10);
    expect(p.peak).toBeCloseTo(50, 10);
  });

  it('gamma 0 keeps raw density; gamma 1 equalises every cluster peak', () => {
    const raw = clusterHazeParams({ n: 500, rms3d: 1 }, tuning({ gamma: 0 }));
    expect(raw.weight).toBe(1);

    const eq = clusterHazeParams({ n: 500, rms3d: 1 }, tuning({ gamma: 1 }));
    expect(eq.peak * eq.weight).toBeCloseTo(1, 10);
  });

  it('gamma 0.5 applies a square-root lift: peak·weight = sqrt(peak)', () => {
    const p = clusterHazeParams({ n: 400, rms3d: 2 }, tuning({ gamma: 0.5 }));
    expect(p.peak * p.weight).toBeCloseTo(Math.sqrt(p.peak), 10);
  });

  it('denser clusters get smaller weights for any gamma > 0', () => {
    const sparse = clusterHazeParams({ n: 50, rms3d: 1 }, tuning({ gamma: 0.5 }));
    const dense = clusterHazeParams({ n: 5000, rms3d: 1 }, tuning({ gamma: 0.5 }));
    expect(dense.weight).toBeLessThan(sparse.weight);
  });

  it('applies the sigma floor to degenerate clusters', () => {
    const p = clusterHazeParams({ n: 10, rms3d: 0 }, tuning(), 0.05);
    expect(p.sigma).toBe(0.05);
    expect(Number.isFinite(p.weight)).toBe(true);
    expect(p.weight).toBeGreaterThan(0);
  });
});

describe('computeExposure', () => {
  it('maps the median cluster peak onto the target', () => {
    expect(computeExposure([0.2, 0.4, 0.8], 0.45)).toBeCloseTo(0.45 / 0.4, 10);
  });

  it('falls back to 1 when there is nothing to normalise', () => {
    expect(computeExposure([], 0.45)).toBe(1);
    expect(computeExposure([0, 0], 0.45)).toBe(1);
  });
});

describe('pointSizeCompensation', () => {
  it('is 1 when the ideal size is inside the clamp range', () => {
    expect(pointSizeCompensation(6, 6)).toBe(1);
  });

  it('divides by the area gain when the minimum clamp enlarges the kernel', () => {
    expect(pointSizeCompensation(2, 4)).toBeCloseTo(0.25, 10);
  });

  it('never boosts when the maximum clamp shrinks the kernel (haze fades at high zoom)', () => {
    expect(pointSizeCompensation(40, 32)).toBe(1);
  });

  it('guards a zero clamped size', () => {
    expect(pointSizeCompensation(5, 0)).toBe(1);
  });

  it('ships a GLSL twin declaring the function the splat shader calls', () => {
    expect(CLAMP_COMPENSATION_GLSL).toContain('float clampCompensation(float ideal, float size)');
    expect(SPRITE_SIGMA_SPAN).toBe(6);
  });
});

describe('buildDensityHazeBuffers', () => {
  const map = new Map<string, ClusterData>([
    ['compact', cluster(20, 0.5, '#ff0000')],
    ['spread', cluster(200, 4, '#00ff00', 10)],
    ['tiny', cluster(MIN_HAZE_CLUSTER_POINTS - 1, 0.5, '#0000ff', 20)],
  ]);

  it('skips clusters below the point threshold and lays points out contiguously', () => {
    const b = buildDensityHazeBuffers(map, tuning());
    expect(b.count).toBe(220);
    expect(b.positions.length).toBe(220 * 3);
    expect(b.colors.length).toBe(220 * 3);
    expect(b.weights.length).toBe(220);
    expect(b.sigmas.length).toBe(220);
    expect(b.clusters.map((c) => c.key)).toEqual(['compact', 'spread']);
    expect(b.clusters[0]).toMatchObject({ offset: 0, count: 20 });
    expect(b.clusters[1]).toMatchObject({ offset: 20, count: 200 });
  });

  it('writes the cluster colour as 0–1 floats on every point of the cluster', () => {
    const b = buildDensityHazeBuffers(map, tuning());
    expect(Array.from(b.colors.slice(0, 3))).toEqual([1, 0, 0]);
    expect(Array.from(b.colors.slice(20 * 3, 20 * 3 + 3))).toEqual([0, 1, 0]);
    expect(Array.from(b.colors.slice(219 * 3, 220 * 3))).toEqual([0, 1, 0]);
  });

  it('gives every point of a cluster that cluster sigma and weight', () => {
    const b = buildDensityHazeBuffers(map, tuning({ gamma: 0.5 }));
    const compact = b.clusters[0].params;
    const spread = b.clusters[1].params;
    expect(b.sigmas[0]).toBeCloseTo(compact.sigma, 6);
    expect(b.sigmas[19]).toBeCloseTo(compact.sigma, 6);
    expect(b.sigmas[20]).toBeCloseTo(spread.sigma, 6);
    expect(b.weights[0]).toBeCloseTo(compact.weight, 6);
    expect(b.weights[219]).toBeCloseTo(spread.weight, 6);
    // the 200-point cluster is denser per kernel area, so it is weighted down
    expect(spread.weight).toBeLessThan(compact.weight);
  });

  it('derives the exposure from the included clusters only', () => {
    const t = tuning({ exposureTarget: 0.45 });
    const b = buildDensityHazeBuffers(map, t);
    const products = b.clusters.map((c) => c.params.peak * c.params.weight);
    expect(b.exposure).toBeCloseTo(0.45 / median(products), 10);
  });

  it('floors the kernel of a degenerate cluster relative to the largest cluster', () => {
    const flat = cluster(12, 0, '#ff00ff', 30);
    const b = buildDensityHazeBuffers(new Map([['compact', cluster(20, 0.5, '#ff0000')], ['flat', flat]]), tuning());
    const maxRms = Math.sqrt(3) * 0.5;
    expect(b.clusters[1].params.sigma).toBeCloseTo(maxRms * 1e-3, 12);
    expect(Number.isFinite(b.clusters[1].params.weight)).toBe(true);
  });

  it('falls back to mid grey for colours hexToRgb cannot parse', () => {
    const b = buildDensityHazeBuffers(new Map([['odd', cluster(20, 0.5, 'rgb(1, 2, 3)')]]), tuning());
    expect(Array.from(b.colors.slice(0, 3))).toEqual([0.5, 0.5, 0.5]);
  });

  it('returns an empty buffer set for an empty map', () => {
    const b = buildDensityHazeBuffers(new Map(), tuning());
    expect(b.count).toBe(0);
    expect(b.exposure).toBe(1);
    expect(b.clusters).toEqual([]);
  });
});

describe('reweightDensityHazeBuffers', () => {
  it('recomputes weights and exposure for a new gamma without touching geometry', () => {
    const map = new Map<string, ClusterData>([
      ['a', cluster(30, 1, '#ffffff')],
      ['b', cluster(300, 1, '#000000', 5)],
    ]);
    const b = buildDensityHazeBuffers(map, tuning({ gamma: 0 }));
    expect(b.weights[0]).toBe(1);
    expect(b.weights[31]).toBe(1);

    const r = reweightDensityHazeBuffers(b, tuning({ gamma: 1 }));
    expect(r.positions).toBe(b.positions);
    expect(r.sigmas).toBe(b.sigmas);
    expect(r.colors).toBe(b.colors);
    // gamma 1: every cluster's peak·weight is 1, so the median is 1
    expect(r.exposure).toBeCloseTo(DEFAULT_DENSITY_HAZE_TUNING.exposureTarget, 10);
    for (const c of r.clusters) {
      expect(c.params.peak * c.params.weight).toBeCloseTo(1, 10);
      expect(r.weights[c.offset]).toBeCloseTo(c.params.weight, 6);
      expect(r.weights[c.offset + c.count - 1]).toBeCloseTo(c.params.weight, 6);
    }
  });
});
