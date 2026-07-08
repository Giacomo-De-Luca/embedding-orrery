/**
 * Tests for the pure math behind the 2D density overlay (port of
 * embedding-atlas's density pipeline). No GL required.
 */

import { describe, it, expect } from 'vitest';
import {
  matrix3Multiply,
  applyMatrix3,
  buildPositionMatrix,
  computeViewingParams,
  approximateMaxDensity2D,
  assignCategoryChannels,
  buildColorMatrix,
  DENSITY_INTENSITY,
  DENSITY_FADE_START_ZOOM,
  DENSITY_FADE_END_ZOOM,
  DENSITY_MAX_ALPHA,
  DENSITY_CONTOUR_ALPHA,
  MUTED_DENSITY_TINT,
  linearizedColorBytes,
  type ActiveCategory,
} from '../densityMath';

// ---------------------------------------------------------------------------
// buildPositionMatrix
// ---------------------------------------------------------------------------

describe('buildPositionMatrix', () => {
  const ranges = { xRange: [2, 6] as [number, number], yRange: [-1, 3] as [number, number] };

  it('maps range corners to clip-space corners', () => {
    const m = buildPositionMatrix(ranges);
    expect(applyMatrix3(m, 2, -1)).toEqual([-1, -1]);
    expect(applyMatrix3(m, 6, 3)).toEqual([1, 1]);
  });

  it('maps the range midpoint to the origin', () => {
    const m = buildPositionMatrix(ranges);
    const [cx, cy] = applyMatrix3(m, 4, 1);
    expect(cx).toBeCloseTo(0, 12);
    expect(cy).toBeCloseTo(0, 12);
  });

  it('handles degenerate (zero-span) ranges without NaN', () => {
    const m = buildPositionMatrix({ xRange: [5, 5], yRange: [0, 2] });
    const [cx] = applyMatrix3(m, 5, 1);
    expect(Number.isFinite(cx)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// matrix3Multiply (safe-margin composition)
// ---------------------------------------------------------------------------

describe('matrix3Multiply', () => {
  it('applies the left matrix after the right one (column-vector convention)', () => {
    // Position matrix: scale by 2, translate by (1, 0)
    const p = [2, 0, 0, 0, 2, 0, 1, 0, 1];
    // Safe-margin adjustment: shrink x by 0.5
    const s = [0.5, 0, 0, 0, 1, 0, 0, 0, 1];
    const m = matrix3Multiply(s, p);
    // (3, 4) → p → (7, 8) → s → (3.5, 8)
    expect(applyMatrix3(m, 3, 4)).toEqual([3.5, 8]);
  });

  it('identity is neutral', () => {
    const p = buildPositionMatrix({ xRange: [0, 10], yRange: [0, 4] });
    const id = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    expect(matrix3Multiply(id, p)).toEqual(p);
  });
});

// ---------------------------------------------------------------------------
// computeViewingParams
// ---------------------------------------------------------------------------

/**
 * Direct transcription of Apple's densityScaler derivation
 * (EmbeddingViewImpl.svelte:44-89, scale form). Used to pin our span-based
 * reformulation of the scaler; the crossfade alpha intentionally diverges
 * (relative-zoom fade instead of an absolute density threshold).
 */
function appleDensityScaler(
  maxDensity: number,
  scale: number,
  pixelWidth: number,
  pixelHeight: number,
  pixelRatio: number,
) {
  const viewDimension = Math.max(pixelWidth, pixelHeight) / pixelRatio;
  const maxPointDensity = maxDensity / (scale * scale) / (viewDimension * viewDimension);
  const maxPixelDensity = maxPointDensity / (pixelRatio * pixelRatio);
  return (1 / maxPixelDensity) * 0.2;
}

describe('computeViewingParams', () => {
  // Map Apple's (scale, viewDim) convention onto span-based inputs for a
  // square view: dataPerCssPx = 2 / (scale · viewDim) in both axes, so
  // span = plotDim · dataPerCssPx = 2 / scale.
  const cases = [
    { maxDensity: 5000, scale: 1, viewDim: 800, dpr: 2 },
    { maxDensity: 5000, scale: 8, viewDim: 800, dpr: 2 },
    { maxDensity: 120, scale: 0.5, viewDim: 500, dpr: 1 },
    { maxDensity: 1e6, scale: 30, viewDim: 1200, dpr: 1.5 },
  ];

  it('scaler matches a direct transcription of Apple’s formula (square view, intensity 1)', () => {
    for (const c of cases) {
      const apple = appleDensityScaler(
        c.maxDensity, c.scale, c.viewDim * c.dpr, c.viewDim * c.dpr, c.dpr,
      );
      const ours = computeViewingParams({
        maxDensity: c.maxDensity,
        xSpan: 2 / c.scale,
        ySpan: 2 / c.scale,
        plotWidthCss: c.viewDim,
        plotHeightCss: c.viewDim,
        dpr: c.dpr,
        intensity: 1,
      });
      expect(ours.densityScaler).toBeCloseTo(apple, 8);
    }
  });

  it('default intensity runs DENSITY_INTENSITY× hotter than Apple’s scaler', () => {
    const c = cases[0];
    const input = {
      maxDensity: c.maxDensity,
      xSpan: 2 / c.scale,
      ySpan: 2 / c.scale,
      plotWidthCss: c.viewDim,
      plotHeightCss: c.viewDim,
      dpr: c.dpr,
    };
    const base = computeViewingParams({ ...input, intensity: 1 });
    const boosted = computeViewingParams(input);
    expect(boosted.densityScaler).toBeCloseTo(base.densityScaler * DENSITY_INTENSITY, 8);
    expect(boosted.densityAlpha).toBeCloseTo(base.densityAlpha, 12);
  });

  const base = {
    maxDensity: 1000,
    plotWidthCss: 800,
    plotHeightCss: 600,
    dpr: 2,
    initialXSpan: 100,
    initialYSpan: 80,
  };

  it('alpha is at the fill ceiling at the initial view and when zoomed out', () => {
    const full = computeViewingParams({ ...base, xSpan: 100, ySpan: 80 });
    expect(full.densityAlpha).toBe(DENSITY_MAX_ALPHA);
    const zoomedOut = computeViewingParams({ ...base, xSpan: 200, ySpan: 160 });
    expect(zoomedOut.densityAlpha).toBe(DENSITY_MAX_ALPHA);
  });

  it('alpha holds until the fade-start zoom, hits 0 at the fade-end zoom', () => {
    const atStart = computeViewingParams({
      ...base,
      xSpan: 100 / DENSITY_FADE_START_ZOOM,
      ySpan: 80 / DENSITY_FADE_START_ZOOM,
    });
    expect(atStart.densityAlpha).toBeCloseTo(DENSITY_MAX_ALPHA, 8);
    const atEnd = computeViewingParams({
      ...base,
      xSpan: 100 / DENSITY_FADE_END_ZOOM,
      ySpan: 80 / DENSITY_FADE_END_ZOOM,
    });
    expect(atEnd.densityAlpha).toBeCloseTo(0, 8);
    const beyond = computeViewingParams({ ...base, xSpan: 1e-6, ySpan: 1e-6 });
    expect(beyond.densityAlpha).toBe(0);
  });

  it('fade is monotone in zoom and hits half the ceiling at the log-space midpoint', () => {
    const midZoom = Math.sqrt(DENSITY_FADE_START_ZOOM * DENSITY_FADE_END_ZOOM);
    const mid = computeViewingParams({ ...base, xSpan: 100 / midZoom, ySpan: 80 / midZoom });
    expect(mid.densityAlpha).toBeCloseTo(0.5 * DENSITY_MAX_ALPHA, 8);
    const z4 = computeViewingParams({ ...base, xSpan: 100 / 4, ySpan: 80 / 4 });
    const z6 = computeViewingParams({ ...base, xSpan: 100 / 6, ySpan: 80 / 6 });
    expect(z4.densityAlpha).toBeGreaterThan(z6.densityAlpha);
  });

  it('without initial spans the overlay never fades', () => {
    const p = computeViewingParams({
      maxDensity: 1000, xSpan: 1e-6, ySpan: 1e-6,
      plotWidthCss: 800, plotHeightCss: 600, dpr: 2,
    });
    expect(p.densityAlpha).toBe(DENSITY_MAX_ALPHA);
  });

  it('contoursAlpha tracks the fade at its own (thinner) ceiling', () => {
    const p = computeViewingParams({ ...base, xSpan: 20, ySpan: 16 });
    expect(p.contoursAlpha).toBeCloseTo(
      p.densityAlpha * (DENSITY_CONTOUR_ALPHA / DENSITY_MAX_ALPHA), 10,
    );
  });
});

// ---------------------------------------------------------------------------
// approximateMaxDensity2D
// ---------------------------------------------------------------------------

describe('approximateMaxDensity2D', () => {
  it('coincident points: maxDensity = N / binWidth² with the std floor', () => {
    const n = 50;
    const x = new Float32Array(n).fill(3);
    const y = new Float32Array(n).fill(-2);
    // std = 0 → binWidth = 0.3 · max(std, 1e-3) = 3e-4
    const binWidth = 0.3 * 1e-3;
    expect(approximateMaxDensity2D(x, y)).toBeCloseTo(n / (binWidth * binWidth), 2);
  });

  it('a dense blob dominates a sparse cloud, outliers ignored', () => {
    const n = 400;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    // Sparse ring, deterministic
    for (let i = 0; i < n; i++) {
      x[i] = Math.cos((i / n) * 2 * Math.PI) * 10;
      y[i] = Math.sin((i / n) * 2 * Math.PI) * 10;
    }
    // Dense blob near the center
    for (let i = 0; i < 100; i++) { x[i] = 0.001 * i; y[i] = 0; }
    const withBlob = approximateMaxDensity2D(x, y);
    // Same data with the blob spread out
    const x2 = Float32Array.from(x);
    for (let i = 0; i < 100; i++) x2[i] = (i - 50) / 10;
    const spread = approximateMaxDensity2D(x2, y);
    expect(withBlob).toBeGreaterThan(spread);
    // Extreme outliers don't crash or dominate
    const x3 = Float32Array.from(x); x3[0] = 1e9;
    expect(Number.isFinite(approximateMaxDensity2D(x3, y))).toBe(true);
  });

  it('empty input returns 0', () => {
    expect(approximateMaxDensity2D(new Float32Array(0), new Float32Array(0))).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// assignCategoryChannels
// ---------------------------------------------------------------------------

const cats = (spec: [string, number][]): ActiveCategory[] =>
  spec.map(([name, count]) => ({ name, count, color: `#${name.charCodeAt(0).toString(16)}0000` }));

describe('assignCategoryChannels', () => {
  it('1-4 active categorical categories → categorical mode, count-desc order', () => {
    const r = assignCategoryChannels(cats([['b', 5], ['a', 20], ['c', 10]]), true, false);
    expect(r.mode).toBe('categorical');
    expect(r.channelOf.get('a')).toBe(0);
    expect(r.channelOf.get('c')).toBe(1);
    expect(r.channelOf.get('b')).toBe(2);
    expect(r.channelColors).toHaveLength(3);
  });

  it('>4 active categories → meanColor with a per-category color map', () => {
    const five = cats([['a', 1], ['b', 2], ['c', 3], ['d', 4], ['e', 5]]);
    const r = assignCategoryChannels(five, true, false);
    expect(r.mode).toBe('meanColor');
    expect(r.colorOfCategory?.size).toBe(5);
    expect(r.colorOfCategory?.get('a')).toBe(five[0].color);
  });

  it('muting down to 4 flips back to categorical', () => {
    const four = cats([['a', 1], ['b', 2], ['c', 3], ['d', 4]]);
    expect(assignCategoryChannels(four, true, false).mode).toBe('categorical');
  });

  it('numeric / no color field → muted regardless of category count', () => {
    const r = assignCategoryChannels(cats([['a', 1], ['b', 2]]), false, true);
    expect(r.mode).toBe('muted');
    expect(r.channelColors).toEqual([MUTED_DENSITY_TINT.dark]);
  });

  it('zero active categories → muted', () => {
    expect(assignCategoryChannels([], true, false).mode).toBe('muted');
  });

  it('count ties break deterministically by name', () => {
    const r1 = assignCategoryChannels(cats([['b', 5], ['a', 5]]), true, false);
    const r2 = assignCategoryChannels(cats([['a', 5], ['b', 5]]), true, false);
    expect(r1.channelOf.get('a')).toBe(0);
    expect(r2.channelOf.get('a')).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// buildColorMatrix
// ---------------------------------------------------------------------------

describe('linearizedColorBytes', () => {
  it('gamma-decodes hex colors into bytes', () => {
    expect(linearizedColorBytes('#ffffff', 2.2)).toEqual([255, 255, 255]);
    expect(linearizedColorBytes('#000000', 2.2)).toEqual([0, 0, 0]);
    const [r, g, b] = linearizedColorBytes('#808080', 2.2);
    const expected = Math.round(Math.pow(128 / 255, 2.2) * 255);
    expect([r, g, b]).toEqual([expected, expected, expected]);
  });
});

describe('buildColorMatrix', () => {
  it('linearizes hex colors with the given gamma and pads to 4 channels', () => {
    const m = buildColorMatrix(['#ff0000', '#808080'], 2.2);
    expect(m).toHaveLength(16);
    // Channel 0: pure red — pow(1, γ) = 1
    expect(m.slice(0, 4)).toEqual([1, 0, 0, 1]);
    // Channel 1: mid gray, linearized
    const g = Math.pow(128 / 255, 2.2);
    expect(m[4]).toBeCloseTo(g, 10);
    expect(m[5]).toBeCloseTo(g, 10);
    expect(m[6]).toBeCloseTo(g, 10);
    expect(m[7]).toBe(1);
    // Unused channels padded with neutral gray (matches embedding-atlas)
    expect(m.slice(8, 12)).toEqual([0.5, 0.5, 0.5, 1]);
    expect(m.slice(12, 16)).toEqual([0.5, 0.5, 0.5, 1]);
  });
});
