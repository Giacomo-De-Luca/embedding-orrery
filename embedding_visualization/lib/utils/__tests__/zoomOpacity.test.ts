import { describe, expect, it } from 'vitest';

import { ZoomOpacity } from '../zoomOpacity';

// 150k points sit on calculateMarkerStyle's 0.15 opacity floor at full view
const POINTS = 150_000;
const BOUNDS = { minX: 0, maxX: 100, minY: 0, maxY: 50 };

// Autorange pads the data, and scaleanchor leaves slack on one axis
const FULL_VIEW: [[number, number], [number, number]] = [[-5, 105], [-30, 80]];
// A quarter of the data extent on both axes: 1/16 of the data box
const ZOOM_4X: [[number, number], [number, number]] = [[40, 65], [20, 32.5]];
// calculateMarkerStyle(150000 / 16).opacity / 0.15 = (1 - 0.17 * log10(9375)) / 0.15
const ZOOM_4X_MULTIPLIER = 2.1651;

describe('ZoomOpacity', () => {
  const zoomOpacity = new ZoomOpacity(POINTS, BOUNDS);

  it('keeps the base opacity at full view and when zoomed out', () => {
    expect(zoomOpacity.multiplier(...FULL_VIEW)).toBe(1);
    expect(zoomOpacity.multiplier([-100, 200], [-100, 150])).toBe(1);
  });

  it('uses the opacity calculateMarkerStyle gives the visible share of the points', () => {
    expect(zoomOpacity.multiplier(...ZOOM_4X)).toBeCloseTo(ZOOM_4X_MULTIPLIER, 4);
  });

  it('does not change while panning, nor with a reversed range', () => {
    expect(zoomOpacity.multiplier([70, 95], [-10, 2.5])).toBeCloseTo(ZOOM_4X_MULTIPLIER, 4);
    expect(zoomOpacity.multiplier([65, 40], [32.5, 20])).toBeCloseTo(ZOOM_4X_MULTIPLIER, 4);
  });

  it('grows with the zoom up to the opacity of a handful of points', () => {
    const at = (zoom: number) => zoomOpacity.multiplier([0, 100 / zoom], [0, 50 / zoom]);
    expect(at(2)).toBeGreaterThan(1);
    expect(at(4)).toBeGreaterThan(at(2));
    // calculateMarkerStyle tops out at 0.8
    expect(at(1000)).toBeCloseTo(0.8 / 0.15, 6);
  });

  it('ignores an axis whose data has no extent', () => {
    const column = new ZoomOpacity(POINTS, { minX: 3, maxX: 3, minY: 0, maxY: 50 });
    // Only y is zoomed 4x: a quarter of the points, (1 - 0.17 * log10(37500)) / 0.15
    expect(column.multiplier([0, 1], [20, 32.5])).toBeCloseTo(1.4828, 4);
  });

  it('keeps the base opacity without data bounds', () => {
    expect(new ZoomOpacity(0, null).multiplier(...ZOOM_4X)).toBe(1);
  });

  describe('traceScale', () => {
    it('scales a trace by the zoom multiplier', () => {
      expect(zoomOpacity.traceScale({ marker: { opacity: 0.15 } }, ...ZOOM_4X)).toBeCloseTo(ZOOM_4X_MULTIPLIER, 4);
    });

    it('stops at fully opaque', () => {
      expect(zoomOpacity.traceScale({ marker: { opacity: 0.6 } }, ...ZOOM_4X)).toBeCloseTo(1 / 0.6, 6);
      expect(zoomOpacity.traceScale({ marker: { opacity: 1 } }, ...ZOOM_4X)).toBe(1);
    });

    it('leaves fixed traces, per-point opacities and traces without markers alone', () => {
      const fixed = { meta: ZoomOpacity.FIXED_META, marker: { opacity: 0.15 } };
      expect(zoomOpacity.traceScale(fixed, ...ZOOM_4X)).toBe(1);
      expect(zoomOpacity.traceScale({ marker: { opacity: [0.15, 0.3] } }, ...ZOOM_4X)).toBe(1);
      expect(zoomOpacity.traceScale({}, ...ZOOM_4X)).toBe(1);
    });
  });
});
