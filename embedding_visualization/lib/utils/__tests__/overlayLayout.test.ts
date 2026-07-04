/**
 * Tests for the overlay-canvas layout helpers used by ScatterPlot3D's haze and
 * label overlays. Pure geometry — no DOM required.
 */

import { describe, it, expect } from 'vitest';
import {
  computeOverlayLayout,
  overlayLayoutEqual,
  type RectLike,
  type OverlayLayout,
} from '../overlayLayout';

const rect = (left: number, top: number, width: number, height: number): RectLike => ({
  left,
  top,
  width,
  height,
});

describe('computeOverlayLayout', () => {
  it('returns zero offset and the GL size when the GL canvas fills the container', () => {
    const container = rect(100, 50, 800, 600);
    const gl = rect(100, 50, 800, 600);
    expect(computeOverlayLayout(container, gl)).toEqual({
      offsetX: 0,
      offsetY: 0,
      cssW: 800,
      cssH: 600,
    });
  });

  it('computes the GL canvas offset within the container (margins/modebar)', () => {
    // Container at viewport (100,50); GL canvas inset by 12px left, 30px top.
    const container = rect(100, 50, 820, 640);
    const gl = rect(112, 80, 796, 590);
    expect(computeOverlayLayout(container, gl)).toEqual({
      offsetX: 12,
      offsetY: 30,
      cssW: 796,
      cssH: 590,
    });
  });

  it('handles fractional (subpixel) rects', () => {
    const container = rect(0.5, 0.25, 400, 300);
    const gl = rect(4.75, 10.125, 391.5, 279.5);
    const layout = computeOverlayLayout(container, gl);
    expect(layout.offsetX).toBeCloseTo(4.25);
    expect(layout.offsetY).toBeCloseTo(9.875);
    expect(layout.cssW).toBe(391.5);
    expect(layout.cssH).toBe(279.5);
  });

  it('handles a zero-size GL canvas (not yet laid out)', () => {
    const container = rect(0, 0, 0, 0);
    const gl = rect(0, 0, 0, 0);
    expect(computeOverlayLayout(container, gl)).toEqual({
      offsetX: 0,
      offsetY: 0,
      cssW: 0,
      cssH: 0,
    });
  });
});

describe('overlayLayoutEqual', () => {
  const base: OverlayLayout = { offsetX: 12, offsetY: 30, cssW: 796, cssH: 590 };

  it('is true for the same reference and for field-identical layouts', () => {
    expect(overlayLayoutEqual(base, base)).toBe(true);
    expect(overlayLayoutEqual(base, { ...base })).toBe(true);
  });

  it('is true when both are null', () => {
    expect(overlayLayoutEqual(null, null)).toBe(true);
  });

  it('is false when exactly one operand is null', () => {
    expect(overlayLayoutEqual(base, null)).toBe(false);
    expect(overlayLayoutEqual(null, base)).toBe(false);
  });

  it('is false when any single field differs', () => {
    expect(overlayLayoutEqual(base, { ...base, offsetX: 13 })).toBe(false);
    expect(overlayLayoutEqual(base, { ...base, offsetY: 31 })).toBe(false);
    expect(overlayLayoutEqual(base, { ...base, cssW: 797 })).toBe(false);
    expect(overlayLayoutEqual(base, { ...base, cssH: 591 })).toBe(false);
  });
});
