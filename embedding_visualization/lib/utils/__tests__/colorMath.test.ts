import { describe, it, expect } from 'vitest';
import { ColorMath } from '../colorMath';

describe('ColorMath', () => {
  it('round-trips hex through OKLCH', () => {
    for (const hex of ['#0b7285', '#d4a017', '#ffffff', '#000000', '#7048e8']) {
      const [L, C, h] = ColorMath.oklch(hex);
      expect(ColorMath.oklchToHex(L, C, h)).toBe(hex);
    }
  });

  it('computes WCAG contrast like the reference values', () => {
    expect(ColorMath.contrast('#ffffff', '#000000')).toBeCloseTo(21, 5);
    expect(ColorMath.contrast('#000000', '#ffffff')).toBeCloseTo(21, 5);
    expect(ColorMath.contrast('#0b7285', '#000000')).toBeCloseTo(3.76, 2);
    expect(ColorMath.contrast('#0b7285', '#ffffff')).toBeCloseTo(5.59, 2);
  });

  it('places known colours at the expected OKLCH lightness and hue', () => {
    const [L, C, h] = ColorMath.oklch('#0b7285');
    expect(L).toBeCloseTo(0.51, 2);
    expect(C).toBeCloseTo(0.087, 3);
    expect(h).toBeCloseTo(215, 0);
  });

  it('reports out-of-gamut OKLCH as null and finds the gamut edge', () => {
    expect(ColorMath.oklchToHex(0.6, 0.4, 200)).toBeNull();
    const edge = ColorMath.maxChroma(0.6, 200);
    expect(edge).toBeGreaterThan(0.09);
    expect(edge).toBeLessThan(0.12);
    expect(ColorMath.oklchToHex(0.6, edge, 200)).not.toBeNull();
  });

  it('measures zero distance for identical colours and symmetric distances otherwise', () => {
    expect(ColorMath.deltaE('#d4a017', '#d4a017')).toBe(0);
    expect(ColorMath.deltaE('#d4a017', '#0b7285')).toBeCloseTo(ColorMath.deltaE('#0b7285', '#d4a017'), 10);
    expect(ColorMath.deltaE('#d4a017', '#0b7285')).toBeGreaterThan(30);
  });

  it('collapses red and green under deutan simulation', () => {
    const normal = ColorMath.deltaE('#c92a2a', '#2f9e44');
    const deutan = ColorMath.deltaE('#c92a2a', '#2f9e44', 'deutan');
    expect(deutan).toBeLessThan(normal / 2);
  });

  it('accepts upper-case hex and rejects anything but #rrggbb', () => {
    expect(ColorMath.oklch('#D4A017')).toEqual(ColorMath.oklch('#d4a017'));
    expect(() => ColorMath.hexToSrgb('#abc')).toThrow(/Invalid hex/);
    expect(() => ColorMath.deltaE('#abc', '#000000')).toThrow(/Invalid hex/);
  });

  it('clamps out-of-range components when formatting', () => {
    expect(ColorMath.srgbToHex([1.2, -0.3, 0.5])).toBe('#ff0080');
  });

  it('tolerance decides whether a slightly out-of-gamut colour is accepted', () => {
    const edge = ColorMath.maxChroma(0.7, 200);
    expect(ColorMath.oklchToHex(0.7, edge + 0.002, 200, 0)).toBeNull();
    expect(ColorMath.oklchToHex(0.7, edge + 0.002, 200, 0.02)).not.toBeNull();
  });

  it('simulated deficiencies leave greys grey', () => {
    for (const kind of ['protan', 'deutan', 'tritan'] as const) {
      const [r, g, b] = ColorMath.simulateCvd('#808080', kind);
      expect(r).toBeCloseTo(g, 6);
      expect(g).toBeCloseTo(b, 6);
    }
  });

  it('tritan simulation separates blue from yellow far less than normal vision', () => {
    expect(ColorMath.deltaE('#364fc7', '#d4a017', 'tritan')).toBeLessThan(ColorMath.deltaE('#364fc7', '#d4a017'));
  });

  it('separation is never above the plain distance', () => {
    const a = '#e8590c', b = '#e67700';
    expect(ColorMath.separation(a, b)).toBeLessThanOrEqual(ColorMath.deltaE(a, b));
  });
});
