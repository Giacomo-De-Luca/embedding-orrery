import { describe, expect, it } from 'vitest';

import { LabelFonts, LabelStyle } from '../labelStyle';

const FALLBACK = 'GeistMono, ui-monospace, monospace';

/** A stand-in for `document.fonts` recording what it was asked to load. */
function fontSet(check: (font: string) => boolean, load?: (font: string) => Promise<unknown>) {
  const loaded: string[] = [];
  return {
    loaded,
    check,
    load: (font: string) => {
      loaded.push(font);
      return load ? load(font) : Promise.resolve([]);
    },
  };
}

describe('LabelFonts', () => {
  it('uses the family the page registered for --font-geist-mono', () => {
    expect(LabelFonts.resolveFamily('"GeistMono", ui-monospace, monospace')).toBe('"GeistMono", ui-monospace, monospace');
    expect(LabelFonts.resolveFamily('  "GeistMono", monospace ')).toBe('"GeistMono", monospace');
  });

  it('falls back to GeistMono when the variable is missing or empty', () => {
    expect(LabelFonts.resolveFamily('')).toBe(FALLBACK);
    expect(LabelFonts.resolveFamily(null)).toBe(FALLBACK);
  });

  it('falls back outside the browser', () => {
    expect(LabelFonts.family()).toBe(FALLBACK);
  });

  it('builds the cluster and point label fonts', () => {
    expect(LabelFonts.cluster()).toBe(`bold 13px ${FALLBACK}`);
    expect(LabelFonts.cluster(14)).toBe(`bold 14px ${FALLBACK}`);
    expect(LabelFonts.point()).toBe(`11px ${FALLBACK}`);
  });

  it('loads every label font when it is not available yet', async () => {
    const fonts = fontSet(() => false);
    await LabelFonts.whenReady(fonts);
    expect(fonts.loaded).toEqual([LabelFonts.cluster(), LabelFonts.point()]);
  });

  it('settles even when a font fails to load', async () => {
    const fonts = fontSet(() => false, () => Promise.reject(new Error('blocked')));
    await expect(LabelFonts.whenReady(fonts)).resolves.toBeUndefined();
  });

  it('reports whether the label fonts are already available', () => {
    expect(LabelFonts.isReady(fontSet(() => true))).toBe(true);
    expect(LabelFonts.isReady(fontSet(font => !font.startsWith('bold')))).toBe(false);
    // Outside the browser there is nothing to wait for
    expect(LabelFonts.isReady(undefined)).toBe(true);
  });
});

describe('LabelStyle', () => {
  it('desaturates cluster text per view and theme (2D and 3D kept apart for now)', () => {
    expect(new LabelStyle(false, '2d').clusterFill('#ff0000')).toBe('rgb(183,43,43)');
    expect(new LabelStyle(true, '2d').clusterFill('#ff0000')).toBe('rgb(236,58,58)');
    expect(new LabelStyle(false, '3d').clusterFill('#ff0000')).toBe('rgb(199,34,34)');
    expect(new LabelStyle(true, '3d').clusterFill('#ff0000')).toBe('rgb(249,19,19)');
  });

  it('outlines cluster and point text for the theme', () => {
    const dark = new LabelStyle(true, '3d');
    const light = new LabelStyle(false, '3d');
    expect(dark.clusterStroke).toEqual({ color: 'rgba(15, 23, 42, 0.85)', width: 4 });
    expect(light.clusterStroke).toEqual({ color: 'rgba(255, 255, 255, 0.6)', width: 3 });
    expect(dark.pointStroke).toEqual({ color: 'rgba(15, 23, 42, 0.8)', width: 3 });
    expect(light.pointStroke).toEqual({ color: 'rgba(255, 255, 255, 0.8)', width: 3 });
    expect(dark.pointFill).toBe('#e2e8f0');
    expect(light.pointFill).toBe('#1e293b');
  });

  it('fades point labels by similarity and keeps the selected one opaque', () => {
    expect(LabelStyle.pointAlpha(0, false)).toBeCloseTo(0.4);
    expect(LabelStyle.pointAlpha(0.5, false)).toBeCloseTo(0.7);
    expect(LabelStyle.pointAlpha(1, false)).toBeCloseTo(1);
    expect(LabelStyle.pointAlpha(0, true)).toBe(1);
  });
});
