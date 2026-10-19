import { describe, it, expect } from 'vitest';
import { PaletteDesigner, PaletteAudit, DEFAULT_CANDIDATE_SPEC } from '../paletteDesign';
import { GENERATED_PALETTE_RECIPES } from '../paletteRecipes';
import { CATEGORY_PALETTES, COSMIC_FAMILIES, COSMIC_FAMILY_ORDER } from '../categoryPalettes';
import { ColorMath } from '../colorMath';
import { MISSING_CATEGORY_COLOR, UNCLUSTERED_COLOR } from '../categoryColors';

describe('PaletteDesigner.candidates', () => {
  it('honours hue arcs, lightness, chroma and contrast constraints', () => {
    const pool = PaletteDesigner.candidates({
      arcs: [[120, 180]], lightness: [0.6, 0.8], hueStep: 10, lightnessStep: 0.1,
      chromaFractions: [1], minChroma: 0.1, minContrastOnBlack: 3,
    });
    expect(pool.length).toBeGreaterThan(0);
    for (const hex of pool) {
      const [L, C, h] = ColorMath.oklch(hex);
      expect(h).toBeGreaterThanOrEqual(119);
      expect(h).toBeLessThanOrEqual(181);
      expect(L).toBeGreaterThanOrEqual(0.59);
      expect(L).toBeLessThanOrEqual(0.81);
      expect(C).toBeGreaterThanOrEqual(0.1);
      expect(ColorMath.contrast(hex, '#000000')).toBeGreaterThanOrEqual(3);
    }
    expect(new Set(pool).size).toBe(pool.length);
  });

  it('treats hue arcs as clockwise spans that may wrap past 360 or through 0', () => {
    const hues = (arcs: [number, number][]) =>
      PaletteDesigner.candidates({ arcs, hueStep: 20, lightness: [0.7, 0.7], chromaFractions: [1] })
        .map((hex) => Math.round(ColorMath.oklch(hex)[2] / 20) * 20 % 360);
    const past360 = new Set(hues([[300, 380]]));
    const through0 = new Set(hues([[300, 20]]));
    expect(past360).toEqual(through0);
    expect(past360.has(0)).toBe(true);
    expect(past360.has(20)).toBe(true);
    expect(past360.has(340)).toBe(true);
    expect(past360.has(180)).toBe(false);
    expect(new Set(hues([[0, 360]])).size).toBe(18);
  });

  it('honours the white-contrast floor and the chroma fraction', () => {
    const pool = PaletteDesigner.candidates({ hueStep: 15, lightnessStep: 0.1, chromaFractions: [0.5], minContrastOnWhite: 3, minChroma: 0.05 });
    expect(pool.length).toBeGreaterThan(0);
    for (const hex of pool) {
      expect(ColorMath.contrast(hex, '#ffffff')).toBeGreaterThanOrEqual(3);
      const [L, C, h] = ColorMath.oklch(hex);
      expect(C / ColorMath.maxChroma(L, h)).toBeCloseTo(0.5, 1);
    }
  });

  it('falls back to the defaults for fields passed as undefined', () => {
    expect(PaletteDesigner.candidates({ arcs: undefined, hueStep: 30, lightnessStep: 0.1 }))
      .toEqual(PaletteDesigner.candidates({ hueStep: 30, lightnessStep: 0.1 }));
  });

  it('keeps candidates away from the colours it is told to avoid', () => {
    const pool = PaletteDesigner.candidates({ hueStep: 6, avoid: [UNCLUSTERED_COLOR], minDistanceToAvoided: 12 });
    for (const hex of pool) expect(ColorMath.deltaE(hex, UNCLUSTERED_COLOR)).toBeGreaterThanOrEqual(12);
  });
});

describe('PaletteDesigner.farthestFirst', () => {
  const pool = PaletteDesigner.candidates({ hueStep: 12, lightnessStep: 0.1, chromaFractions: [1] });

  it('starts with the seed and never repeats a colour', () => {
    const out = PaletteDesigner.farthestFirst(['#0b7285'], pool, 12);
    expect(out[0]).toBe('#0b7285');
    expect(out).toHaveLength(12);
    expect(new Set(out).size).toBe(12);
  });

  it('is pure: repeated calls return the same order', () => {
    const a = PaletteDesigner.farthestFirst(['#0b7285'], pool, 10);
    const b = PaletteDesigner.farthestFirst(['#0b7285'], pool, 10);
    expect(a).toEqual(b);
  });

  it('depends on pool order only for exact ties, and skips pool entries already in the seed', () => {
    const forward = PaletteDesigner.farthestFirst(['#0b7285'], pool, 12);
    const reversed = PaletteDesigner.farthestFirst(['#0b7285'], [...pool].reverse(), 12);
    expect(reversed).toEqual(forward);
    expect(PaletteDesigner.farthestFirst(['#0b7285'], ['#0b7285', '#d4a017'], 3)).toEqual(['#0b7285', '#d4a017']);
  });

  it('picks the farthest candidate at each step', () => {
    const out = PaletteDesigner.farthestFirst(['#0b7285'], pool, 3);
    const chosenFar = Math.min(...out.slice(0, 2).map((c) => ColorMath.separation(c, out[2])));
    for (const hex of pool) {
      if (out.includes(hex)) continue;
      const d = Math.min(...out.slice(0, 2).map((c) => ColorMath.separation(c, hex)));
      expect(d).toBeLessThanOrEqual(chosenFar + 1e-9);
    }
  });

  it('stops when the pool runs out', () => {
    expect(PaletteDesigner.farthestFirst(['#0b7285'], ['#d4a017'], 5)).toEqual(['#0b7285', '#d4a017']);
  });
});

describe('PaletteDesigner.prefixOrder', () => {
  it('reorders a set so every prefix is as spread out as possible', () => {
    const mains = COSMIC_FAMILY_ORDER.map((f) => COSMIC_FAMILIES[f].main);
    const ordered = PaletteDesigner.prefixOrder(mains, COSMIC_FAMILIES.teal.main);
    expect(ordered).toHaveLength(mains.length);
    expect(new Set(ordered)).toEqual(new Set(mains));
    expect(ordered[0]).toBe(COSMIC_FAMILIES.teal.main);
  });

  it('refuses a first colour that is not in the set instead of dropping one', () => {
    expect(() => PaletteDesigner.prefixOrder(['#0b7285', '#d4a017'], '#ffffff')).toThrow(/not in the palette/);
  });
});

describe('committed palettes match their recipes', () => {
  it('cosmic family order is the prefix-optimal order starting from teal', () => {
    const mains = COSMIC_FAMILY_ORDER.map((f) => COSMIC_FAMILIES[f].main);
    expect(PaletteDesigner.prefixOrder(mains, COSMIC_FAMILIES.teal.main)).toEqual(mains);
  });

  for (const recipe of GENERATED_PALETTE_RECIPES) {
    it(`${recipe.key} regenerates from its recipe`, () => {
      expect(PaletteDesigner.generate(recipe)).toEqual([...CATEGORY_PALETTES[recipe.key].colors]);
    });
  }
});

describe('PaletteAudit', () => {
  it('reports the closest pair under normal and CVD vision', () => {
    const stats = PaletteAudit.pairStats(['#0b7285', '#d4a017', '#e8590c', '#e67700'], 'all');
    expect(stats.normalPair).toEqual(['#e8590c', '#e67700']);
    expect(stats.normal).toBeCloseTo(6.1, 1);
    expect(stats.cvd).toBeLessThan(stats.normal);
    expect(stats.cvdPair).toEqual(['#e8590c', '#e67700']);
    expect(['protan', 'deutan']).toContain(stats.cvdKind);
  });

  it('prefix report skips sizes larger than the palette and minContrast finds the weakest slot', () => {
    expect(PaletteAudit.prefixReport(['#0b7285', '#d4a017', '#c92a2a'], [2, 3, 8]).map((p) => p.k)).toEqual([2, 3]);
    expect(PaletteAudit.minContrast(['#0b7285', '#ffffff'], '#000000')).toBeCloseTo(ColorMath.contrast('#0b7285', '#000000'), 6);
  });

  it('adjacent mode only looks at neighbours', () => {
    const adjacent = PaletteAudit.pairStats(['#e8590c', '#0b7285', '#e67700'], 'adjacent');
    expect(adjacent.normal).toBeGreaterThan(15);
  });

  it('prefix report shrinks or holds as the prefix grows', () => {
    const report = PaletteAudit.prefixReport(CATEGORY_PALETTES.deepField.colors, [4, 8, 16, 32, 64]);
    for (let i = 1; i < report.length; i++) expect(report[i].normal).toBeLessThanOrEqual(report[i - 1].normal);
  });

  it('every new palette keeps clear of the reserved Missing and Unclustered colours', () => {
    for (const recipe of GENERATED_PALETTE_RECIPES) {
      const floor = recipe.candidates.minDistanceToAvoided ?? DEFAULT_CANDIDATE_SPEC.minDistanceToAvoided;
      for (const hex of CATEGORY_PALETTES[recipe.key].colors) {
        expect(ColorMath.deltaE(hex, MISSING_CATEGORY_COLOR)).toBeGreaterThanOrEqual(floor);
        expect(ColorMath.deltaE(hex, UNCLUSTERED_COLOR)).toBeGreaterThanOrEqual(floor);
      }
    }
  });
});
