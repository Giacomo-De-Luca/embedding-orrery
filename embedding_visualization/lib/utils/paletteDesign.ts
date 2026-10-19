/**
 * Computational design of categorical palettes for the scatter plots.
 *
 * `PaletteDesigner` turns a recipe (hue arcs, lightness band, chroma, contrast
 * floors) into a candidate grid in OKLCH and then picks colours greedily by
 * farthest-first traversal (a greedy 2-approximation of max–min dispersion),
 * so that every prefix of the result is well spread. Scatter plots put any two
 * categories side by side, so the objective is the minimum pairwise
 * separation, not adjacency.
 * `PaletteAudit` reports the numbers documented in
 * `documentation/CATEGORICAL_PALETTES.md`.
 *
 * Regenerate the committed palettes with `npm run palettes:design`.
 */

import { ColorMath } from './colorMath';
import type { CvdKind } from './colorMath';

/** Constraints that define the pool of colours a recipe may draw from. */
export interface CandidateSpec {
  /** Allowed hue arcs in OKLCH degrees, `[from, to]` going clockwise; wrap with `to < from` or `to > 360`. Default: the full circle. */
  arcs?: readonly (readonly [number, number])[];
  /** OKLCH lightness range of the grid. */
  lightness?: readonly [number, number];
  lightnessStep?: number;
  hueStep?: number;
  /** Chroma as fractions of the in-gamut maximum at each grid point. */
  chromaFractions?: readonly number[];
  /** Colours below this OKLCH chroma read as grey and are dropped. */
  minChroma?: number;
  minContrastOnBlack?: number;
  minContrastOnWhite?: number;
  /** Colours that must stay visually distinct from every candidate (e.g. reserved Missing / Unclustered). */
  avoid?: readonly string[];
  minDistanceToAvoided?: number;
}

export interface PaletteRecipe {
  /** Key in `CATEGORY_PALETTES`. */
  key: string;
  /** Colours fixed at the start of the palette. */
  seed: readonly string[];
  /** Total number of colours. */
  count: number;
  candidates: CandidateSpec;
}

/** Defaults every recipe field falls back to. */
export const DEFAULT_CANDIDATE_SPEC: Required<CandidateSpec> = {
  arcs: [[0, 360]],
  lightness: [0.5, 0.92],
  lightnessStep: 0.025,
  hueStep: 4,
  chromaFractions: [0.7, 0.85, 1.0],
  minChroma: 0.1,
  minContrastOnBlack: 3.0,
  minContrastOnWhite: 0,
  avoid: [],
  minDistanceToAvoided: 12,
};

const mod360 = (x: number) => ((x % 360) + 360) % 360;
/** Clockwise arc membership: `[300, 20]`, `[300, 380]` and `[0, 360]` all behave as expected. */
const inArc = (hue: number, [from, to]: readonly [number, number]) => {
  const span = to - from;
  return span >= 360 || mod360(hue - from) <= mod360(span);
};

export class PaletteDesigner {
  /**
   * Every in-gamut OKLCH grid point that satisfies the spec, in grid order
   * (lightness outer, then hue, then chroma fraction). Fields set to
   * `undefined` fall back to the defaults. The committed palettes depend on
   * this exact loop nesting: farthest-first breaks ties by pool order.
   */
  static candidates(partial: CandidateSpec = {}): string[] {
    const given = Object.fromEntries(Object.entries(partial).filter(([, v]) => v !== undefined));
    const spec: Required<CandidateSpec> = { ...DEFAULT_CANDIDATE_SPEC, ...given };
    const out = new Set<string>();
    const [lMin, lMax] = spec.lightness;
    const lightnessSteps = Math.round((lMax - lMin) / spec.lightnessStep);
    for (let i = 0; i <= lightnessSteps; i++) {
      const L = lMin + i * spec.lightnessStep;
      for (let hue = 0; hue < 360; hue += spec.hueStep) {
        if (!spec.arcs.some((arc) => inArc(hue, arc))) continue;
        const ceiling = ColorMath.maxChroma(L, hue);
        for (const fraction of spec.chromaFractions) {
          const C = ceiling * fraction;
          if (C < spec.minChroma) continue;
          const hex = ColorMath.oklchToHex(L, C, hue, 0);
          if (!hex) continue;
          if (ColorMath.contrast(hex, '#000000') < spec.minContrastOnBlack) continue;
          if (spec.minContrastOnWhite > 0 && ColorMath.contrast(hex, '#ffffff') < spec.minContrastOnWhite) continue;
          if (spec.avoid.some((reserved) => ColorMath.deltaE(hex, reserved) < spec.minDistanceToAvoided)) continue;
          out.add(hex);
        }
      }
    }
    return [...out];
  }

  /**
   * Greedy farthest-first traversal: repeatedly add the pool colour whose
   * minimum separation from everything chosen so far is largest. Ties keep
   * pool order. Stops early if the pool runs out.
   */
  static farthestFirst(seed: readonly string[], pool: readonly string[], count: number): string[] {
    const chosen = [...seed];
    const nearest = new Map<string, number>();
    for (const hex of pool) {
      if (chosen.includes(hex)) continue;
      nearest.set(hex, chosen.reduce((min, c) => Math.min(min, ColorMath.separation(hex, c)), Infinity));
    }
    while (chosen.length < count && nearest.size > 0) {
      let best: string | null = null;
      let bestDistance = -1;
      for (const [hex, distance] of nearest) {
        if (distance > bestDistance) { bestDistance = distance; best = hex; }
      }
      if (best === null) break;
      chosen.push(best);
      nearest.delete(best);
      for (const [hex, distance] of nearest) {
        const d = ColorMath.separation(hex, best);
        if (d < distance) nearest.set(hex, d);
      }
    }
    return chosen;
  }

  /** Reorder a fixed set greedily so every prefix is well spread, starting from `first` (which must be in the set). */
  static prefixOrder(colors: readonly string[], first: string): string[] {
    if (!colors.includes(first)) throw new Error(`prefixOrder: ${first} is not in the palette`);
    return PaletteDesigner.farthestFirst([first], colors.filter((c) => c !== first), colors.length);
  }

  static generate(recipe: PaletteRecipe): string[] {
    return PaletteDesigner.farthestFirst(recipe.seed, PaletteDesigner.candidates(recipe.candidates), recipe.count);
  }
}

export interface PairStats {
  /** Smallest OKLab distance (×100) under normal vision on the pair list, and that pair. */
  normal: number;
  normalPair: [string, string] | null;
  /** Smallest protan/deutan-simulated distance, that pair and which simulation produced it. */
  cvd: number;
  cvdPair: [string, string] | null;
  cvdKind: CvdKind | null;
}

export interface PrefixStat { k: number; normal: number; cvd: number }

export class PaletteAudit {
  /** Closest pairs under normal and red–green deficient vision; `all` pairs for scatter plots, `adjacent` for legends. */
  static pairStats(palette: readonly string[], pairs: 'all' | 'adjacent' = 'all'): PairStats {
    const n = palette.length;
    const list: [number, number][] = [];
    for (let i = 0; i < n - 1; i++) {
      if (pairs === 'adjacent') list.push([i, i + 1]);
      else for (let j = i + 1; j < n; j++) list.push([i, j]);
    }
    const stats: PairStats = { normal: Infinity, normalPair: null, cvd: Infinity, cvdPair: null, cvdKind: null };
    for (const [i, j] of list) {
      const d = ColorMath.deltaE(palette[i], palette[j]);
      if (d < stats.normal) { stats.normal = d; stats.normalPair = [palette[i], palette[j]]; }
      for (const kind of ['protan', 'deutan'] as const) {
        const dc = ColorMath.deltaE(palette[i], palette[j], kind);
        if (dc < stats.cvd) { stats.cvd = dc; stats.cvdPair = [palette[i], palette[j]]; stats.cvdKind = kind; }
      }
    }
    return stats;
  }

  /** All-pairs minima for the first `k` colours, for each requested `k` that fits. */
  static prefixReport(palette: readonly string[], ks: readonly number[]): PrefixStat[] {
    return ks.filter((k) => k <= palette.length).map((k) => {
      const s = PaletteAudit.pairStats(palette.slice(0, k), 'all');
      return { k, normal: s.normal, cvd: s.cvd };
    });
  }

  static minContrast(palette: readonly string[], surface: string): number {
    return Math.min(...palette.map((hex) => ColorMath.contrast(hex, surface)));
  }

  /** One-line description: OKLCH coordinates and contrast on black / white. */
  static describe(hex: string): string {
    const [L, C, h] = ColorMath.oklch(hex);
    const black = ColorMath.contrast(hex, '#000000').toFixed(2);
    const white = ColorMath.contrast(hex, '#ffffff').toFixed(2);
    return `${hex}  L ${L.toFixed(2)} C ${C.toFixed(3)} h ${h.toFixed(0).padStart(3)}  ${black}:1 on black / ${white}:1 on white`;
  }
}
