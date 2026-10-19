/**
 * Recipes for the computationally designed categorical palettes in
 * `categoryPalettes.ts`. Edit a recipe, run `npm run palettes:design`, and paste
 * the printed arrays into the registry; `paletteDesign.test.ts` fails whenever
 * the committed colours and the recipes disagree.
 *
 * Every recipe draws from a dark-first pool: ≥3:1 contrast on the black
 * starfield, chroma high enough never to read as grey, and clear of the
 * reserved Missing / Unclustered colours.
 */

import type { PaletteRecipe } from './paletteDesign';
import { MISSING_CATEGORY_COLOR, UNCLUSTERED_COLOR } from './categoryColors';

const RESERVED = [UNCLUSTERED_COLOR, MISSING_CATEGORY_COLOR] as const;

export const GENERATED_PALETTE_RECIPES: readonly PaletteRecipe[] = [
  {
    // Hubble Deep Field: the whole hue circle, saturated, 64 colours for maps
    // with many topics. Seeded with the default palette's teal and gold so
    // switching from Cosmic Galaxy keeps the two opening colours.
    key: 'deepField',
    seed: ['#0b7285', '#d4a017'],
    count: 64,
    candidates: { lightness: [0.5, 0.9], hueStep: 3, chromaFractions: [0.8, 1.0], minChroma: 0.12, minContrastOnWhite: 1.6, avoid: RESERVED },
  },
  {
    // Polar lights: oxygen greens and limes, teal, then the violet and
    // magenta curtains of a strong display. No reds or oranges.
    key: 'aurora',
    seed: ['#29d975'],
    count: 16,
    candidates: { arcs: [[125, 178], [182, 215], [278, 352]], lightness: [0.55, 0.9], hueStep: 2, chromaFractions: [0.8, 1.0], minChroma: 0.12, minContrastOnWhite: 1.5, avoid: RESERVED },
  },
  {
    // Hubble SHO false colour: sulphur golds, ambers and rusts against the
    // oxygen teals, cyans and sapphire blues of the Pillars of Creation.
    key: 'hubble',
    seed: ['#efb225'],
    count: 16,
    candidates: { arcs: [[38, 96], [192, 262]], lightness: [0.5, 0.92], hueStep: 2, chromaFractions: [0.75, 1.0], minChroma: 0.11, minContrastOnWhite: 1.5, avoid: RESERVED },
  },
  {
    // Pulsar: every colour at the sRGB chroma ceiling and bright enough to
    // glow on black. Electric, for small category counts.
    key: 'pulsar',
    seed: ['#00e9f3'],
    count: 14,
    candidates: { lightness: [0.62, 0.9], hueStep: 2, chromaFractions: [1.0], minChroma: 0.14, minContrastOnWhite: 1.3, avoid: RESERVED },
  },
];
