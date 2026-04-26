
import { scaleSequential, scaleDiverging } from 'd3-scale';
import {
  // Sequential
  interpolateSinebow,
  interpolateViridis,
  interpolateCividis,
  interpolateTurbo,
  interpolatePlasma,
  interpolateInferno,
  interpolateMagma,
  interpolateRainbow,
  // Diverging
  interpolateRdBu,
  interpolateSpectral,
  interpolatePiYG,
  interpolatePuOr,
  interpolateBrBG,
} from 'd3-scale-chromatic';
import { interpolateRgb } from 'd3-interpolate';
import { color } from 'd3-color';
import { makeCrameriInterpolator, getCrameriColors, isCrameriScale, crameriGradientCSS } from '../colorMaps/crameriScales';
import type { ColorScale } from '../types/types';
import { CATEGORY_PALETTES, DEFAULT_PALETTE_KEY, getPaletteColors } from './categoryPalettes';


/**
 * Category color system for embedding visualization.
 *
 * Provides:
 * - Preset palettes for known category types (e.g., WordNet POS)
 * - Dynamic color generation for arbitrary categories
 * - Color utility functions
 * - Sequential and diverging color scales with multiple interpolator options
 */

export interface CategoryColorPreset {
  name: string;
  colors: Record<string, string>;
  labels?: Record<string, string>;
}

// ============ Scale Name Types ============

/** D3 built-in sequential scale names */
export type D3SequentialScaleName = 'sinebow' | 'viridis' | 'cividis' | 'turbo' | 'plasma' | 'inferno' | 'magma' | 'rainbow';

/** D3 built-in diverging scale names */
export type D3DivergingScaleName = 'blueGold' | 'rdBu' | 'spectral' | 'piYG' | 'puOr' | 'brBG';

/** Union of all sequential scale names (D3 + Crameri) */
export type SequentialScaleName = D3SequentialScaleName | (string & {});

/** Union of all diverging scale names (D3 + Crameri) */
export type DivergingScaleName = D3DivergingScaleName | (string & {});

// ============ Sequential & Diverging Generators ============

// Cache D3 scales by domain to avoid recreation on every render
const sequentialScaleCache = new Map<string, (v: number) => string>();
const divergingScaleCache = new Map<string, (v: number) => string>();
const monochromeScaleCache = new Map<string, (v: number) => string>();

/**
 * Custom blue-purple-gold interpolator (saturated center, no white).
 * Deep blue (#1f4e79) → Purple (#8B5CF6) → Gold (#d4a017)
 * Better visibility than white center on all backgrounds.
 */
function interpolateBlueGold(t: number): string {
  if (t < 0.5) {
    // Blue to purple (t: 0 → 0.5 maps to 0 → 1)
    return interpolateRgb('#1f4e79', '#8B5CF6')(t * 2);
  } else {
    // Purple to gold (t: 0.5 → 1 maps to 0 → 1)
    return interpolateRgb('#8B5CF6', '#d4a017')((t - 0.5) * 2);
  }
}

// ============ Interpolator Maps ============

const SEQUENTIAL_INTERPOLATORS: Record<D3SequentialScaleName, (t: number) => string> = {
  sinebow: interpolateSinebow,
  rainbow: interpolateRainbow,
  viridis: interpolateViridis,
  cividis: interpolateCividis,
  turbo: interpolateTurbo,
  plasma: interpolatePlasma,
  inferno: interpolateInferno,
  magma: interpolateMagma,
};

const DIVERGING_INTERPOLATORS: Record<D3DivergingScaleName, (t: number) => string> = {
  blueGold: interpolateBlueGold,
  rdBu: interpolateRdBu,
  spectral: interpolateSpectral,
  piYG: interpolatePiYG,
  puOr: interpolatePuOr,
  brBG: interpolateBrBG,
};

/** D3 sequential scale names for UI iteration */
export const D3_SEQUENTIAL_NAMES = Object.keys(SEQUENTIAL_INTERPOLATORS) as D3SequentialScaleName[];

/** D3 diverging scale names for UI iteration */
export const D3_DIVERGING_NAMES = Object.keys(DIVERGING_INTERPOLATORS) as D3DivergingScaleName[];

/**
 * Creates a sequential scale (e.g., for probability, density).
 * Maps [min, max] -> Color using the specified interpolator.
 * Supports both D3 built-in scales and Crameri scales (from cache).
 * Cached by domain + scale name for performance.
 */
export function getSequentialScale(
  domain: [number, number] = [0, 1],
  scaleName: SequentialScaleName = 'sinebow'
): (v: number) => string {
  const key = `seq_${domain[0]}_${domain[1]}_${scaleName}`;
  if (!sequentialScaleCache.has(key)) {
    const d3Interpolator = SEQUENTIAL_INTERPOLATORS[scaleName as D3SequentialScaleName];
    if (d3Interpolator) {
      sequentialScaleCache.set(key, scaleSequential(d3Interpolator).domain(domain));
    } else {
      // Try Crameri interpolator from cache
      const crameriInterp = makeCrameriInterpolator(scaleName);
      if (crameriInterp) {
        sequentialScaleCache.set(key, scaleSequential(crameriInterp).domain(domain));
      } else {
        // Fallback to viridis if Crameri not loaded yet
        sequentialScaleCache.set(key, scaleSequential(interpolateViridis).domain(domain));
      }
    }
  }
  return sequentialScaleCache.get(key)!;
}

/**
 * Creates a diverging scale (e.g., for sentiment, correlation).
 * Maps [min, mid, max] -> Color using the specified interpolator.
 * Supports both D3 built-in scales and Crameri scales (from cache).
 * Cached by domain + scale name for performance.
 */
export function getDivergingScale(
  domain: [number, number, number] = [-1, 0, 1],
  scaleName: DivergingScaleName = 'blueGold'
): (v: number) => string {
  const key = `div_${domain[0]}_${domain[1]}_${domain[2]}_${scaleName}`;
  if (!divergingScaleCache.has(key)) {
    const d3Interpolator = DIVERGING_INTERPOLATORS[scaleName as D3DivergingScaleName];
    if (d3Interpolator) {
      divergingScaleCache.set(key, scaleDiverging(d3Interpolator).domain(domain));
    } else {
      // Try Crameri interpolator from cache
      const crameriInterp = makeCrameriInterpolator(scaleName);
      if (crameriInterp) {
        divergingScaleCache.set(key, scaleDiverging(crameriInterp).domain(domain));
      } else {
        // Fallback to blueGold if Crameri not loaded yet
        divergingScaleCache.set(key, scaleDiverging(interpolateBlueGold).domain(domain));
      }
    }
  }
  return divergingScaleCache.get(key)!;
}

/**
 * Creates a monochrome scale (single-color opacity gradient).
 * Low values fade out (10% opacity), high values solid (100%).
 * Cached by domain + baseColor for performance.
 */
export function getMonochromeScale(
  baseColor: string = '#1f77b4',
  domain: [number, number] = [0, 1]
): (v: number) => string {
  const key = `mono_${baseColor}_${domain[0]}_${domain[1]}`;
  if (!monochromeScaleCache.has(key)) {
    const rgb = color(baseColor)?.rgb();
    if (!rgb) {
      // Fallback to blue if color parsing fails
      monochromeScaleCache.set(key, (t: number) => `rgba(31, 119, 180, ${0.1 + t * 0.9})`);
    } else {
      monochromeScaleCache.set(key, scaleSequential((t: number) => {
        const opacity = 0.1 + t * 0.9; // 10% → 100%
        return `rgba(${Math.round(rgb.r)}, ${Math.round(rgb.g)}, ${Math.round(rgb.b)}, ${opacity})`;
      }).domain(domain));
    }
  }
  return monochromeScaleCache.get(key)!;
}

/**
 * Clear scale caches (useful if memory is a concern).
 */
export function clearScaleCaches(): void {
  sequentialScaleCache.clear();
  divergingScaleCache.clear();
  monochromeScaleCache.clear();
}

/**
 * Generate a CSS linear-gradient string from a scale function.
 * Useful for rendering scale previews in Legend or ColorScaleSelector.
 */
export function generateGradientCSS(
  scaleFunc: (t: number) => string,
  steps: number = 10
): string {
  const colors = Array.from({ length: steps }, (_, i) => scaleFunc(i / (steps - 1)));
  return `linear-gradient(to right, ${colors.join(', ')})`;
}


/**
 * Generate a CSS linear-gradient string for any continuous ColorScale.
 * Single source of truth — used by Legend and ColorScaleSelector.
 */
export function colorScaleGradientCSS(colorScale: ColorScale, steps: number = 20): string {
  if (colorScale.type === 'categorical') return '';
  const scaleName = (colorScale.type === 'sequential' || colorScale.type === 'diverging')
    ? colorScale.scaleName : undefined;
  if (scaleName && isCrameriScale(scaleName)) {
    const crameri = crameriGradientCSS(scaleName, steps);
    if (crameri) return crameri;
  }
  let scaleFunc: (t: number) => string;
  if (colorScale.type === 'monochrome') {
    scaleFunc = getMonochromeScale(colorScale.baseColor, [0, 1]);
  } else if (colorScale.type === 'diverging') {
    scaleFunc = getDivergingScale([0, 0.5, 1], colorScale.scaleName);
  } else {
    scaleFunc = getSequentialScale([0, 1], colorScale.scaleName);
  }
  return generateGradientCSS(scaleFunc, steps);
}

/**
 * Get a (t: number) => color interpolator for any continuous ColorScale.
 * Used by NumericRangeChart to color individual histogram bars.
 */
export function colorScaleInterpolator(colorScale: ColorScale): ((t: number) => string) | null {
  if (colorScale.type === 'categorical') return null;
  if (colorScale.type === 'monochrome') return getMonochromeScale(colorScale.baseColor, [0, 1]);
  const name = colorScale.scaleName;
  if (isCrameriScale(name)) {
    const interp = makeCrameriInterpolator(name);
    if (interp) return interp;
  }
  if (colorScale.type === 'diverging') return getDivergingScale([0, 0.5, 1], name);
  return getSequentialScale([0, 1], name);
}

/**
 * Generate a CSS linear-gradient string for any continuous ColorScale.
<<<<<<< HEAD
 * Single source of truth used by both Legend and ColorScaleSelector.
 * Returns empty string for categorical scales.
 */
export function colorScaleGradientCSS(colorScale: ColorScale, steps: number = 20): string {
  if (colorScale.type === 'categorical') return '';

  // Crameri fast-path (pre-computed arrays)
=======
 * Single source of truth — used by Legend and ColorScaleSelector.
 */
export function colorScaleGradientCSS(colorScale: ColorScale, steps: number = 20): string {
  if (colorScale.type === 'categorical') return '';
>>>>>>> backendMigration
  const scaleName = (colorScale.type === 'sequential' || colorScale.type === 'diverging')
    ? colorScale.scaleName : undefined;
  if (scaleName && isCrameriScale(scaleName)) {
    const crameri = crameriGradientCSS(scaleName, steps);
    if (crameri) return crameri;
  }
<<<<<<< HEAD

  // D3 / monochrome fallback
=======
>>>>>>> backendMigration
  let scaleFunc: (t: number) => string;
  if (colorScale.type === 'monochrome') {
    scaleFunc = getMonochromeScale(colorScale.baseColor, [0, 1]);
  } else if (colorScale.type === 'diverging') {
    scaleFunc = getDivergingScale([0, 0.5, 1], colorScale.scaleName);
  } else {
    scaleFunc = getSequentialScale([0, 1], colorScale.scaleName);
  }
  return generateGradientCSS(scaleFunc, steps);
}

<<<<<<< HEAD
=======
/**
 * Get a (t: number) => color interpolator for any continuous ColorScale.
 * Used by NumericRangeChart to color individual histogram bars.
 */
export function colorScaleInterpolator(colorScale: ColorScale): ((t: number) => string) | null {
  if (colorScale.type === 'categorical') return null;
  if (colorScale.type === 'monochrome') return getMonochromeScale(colorScale.baseColor, [0, 1]);
  const name = colorScale.scaleName;
  if (isCrameriScale(name)) {
    const interp = makeCrameriInterpolator(name);
    if (interp) return interp;
  }
  if (colorScale.type === 'diverging') return getDivergingScale([0, 0.5, 1], name);
  return getSequentialScale([0, 1], name);
}

>>>>>>> backendMigration
// Re-export palette registry for convenience
export { CATEGORY_PALETTES, BUILTIN_PALETTE_NAMES, DEFAULT_PALETTE_KEY, getPaletteColors } from './categoryPalettes';
export type { CategoricalPalette } from './categoryPalettes';


// ============ Preset palettes for known category types ============

/**
 * WordNet Part-of-Speech colors.
 * Original palette from the project.
 */
export const POS_PRESET: CategoryColorPreset = {
  name: 'Part of Speech',
  colors: {
    n: '#1f77b4', // noun - blue
    v: '#ff7f0e', // verb - orange
    a: '#2ca02c', // adjective - green
    r: '#d62728', // adverb - red
    s: '#9467bd', // adjective satellite - purple
    unknown: '#7f7f7f', // unknown - gray
  },
  labels: {
    n: 'Noun',
    v: 'Verb',
    a: 'Adjective',
    r: 'Adverb',
    s: 'Adj. Satellite',
    unknown: 'Unknown',
  },
};

/**
 * Fixed color for unclustered/noise points (HDBSCAN label -1).
 * Distinct from topic palette colors, clearly readable as "uncategorized".
 */
export const UNCLUSTERED_COLOR = '#e416be';

/**
 * Topic clustering colors.
 * Blue for unclustered/noise points (HDBSCAN label -1).
 */
export const TOPIC_PRESET: CategoryColorPreset = {
  name: 'Topic',
  colors: {
    '-1': UNCLUSTERED_COLOR,
    'Unclustered': UNCLUSTERED_COLOR,
    'unclustered': UNCLUSTERED_COLOR,
  },
  labels: {
    '-1': 'Unclustered',
  },
};

/**
 * All known presets, keyed by category field name.
 */
export const CATEGORY_PRESETS: Record<string, CategoryColorPreset> = {
  pos: POS_PRESET,
  part_of_speech: POS_PRESET,
  topic: TOPIC_PRESET,
  topic_id: TOPIC_PRESET,
  topic_label: TOPIC_PRESET,
};

// ============ Dynamic color generation ============

/**
 * Generate colors for a given number of categories.
 * Uses D3-style category palettes by default, or a Crameri categorical palette
 * (100 distinct colors) if a palette name is provided.
 */
export function generateCategoryColors(count: number, palette?: string): string[] {
  if (count < 1) count = 1;

  // Resolve color array: built-in registry first, then Crameri, then default
  let colors: readonly string[] | null = null;

  if (palette) {
    colors = getPaletteColors(palette) ?? getCrameriColors(palette) ?? null;
  }

  if (!colors) {
    colors = CATEGORY_PALETTES[DEFAULT_PALETTE_KEY].colors;
  }

  if (count <= colors.length) {
    return colors.slice(0, count) as string[];
  }

  const result: string[] = [];
  for (let i = 0; i < count; i++) {
    result[i] = colors[i % colors.length] as string;
  }
  return result;
}

/**
 * Build a color map for a set of category values.
 *
 * If a preset exists for the category field, use it.
 * Otherwise, generate colors dynamically.
 *
 * @param categoryField - The name of the category field (e.g., "pos", "topic")
 * @param values - Array of unique category values
 * @param palette - Optional Crameri categorical palette name (e.g., "batlowS")
 * @returns Object mapping category values to colors
 */
export function buildCategoryColorMap(
  categoryField: string | null,
  values: string[],
  palette?: string,
): Record<string, string> {
  const colorMap: Record<string, string> = {};

  // Check for preset
  const preset = categoryField ? CATEGORY_PRESETS[categoryField.toLowerCase()] : null;

  if (preset) {
    // Separate values into those with preset colors and those without
    const valuesWithoutPreset: string[] = [];

    for (const value of values) {
      if (preset.colors[value]) {
        // Use preset color if available
        colorMap[value] = preset.colors[value];
      } else {
        // Queue for dynamic color generation
        valuesWithoutPreset.push(value);
      }
    }

    // Generate dynamic colors for values not in preset
    if (valuesWithoutPreset.length > 0) {
      const dynamicColors = generateCategoryColors(valuesWithoutPreset.length, palette);
      for (let i = 0; i < valuesWithoutPreset.length; i++) {
        colorMap[valuesWithoutPreset[i]] = dynamicColors[i];
      }
    }
  } else {
    // Generate colors dynamically
    const colors = generateCategoryColors(values.length, palette);
    for (let i = 0; i < values.length; i++) {
      colorMap[values[i]] = colors[i];
    }
  }

  return colorMap;
}

/**
 * Get human-readable label for a category value.
 *
 * @param categoryField - The name of the category field
 * @param value - The category value
 * @returns Human-readable label or the original value
 */
export function getCategoryLabel(categoryField: string | null, value: string): string {
  const preset = categoryField ? CATEGORY_PRESETS[categoryField.toLowerCase()] : null;
  return preset?.labels?.[value] ?? value;
}

/**
 * Get the display name for a category field.
 *
 * @param categoryField - The name of the category field
 * @returns Human-readable name for the category
 */
export function getCategoryDisplayName(categoryField: string | null): string {
  if (!categoryField) return 'Category';
  const preset = CATEGORY_PRESETS[categoryField.toLowerCase()];
  if (preset) return preset.name;
  // Convert snake_case/camelCase to Title Case
  return categoryField
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ============ Color manipulation utilities ============

/**
 * Desaturate a hex color by blending it toward a neutral gray.
 * @param hex - Hex color string (e.g. '#ff6600' or 'ff6600')
 * @param amount - Blend amount: 0 = original color, 1 = fully neutral (default 0.3)
 * @param isDark - Theme mode: blends toward light gray in dark mode, dark gray in light mode
 * @returns CSS rgb() string
 */
export function desaturateHex(hex: string, amount: number = 0.3, isDark: boolean = false): string {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.substring(0, 2), 16);
  const g = parseInt(clean.substring(2, 4), 16);
  const b = parseInt(clean.substring(4, 6), 16);
  const neutral = isDark ? 0xc0 : 0x60;
  const keep = 1 - amount;
  return `rgb(${Math.round(r * keep + neutral * amount)},${Math.round(g * keep + neutral * amount)},${Math.round(b * keep + neutral * amount)})`;
}
