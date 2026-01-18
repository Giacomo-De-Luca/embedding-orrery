
import { scaleSequential, scaleDiverging } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';
import { interpolateRgb } from 'd3-interpolate';
import { color } from 'd3-color';


/**
 * Category color system for embedding visualization.
 *
 * Provides:
 * - Preset palettes for known category types (e.g., WordNet POS)
 * - Dynamic color generation for arbitrary categories
 * - Color utility functions
 */

export interface CategoryColorPreset {
  name: string;
  colors: Record<string, string>;
  labels?: Record<string, string>;
}

// ============ Sequential & Diverging Generators ============

// Cache D3 scales by domain to avoid recreation on every render
const sequentialScaleCache = new Map<string, (v: number) => string>();
const divergingScaleCache = new Map<string, (v: number) => string>();
const monochromeScaleCache = new Map<string, (v: number) => string>();

/**
 * Custom blue-gold interpolator (Tableau-style diverging).
 * Deep blue (#1f4e79) → White (#f5f5f5) → Gold (#d4a017)
 */
function interpolateBlueGold(t: number): string {
  if (t < 0.5) {
    // Blue to white (t: 0 → 0.5 maps to 0 → 1)
    return interpolateRgb('#1f4e79', '#f5f5f5')(t * 2);
  } else {
    // White to gold (t: 0.5 → 1 maps to 0 → 1)
    return interpolateRgb('#f5f5f5', '#d4a017')((t - 0.5) * 2);
  }
}

/**
 * Creates a sequential scale (e.g., for probability, density).
 * Maps [min, max] -> Color (Viridis)
 * Cached by domain for performance.
 */
export function getSequentialScale(domain: [number, number] = [0, 1]): (v: number) => string {
  const key = `seq_${domain[0]}_${domain[1]}`;
  if (!sequentialScaleCache.has(key)) {
    sequentialScaleCache.set(key, scaleSequential(interpolateViridis).domain(domain));
  }
  return sequentialScaleCache.get(key)!;
}

/**
 * Creates a diverging scale (e.g., for sentiment, correlation).
 * Maps [min, mid, max] -> Blue → White → Gold (Tableau-style)
 * Cached by domain for performance.
 */
export function getDivergingScale(domain: [number, number, number] = [-1, 0, 1]): (v: number) => string {
  const key = `div_${domain[0]}_${domain[1]}_${domain[2]}`;
  if (!divergingScaleCache.has(key)) {
    divergingScaleCache.set(key, scaleDiverging(interpolateBlueGold).domain(domain));
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


// ============ D3-style category palettes ============

const category10 = [
  '#1f77b4',
  '#ff7f0e',
  '#2ca02c',
  '#d62728',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf',
];

const category20 = [
  '#1f77b4',
  '#aec7e8',
  '#ff7f0e',
  '#ffbb78',
  '#2ca02c',
  '#98df8a',
  '#d62728',
  '#ff9896',
  '#9467bd',
  '#c5b0d5',
  '#8c564b',
  '#c49c94',
  '#e377c2',
  '#f7b6d2',
  '#7f7f7f',
  '#c7c7c7',
  '#bcbd22',
  '#dbdb8d',
  '#17becf',
  '#9edae5',
];

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
 * All known presets, keyed by category field name.
 */
export const CATEGORY_PRESETS: Record<string, CategoryColorPreset> = {
  pos: POS_PRESET,
  part_of_speech: POS_PRESET,
};

// ============ Dynamic color generation ============

/**
 * Generate colors for a given number of categories.
 * Uses D3-style category palettes.
 */
export function generateCategoryColors(count: number): string[] {
  if (count < 1) {
    count = 1;
  }
  if (count <= category10.length) {
    return category10.slice(0, count);
  } else if (count <= category20.length) {
    return category20.slice(0, count);
  } else {
    const colors: string[] = [];
    for (let i = 0; i < count; i++) {
      colors[i] = category20[i % category20.length];
    }
    return colors;
  }
}

/**
 * Build a color map for a set of category values.
 *
 * If a preset exists for the category field, use it.
 * Otherwise, generate colors dynamically.
 *
 * @param categoryField - The name of the category field (e.g., "pos", "topic")
 * @param values - Array of unique category values
 * @returns Object mapping category values to colors
 */
export function buildCategoryColorMap(
  categoryField: string | null,
  values: string[]
): Record<string, string> {
  const colorMap: Record<string, string> = {};

  // Check for preset
  const preset = categoryField ? CATEGORY_PRESETS[categoryField.toLowerCase()] : null;

  if (preset) {
    // Use preset colors, fall back to gray for unknown values
    for (const value of values) {
      colorMap[value] = preset.colors[value] ?? preset.colors['unknown'] ?? '#7f7f7f';
    }
  } else {
    // Generate colors dynamically
    const colors = generateCategoryColors(values.length);
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

// ============ Legacy exports for backwards compatibility ============

/** @deprecated Use buildCategoryColorMap with 'pos' field instead */
export const POS_COLORS = POS_PRESET.colors;

/** @deprecated Use getCategoryLabel with 'pos' field instead */
export const POS_LABELS = POS_PRESET.labels;
