/**
 * Crameri Scientific Colormaps - Lazy loading adapter module.
 *
 * Provides perceptually-uniform colormaps by Fabio Crameri for use with
 * Plotly.js and D3.js. Only the index (~2KB) is bundled; individual
 * colormap JSON files (~8-12KB each) are loaded on demand.
 *
 * Reference: Crameri, F. (2023). Scientific colour maps (Version 8.0).
 * Zenodo. https://doi.org/10.5281/zenodo.8035877
 */

import index from './colormaps/index.json';

// ============ Types ============

interface CrameriColormapData {
  name: string;
  type: string;
  colors: string[];
  plotly: [number, string][];
}

type CrameriIndex = Record<string, { type: string; numColors: number }>;

// ============ Static Metadata from index.json ============

const crameriIndex = index as CrameriIndex;

// Curated selection: 3 most colorful per category, batlow as base

/** Curated Crameri sequential colormaps (3) */
export const CRAMERI_SEQUENTIAL_NAMES = ['batlow', 'hawaii', 'lipari'];

/** Custom color-column colormaps (Hilbert-ordered strips for direct color visualization) */
export const COLOR_STRIP_NAMES = ['hilbertColor', 'hueSatColor', 'xkcdColor', 'rainbow'];

/** Curated Crameri diverging colormaps (3) */
export const CRAMERI_DIVERGING_NAMES = ['managua', 'berlin', 'roma'];

/** Curated Crameri categorical colormaps (3, S-suffix, 100 colors each) */
export const CRAMERI_CATEGORICAL_NAMES = ['batlowS', 'hawaiiS', 'lipariS'];

/** Set of all Crameri colormap names for fast lookup */
const ALL_CRAMERI_NAMES = new Set(Object.keys(crameriIndex));

// ============ Human-Readable Labels ============

function formatLabel(name: string): string {
  // Known special cases
  const special: Record<string, string> = {
    lajolla: 'La Jolla',
    lapaz: 'La Paz',
    grayC: 'Gray C',
    batlowK: 'Batlow K',
    batlowW: 'Batlow W',
    batlowS: 'Batlow S',
    batlowKS: 'Batlow KS',
    batlowWS: 'Batlow WS',
  };
  const colorStrip: Record<string, string> = {
    hilbertColor: 'Hilbert RGB',
    hueSatColor: 'Hue × Saturation',
    xkcdColor: 'XKCD Survey',
    rainbow: 'Rainbow',
  };
  if (colorStrip[name]) return colorStrip[name];
  if (special[name]) return special[name];
  // For S-suffix categorical: strip S, capitalize, add " S"
  if (name.endsWith('S') && name.length > 1) {
    const base = name.slice(0, -1);
    return `${formatLabel(base)} S`;
  }
  // Default: capitalize first letter
  return name.charAt(0).toUpperCase() + name.slice(1);
}

export const CRAMERI_SEQUENTIAL_LABELS: Record<string, string> = Object.fromEntries(
  CRAMERI_SEQUENTIAL_NAMES.map(n => [n, formatLabel(n)])
);

export const COLOR_STRIP_LABELS: Record<string, string> = Object.fromEntries(
  COLOR_STRIP_NAMES.map(n => [n, formatLabel(n)])
);

export const CRAMERI_DIVERGING_LABELS: Record<string, string> = Object.fromEntries(
  CRAMERI_DIVERGING_NAMES.map(n => [n, formatLabel(n)])
);

export const CRAMERI_CATEGORICAL_LABELS: Record<string, string> = Object.fromEntries(
  CRAMERI_CATEGORICAL_NAMES.map(n => [n, formatLabel(n)])
);

// ============ Lazy Loading ============

/** Cache of loaded colormap data */
const cache = new Map<string, CrameriColormapData>();

/** Pending load promises to deduplicate concurrent requests */
const pending = new Map<string, Promise<CrameriColormapData>>();

/**
 * Dynamically load a Crameri colormap JSON file.
 * Uses dynamic import() with a Map cache to avoid re-fetching.
 */
export async function loadCrameriColormap(name: string): Promise<CrameriColormapData> {
  if (cache.has(name)) return cache.get(name)!;
  if (pending.has(name)) return pending.get(name)!;

  const promise = (async () => {
    try {
      // Dynamic import for code-splitting - each JSON is a separate chunk
      const mod = await import(`./colormaps/${name}.json`);
      const data = (mod.default ?? mod) as CrameriColormapData;
      cache.set(name, data);
      pending.delete(name);
      return data;
    } catch (e) {
      pending.delete(name);
      throw new Error(`Failed to load Crameri colormap "${name}": ${e}`);
    }
  })();

  pending.set(name, promise);
  return promise;
}

/**
 * Preload multiple Crameri colormaps in parallel.
 */
export async function preloadCrameriColormaps(names: string[]): Promise<void> {
  await Promise.all(names.map(n => loadCrameriColormap(n)));
}

// ============ Sync Accessors (read from cache) ============

/**
 * Get the pre-computed Plotly colorscale array for a Crameri colormap.
 * Returns the 256-step `[number, string][]` array directly from the JSON.
 * Returns null if not yet loaded.
 */
export function getCrameriPlotlyScale(name: string): [number, string][] | null {
  const data = cache.get(name);
  return data?.plotly ?? null;
}

/**
 * Get the raw color array for a Crameri colormap.
 * Returns null if not yet loaded.
 */
export function getCrameriColors(name: string): string[] | null {
  const data = cache.get(name);
  return data?.colors ?? null;
}

/**
 * Check if a colormap is already loaded into the cache.
 */
export function isCrameriLoaded(name: string): boolean {
  return cache.has(name);
}

// ============ Interpolator Factory ============

/**
 * Create an interpolator function `(t: number) => string` from a Crameri colormap.
 * Reads from cache. Returns null if not loaded.
 */
export function makeCrameriInterpolator(name: string): ((t: number) => string) | null {
  const data = cache.get(name);
  if (!data) return null;

  const { colors } = data;
  const maxIdx = colors.length - 1;

  return (t: number): string => {
    const clamped = Math.max(0, Math.min(1, t));
    return colors[Math.round(clamped * maxIdx)];
  };
}

// ============ CSS Gradient ============

/**
 * Generate a CSS linear-gradient from a Crameri colormap.
 * Returns null if the colormap is not loaded.
 */
export function crameriGradientCSS(name: string, steps: number = 20): string | null {
  const data = cache.get(name);
  if (!data) return null;

  const { colors } = data;
  const maxIdx = colors.length - 1;
  const sampled = Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1);
    return colors[Math.round(t * maxIdx)];
  });

  return `linear-gradient(to right, ${sampled.join(', ')})`;
}

// ============ Type Guard ============

/**
 * Check whether a scale name is a Crameri colormap (not a D3 scale).
 */
export function isCrameriScale(name: string): boolean {
  return ALL_CRAMERI_NAMES.has(name);
}

/**
 * Get the type of a Crameri colormap ('sequential', 'diverging', 'multi-sequential', 'categorical', 'cyclic').
 */
export function getCrameriType(name: string): string | null {
  return crameriIndex[name]?.type ?? null;
}
