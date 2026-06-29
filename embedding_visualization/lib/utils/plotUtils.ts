
export interface MarkerStyle {
  size: number;
  opacity: number;
}

/**
 * Calculates optimal marker size and opacity based on the number of points.
 * Uses logarithmic scaling to ensure visualization remains readable across different scales.
 *
 * @param pointCount Number of points in the plot
 * @returns Object containing calculated size and opacity
 */
export function calculateMarkerStyle(pointCount: number): MarkerStyle {
  // Clamp point count to avoid issues with 0 or 1
  const count = Math.max(pointCount, 1);
  const logCount = Math.log10(count);

  // Size calculation
  // Target: ~10px at 20 points, ~3px at 150k points
  // Formula derived from linear interpolation on log scale
  let size = -1.8 * logCount + 12.5;
  size = Math.max(2, Math.min(10, size)); // Clamp between 2 and 10

  // Opacity calculation
  // Target: ~0.8 at 20 points, ~0.15 at 150k points
  let opacity = -0.17 * logCount + 1.0;
  opacity = Math.max(0.15, Math.min(0.8, opacity)); // Clamp between 0.15 and 0.8

  return { size, opacity };
}

export interface LayerOpacity {
  outer: number;
  inner: number;
  core: number;
}

/**
 * Calculate layer opacities based on similarity score.
 * Uses linear mapping from minimum to maximum opacity thresholds.
 *
 * This creates a variable-brightness glow effect where:
 * - Low similarity (0.0): Very faint glow, barely visible
 * - Medium similarity (0.5): Moderate brightness
 * - High similarity (1.0): Full brightness
 *
 * @param similarity Similarity score between 0 and 1
 * @returns Object containing opacity values for outer, inner, and core layers
 */
export function calculateLuminosity(similarity: number): LayerOpacity {
  const MIN_OUTER = 0.05;
  const MAX_OUTER = 0.2;
  const MIN_INNER = 0.1;
  const MAX_INNER = 0.4;
  const MIN_CORE = 0.5;
  const MAX_CORE = 1.0;

  return {
    outer: MIN_OUTER + (similarity * (MAX_OUTER - MIN_OUTER)),
    inner: MIN_INNER + (similarity * (MAX_INNER - MIN_INNER)),
    core: MIN_CORE + (similarity * (MAX_CORE - MIN_CORE)),
  };
}

export interface HighlightScale {
  outerMultiplier: number;
  innerMultiplier: number;
  coreMultiplier: number;
  // For selected point (slightly larger than regular highlights)
  selectedOuterMultiplier: number;
  selectedInnerMultiplier: number;
  selectedCoreMultiplier: number;
}

/**
 * Calculates size multipliers for highlighted points based on total point count.
 * 
 * At low counts (~20 points), base marker size is large (~10px), so modest
 * multipliers suffice. At high counts (~150k points), base size is tiny (~2px),
 * so we need aggressive multipliers to make highlights visible.
 * 
 * Derived from manually-tuned minimum sizes in ScatterPlot3D:
 * - At 150k: outer=10px, inner=6px, core=3px (with base ~2.2px)
 * - At 20: outer=15px, inner=12px, core=9px (with base ~10px)
 * 
 * @param pointCount Total number of points in the plot
 * @returns Multipliers for each glow layer
 */
export function calculateHighlightScale(pointCount: number): HighlightScale {
  const count = Math.max(pointCount, 1);
  const logCount = Math.log10(count);

  // Linear interpolation on log scale from ~20 points (log=1.3) to ~150k (log=5.2)
  // At 20 points:  outer=1.5, inner=1.2, core=0.9
  // At 150k points: outer=4.5, inner=2.7, core=1.4

  const outerMultiplier = 0.77 * logCount + 0.5;    // ~1.5 at 20, ~4.5 at 150k
  const innerMultiplier = 0.38 * logCount + 0.7;    // ~1.2 at 20, ~2.7 at 150k
  const coreMultiplier = 0.13 * logCount + 0.73;    // ~0.9 at 20, ~1.4 at 150k

  // Selected point is slightly larger than regular highlights
  const selectedOuterMultiplier = outerMultiplier * 1.15;
  const selectedInnerMultiplier = innerMultiplier * 1.15;
  const selectedCoreMultiplier = coreMultiplier * 1.1;

  return {
    outerMultiplier: Math.max(1.5, Math.min(5.0, outerMultiplier)),
    innerMultiplier: Math.max(1.2, Math.min(3.5, innerMultiplier)),
    coreMultiplier: Math.max(0.9, Math.min(2.0, coreMultiplier)),
    selectedOuterMultiplier: Math.max(1.7, Math.min(5.5, selectedOuterMultiplier)),
    selectedInnerMultiplier: Math.max(1.4, Math.min(4.0, selectedInnerMultiplier)),
    selectedCoreMultiplier: Math.max(1.0, Math.min(2.2, selectedCoreMultiplier)),
  };
}

/**
 * Parse hex color to RGB components.
 */
function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) throw new Error(`Invalid hex color: ${hex}`);
  return {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16),
  };
}

/**
 * Convert RGB to hex color.
 */
function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
}

/**
 * Interpolate between two hex colors (e.g., '#e8f4ff' and '#fff8e8').
 */
function interpolateHexColor(color1: string, color2: string, factor: number): string {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);

  const r = Math.round(c1.r + (c2.r - c1.r) * factor);
  const g = Math.round(c1.g + (c2.g - c1.g) * factor);
  const b = Math.round(c1.b + (c2.b - c1.b) * factor);

  return rgbToHex(r, g, b);
}

/**
 * Parse rgba color string to components.
 */
function parseRgba(rgba: string): { r: number; g: number; b: number; a: number } {
  const match = rgba.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
  if (!match) throw new Error(`Invalid rgba color: ${rgba}`);
  return {
    r: parseInt(match[1]),
    g: parseInt(match[2]),
    b: parseInt(match[3]),
    a: match[4] ? parseFloat(match[4]) : 1,
  };
}

/**
 * Interpolate between two rgba colors.
 */
function interpolateRgbaColor(color1: string, color2: string, factor: number): string {
  const c1 = parseRgba(color1);
  const c2 = parseRgba(color2);

  const r = Math.round(c1.r + (c2.r - c1.r) * factor);
  const g = Math.round(c1.g + (c2.g - c1.g) * factor);
  const b = Math.round(c1.b + (c2.b - c1.b) * factor);
  const a = c1.a + (c2.a - c1.a) * factor;

  return `rgba(${r}, ${g}, ${b}, ${a.toFixed(2)})`;
}

export interface SimilarityColors {
  coreColor: string;
  glowColor: string;
  outerGlow: string;
}

/**
 * Calculate colors for highlighted points based on similarity score.
 * Interpolates from blue constellation colors (low similarity) to golden colors (high similarity).
 *
 * This creates a visual gradient where:
 * - Low similarity (0.0): Pure blue constellation colors
 * - Medium similarity (0.5): Blended teal/bronze colors
 * - High similarity (1.0): Golden colors matching the selected point
 *
 * The default (dark) palette uses near-white "glowing star" cores, which are invisible
 * against a light background. When `isDark` is false we swap to deeper, more saturated
 * cores/glows so highlighted points stay visible in light mode.
 *
 * @param similarity Similarity score between 0 and 1
 * @param isDark Whether the dark theme is active (default true for backwards compatibility)
 * @returns Color strings for each layer (core, inner glow, outer glow)
 */
export function calculateSimilarityColors(similarity: number, isDark: boolean = true): SimilarityColors {
  if (isDark) {
    // Blue colors (low similarity) — bright glowing cores against a dark sky
    const BLUE_CORE = '#e8f4ff';
    const BLUE_GLOW = 'rgba(170, 200, 235, 0.35)';
    const BLUE_OUTER = 'rgba(140, 180, 230, 0.15)';

    // Golden colors (high similarity - matching selected point)
    const GOLD_CORE = '#fff8e8';
    const GOLD_GLOW = 'rgba(255, 223, 160, 0.35)';
    const GOLD_OUTER = 'rgba(255, 215, 140, 0.15)';

    return {
      coreColor: interpolateHexColor(BLUE_CORE, GOLD_CORE, similarity),
      glowColor: interpolateRgbaColor(BLUE_GLOW, GOLD_GLOW, similarity),
      outerGlow: interpolateRgbaColor(BLUE_OUTER, GOLD_OUTER, similarity),
    };
  }

  // Light mode — gold/near-white washes out on a white background, so we stay in the
  // blue family throughout: a brighter sky blue for low similarity deepening to a strong
  // navy for high similarity. Cores are fully opaque and dark enough to read on white.
  const LOW_CORE = '#0284c7'; // sky-600
  const LOW_GLOW = 'rgba(56, 189, 248, 0.55)'; // sky-400
  const LOW_OUTER = 'rgba(14, 165, 233, 0.22)'; // sky-500

  const HIGH_CORE = '#1e3a8a'; // blue-900
  const HIGH_GLOW = 'rgba(37, 99, 235, 0.55)'; // blue-600
  const HIGH_OUTER = 'rgba(29, 78, 216, 0.22)'; // blue-700

  return {
    coreColor: interpolateHexColor(LOW_CORE, HIGH_CORE, similarity),
    glowColor: interpolateRgbaColor(LOW_GLOW, HIGH_GLOW, similarity),
    outerGlow: interpolateRgbaColor(LOW_OUTER, HIGH_OUTER, similarity),
  };
}
