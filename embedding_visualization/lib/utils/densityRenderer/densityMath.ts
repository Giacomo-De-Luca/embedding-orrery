/**
 * Pure math for the 2D density overlay — port of the corresponding pieces of
 * Apple's embedding-atlas (MIT licensed):
 *   - Matrix3 helpers            (packages/component/src/lib/matrix.ts)
 *   - approximateMaxDensity2D    (embedding_view/statistics.ts + EmbeddingView.svelte)
 *   - computeViewingParams       (embedding_view/EmbeddingViewImpl.svelte viewingParameters)
 *   - colorMatrix construction   (webgl2_renderer/renderer.ts)
 *
 * No WebGL here — everything is unit-testable. See README.md in this folder.
 */

import type { AxisRanges } from '../../../app/utils/labelPlacement2D';

/**
 * 3×3 matrix, embedding-atlas layout: 9 elements, columns first — the
 * translation lives at indices 6 and 7. GLSL `matrix * vec3(x, y, 1)` with
 * this layout gives clipX = m0·x + m3·y + m6, clipY = m1·x + m4·y + m7.
 */
export type Matrix3 = number[];

export const MATRIX3_IDENTITY: Matrix3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** m1 · m2 (apply m2 first, then m1). Port of matrix3_matrix_mul_matrix. */
export function matrix3Multiply(m1: Matrix3, m2: Matrix3): Matrix3 {
  return [
    m1[0] * m2[0] + m1[3] * m2[1] + m1[6] * m2[2],
    m1[1] * m2[0] + m1[4] * m2[1] + m1[7] * m2[2],
    m1[2] * m2[0] + m1[5] * m2[1] + m1[8] * m2[2],
    m1[0] * m2[3] + m1[3] * m2[4] + m1[6] * m2[5],
    m1[1] * m2[3] + m1[4] * m2[4] + m1[7] * m2[5],
    m1[2] * m2[3] + m1[5] * m2[4] + m1[8] * m2[5],
    m1[0] * m2[6] + m1[3] * m2[7] + m1[6] * m2[8],
    m1[1] * m2[6] + m1[4] * m2[7] + m1[7] * m2[8],
    m1[2] * m2[6] + m1[5] * m2[7] + m1[8] * m2[8],
  ];
}

/** Apply the matrix to a 2D point (w = 1), returning [x', y']. */
export function applyMatrix3(m: Matrix3, x: number, y: number): [number, number] {
  return [m[0] * x + m[3] * y + m[6], m[1] * x + m[4] * y + m[7]];
}

/**
 * Data→clip matrix for a canvas covering exactly the Plotly plot area:
 * xRange maps to clip [-1, 1] and yRange to clip [-1, 1] (Plotly's y-up
 * matches clip-space y-up, so no flip).
 */
export function buildPositionMatrix(ranges: AxisRanges): Matrix3 {
  const xSpan = ranges.xRange[1] - ranges.xRange[0] || 1;
  const ySpan = ranges.yRange[1] - ranges.yRange[0] || 1;
  const sx = 2 / xSpan;
  const sy = 2 / ySpan;
  const cx = (ranges.xRange[0] + ranges.xRange[1]) / 2;
  const cy = (ranges.yRange[0] + ranges.yRange[1]) / 2;
  return [sx, 0, 0, 0, sy, 0, -cx * sx, -cy * sy, 1];
}

// ---------------------------------------------------------------------------
// Max density estimation (statistics.ts port)
// ---------------------------------------------------------------------------

function median(values: Float32Array): number {
  const n = values.length;
  if (n === 0) return 0;
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.floor(n / 2)];
}

function stdev(values: Float32Array): number {
  const n = values.length;
  if (n === 0) return 0;
  // Shift by the first value to avoid catastrophic cancellation.
  const shift = values[0];
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - shift;
    sum += d;
    sumSq += d * d;
  }
  const m = sum / n;
  const variance = sumSq / n - m * m;
  return Math.sqrt(variance > 0 ? variance : 0);
}

const DENSITY_GRID = 256;

/**
 * Approximate peak point density in points per data-unit², via a 256×256 bin
 * grid centered on the medians with binWidth = 0.3 · max(std, 1e-3). Points
 * outside the grid are sparse outliers and are ignored. Combines
 * embedding-atlas's approximateDensity2D with its caller's binWidth choice
 * (scaler = 1 / (3·max(std, 1e-3)); binWidth = 0.1 / scaler).
 */
export function approximateMaxDensity2D(x: Float32Array, y: Float32Array): number {
  if (x.length === 0) return 0;
  const xCenter = median(x);
  const yCenter = median(y);
  const binWidth = 0.3 * Math.max(stdev(x), stdev(y), 1e-3);

  const g = DENSITY_GRID;
  const h = g >> 1;
  const grid = new Int32Array(g * g);
  let maxCount = 0;
  for (let i = 0; i < x.length; i++) {
    const bx = Math.floor((x[i] - xCenter) / binWidth) + h;
    const by = Math.floor((y[i] - yCenter) / binWidth) + h;
    if (bx < 0 || bx >= g || by < 0 || by >= g) continue;
    const v = ++grid[by * g + bx];
    if (v > maxCount) maxCount = v;
  }
  return maxCount / (binWidth * binWidth);
}

// ---------------------------------------------------------------------------
// Zoom-adaptive viewing parameters (viewingParameters port)
// ---------------------------------------------------------------------------

export interface ViewingParams {
  /** Scales blurred point counts into the [0, 1] density range. */
  densityScaler: number;
  /** Global alpha for the density-map paint (0 = fully zoomed in). */
  densityAlpha: number;
  /** Global alpha for the contour paint (mirrors densityAlpha). */
  contoursAlpha: number;
}

/**
 * Default multiplier on Apple's density scaler (user-tunable via the Density
 * intensity slider). Their 0.2 constant caps the densest region at ~20% band
 * opacity — tuned for a renderer that draws its own points into the same
 * buffer; over Plotly's independently-drawn points that reads as nearly
 * invisible. Too high and every lone point's blur crosses several
 * quantization bands, ringing each point with contour circles.
 */
export const DENSITY_INTENSITY = 1.5;

/**
 * Ceilings on the overlay alphas at full crossfade, so the density fill never
 * fully hides the points beneath it and the contour lines stay hairline
 * rather than merging into thick borders.
 */
export const DENSITY_MAX_ALPHA = 0.75;
export const DENSITY_CONTOUR_ALPHA = 0.5;

/**
 * Zoom-in factors (relative to the initial full view) where the density
 * crossfade starts and finishes. Apple fades against an absolute density
 * constant (1/16 pts per px²) tuned for multi-million-point datasets — on
 * collections of 1k-150k points that leaves the overlay near-invisible even
 * fully zoomed out. Anchoring to relative zoom keeps the intent (density at
 * overview, points close-up) while behaving at any dataset size.
 */
export const DENSITY_FADE_START_ZOOM = 3;
export const DENSITY_FADE_END_ZOOM = 8;

/**
 * Reformulation of Apple's viewingParameters in axis-span form (we have
 * Plotly axis ranges, not a viewport scale).
 *
 * The density scaler follows Apple exactly (pinned by a unit test): their
 * convention measures density against viewDimension = max(w, h) with
 * dataPerCssPx = 2 / (scale · viewDim) — exactly 4× smaller than the true
 * per-CSS-px² density. The /4 below keeps their tuned 0.2 constant meaning
 * the same thing here. Because the scaler grows as spans shrink while per-px
 * counts drop in step, band brightness stays constant across zoom.
 *
 * The crossfade alpha deliberately diverges: it fades on zoom-in relative to
 * the initial view (DENSITY_FADE_START_ZOOM → DENSITY_FADE_END_ZOOM,
 * log-space linear) instead of Apple's absolute density threshold — see the
 * constant's doc comment. Without initial spans the overlay stays at alpha 1.
 */
export function computeViewingParams(args: {
  maxDensity: number;
  xSpan: number;
  ySpan: number;
  /** Full-view spans anchoring the zoom crossfade (omit → no fade). */
  initialXSpan?: number;
  initialYSpan?: number;
  plotWidthCss: number;
  plotHeightCss: number;
  dpr: number;
  /** Density brightness multiplier; 1 = Apple's original tuning. */
  intensity?: number;
}): ViewingParams {
  const {
    maxDensity, xSpan, ySpan, initialXSpan, initialYSpan,
    plotWidthCss, plotHeightCss, dpr, intensity = DENSITY_INTENSITY,
  } = args;

  const exactCssDensity = maxDensity * (xSpan / plotWidthCss) * (ySpan / plotHeightCss);
  const maxPointDensity = exactCssDensity / 4;
  const maxPixelDensity = maxPointDensity / (dpr * dpr);

  const densityScaler = maxPixelDensity > 0 ? (intensity * 0.2) / maxPixelDensity : 0;

  let fade = 1;
  if (initialXSpan && initialYSpan && xSpan > 0 && ySpan > 0) {
    const zoom = Math.sqrt((initialXSpan / xSpan) * (initialYSpan / ySpan));
    const t =
      (Math.log(DENSITY_FADE_END_ZOOM) - Math.log(zoom)) /
      (Math.log(DENSITY_FADE_END_ZOOM) - Math.log(DENSITY_FADE_START_ZOOM));
    fade = Math.min(Math.max(t, 0), 1);
  }

  return {
    densityScaler,
    densityAlpha: fade * DENSITY_MAX_ALPHA,
    contoursAlpha: fade * DENSITY_CONTOUR_ALPHA,
  };
}

// ---------------------------------------------------------------------------
// Category → channel assignment
// ---------------------------------------------------------------------------

/** WizMap-style neutral density tints for the muted (single-channel) mode. */
export const MUTED_DENSITY_TINT = {
  light: '#8fb3cc',
  dark: '#64748b',
} as const;

export interface ActiveCategory {
  name: string;
  count: number;
  color: string;
}

export interface ChannelAssignment {
  mode: 'categorical' | 'meanColor' | 'muted';
  /** Category name → channel index 0-3. Only set in categorical mode. */
  channelOf: Map<string, number>;
  /** Hex color per channel (categorical: ≤4; muted: the single tint). */
  channelColors: string[];
  /** Category name → hex color. Only set in meanColor mode. */
  colorOfCategory?: Map<string, string>;
}

/**
 * Pick the density coloring mode:
 * - 1-4 active categories on a categorical field → 'categorical': exact
 *   per-category density via the count buffer's 4 one-hot RGBA channels
 *   (channel order stable: count desc, then name asc);
 * - >4 categories → 'meanColor': each point splats its category color, so
 *   bands and contour lines take the locally dominant cluster's color;
 * - numeric / no color field / no categories → 'muted' single neutral tint.
 */
export function assignCategoryChannels(
  activeCategories: ActiveCategory[],
  isCategoricalField: boolean,
  isDark: boolean,
): ChannelAssignment {
  if (isCategoricalField && activeCategories.length >= 1 && activeCategories.length <= 4) {
    const ordered = [...activeCategories].sort(
      (a, b) => b.count - a.count || a.name.localeCompare(b.name),
    );
    return {
      mode: 'categorical',
      channelOf: new Map(ordered.map((c, i) => [c.name, i])),
      channelColors: ordered.map((c) => c.color),
    };
  }
  if (isCategoricalField && activeCategories.length > 4) {
    return {
      mode: 'meanColor',
      channelOf: new Map(),
      channelColors: [],
      colorOfCategory: new Map(activeCategories.map((c) => [c.name, c.color])),
    };
  }
  return {
    mode: 'muted',
    channelOf: new Map(),
    channelColors: [isDark ? MUTED_DENSITY_TINT.dark : MUTED_DENSITY_TINT.light],
  };
}

/**
 * Linearize a hex color into 3 bytes of gamma-decoded RGB for the mean-color
 * point buffer (the pipeline blends in linear space; the final gamma pass
 * re-encodes).
 */
export function linearizedColorBytes(hex: string, gamma: number): [number, number, number] {
  const { r, g, b } = parseHexNormalizedRgb(hex);
  return [
    Math.round(Math.pow(r, gamma) * 255),
    Math.round(Math.pow(g, gamma) * 255),
    Math.round(Math.pow(b, gamma) * 255),
  ];
}

// ---------------------------------------------------------------------------
// Color matrix (renderer.ts port)
// ---------------------------------------------------------------------------

function parseHexNormalizedRgb(hex: string): { r: number; g: number; b: number } {
  let s = hex.replace('#', '');
  if (s.length === 3) s = s.split('').map((c) => c + c).join('');
  const v = parseInt(s.slice(0, 6), 16);
  if (Number.isNaN(v)) return { r: 0.5, g: 0.5, b: 0.5 };
  return { r: ((v >> 16) & 0xff) / 255, g: ((v >> 8) & 0xff) / 255, b: (v & 0xff) / 255 };
}

/**
 * 4×RGBA floats for the paint shaders: each channel's color linearized via
 * pow(c, gamma); unused channels padded with neutral gray, matching
 * embedding-atlas's densityRenderCommand.
 */
export function buildColorMatrix(channelColors: string[], gamma: number): number[] {
  const matrix: number[] = [];
  for (let i = 0; i < 4; i++) {
    if (i < channelColors.length) {
      const { r, g, b } = parseHexNormalizedRgb(channelColors[i]);
      matrix.push(Math.pow(r, gamma), Math.pow(g, gamma), Math.pow(b, gamma), 1);
    } else {
      matrix.push(0.5, 0.5, 0.5, 1);
    }
  }
  return matrix;
}
