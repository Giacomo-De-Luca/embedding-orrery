import { calculateMarkerStyle } from './plotUtils';

/** Extent of the 2D plot's points, in data units. */
export interface DataBounds2D {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/** The parts of a Plotly full trace the zoom opacity reads. */
export interface OpacityTrace {
  meta?: unknown;
  marker?: { opacity?: unknown };
}

/**
 * Zoom-dependent marker opacity for the 2D scatter plot.
 *
 * The base opacity is calculateMarkerStyle's for the number of points: the more
 * points, the fainter each one. Zooming in shows a smaller share of the data box at
 * the same marker size, so each trace's opacity is multiplied by calculateMarkerStyle's
 * opacity for that share of the points over its opacity for all of them. Panning
 * keeps the share, so it keeps the opacity.
 *
 * The plot applies it through the Plotly fork's `scatterglOpacityScale` config
 * option, which sets each trace's opacity uniform before every draw: no recalc.
 * See documentation/ZOOM_OPACITY.md.
 */
export class ZoomOpacity {
  /** `meta` of traces that keep their opacity at every zoom (search glow, selected point). */
  static readonly FIXED_META = 'fixed-opacity';

  private readonly fullViewOpacity: number;
  // Every trace of a draw asks with the same ranges
  private lastRanges: number[] = [];
  private lastMultiplier = 1;

  constructor(
    private readonly pointCount: number,
    private readonly bounds: DataBounds2D | null,
  ) {
    this.fullViewOpacity = calculateMarkerStyle(pointCount).opacity;
  }

  /** Factor on the full-view opacity for the given axis ranges, 1 at full view or zoomed out. */
  multiplier(xRange: readonly number[], yRange: readonly number[]): number {
    const last = this.lastRanges;
    if (last[0] === xRange[0] && last[1] === xRange[1] && last[2] === yRange[0] && last[3] === yRange[1]) {
      return this.lastMultiplier;
    }
    this.lastRanges = [xRange[0], xRange[1], yRange[0], yRange[1]];
    this.lastMultiplier = this.computeMultiplier(xRange, yRange);
    return this.lastMultiplier;
  }

  /**
   * Factor for one trace's opacity, never past fully opaque. Fixed traces and traces
   * with per-point opacities keep theirs.
   */
  traceScale(trace: OpacityTrace, xRange: readonly number[], yRange: readonly number[]): number {
    const opacity = trace.marker?.opacity;
    if (trace.meta === ZoomOpacity.FIXED_META || typeof opacity !== 'number' || opacity <= 0) return 1;
    return Math.min(this.multiplier(xRange, yRange), 1 / opacity);
  }

  private computeMultiplier(xRange: readonly number[], yRange: readonly number[]): number {
    if (!this.bounds) return 1;
    const share =
      ZoomOpacity.axisShare(xRange, this.bounds.maxX - this.bounds.minX) *
      ZoomOpacity.axisShare(yRange, this.bounds.maxY - this.bounds.minY);
    if (share >= 1) return 1;
    return calculateMarkerStyle(this.pointCount * share).opacity / this.fullViewOpacity;
  }

  /** Share of the data extent one axis range shows; an axis whose data has no extent counts as fully shown. */
  private static axisShare(range: readonly number[], extent: number): number {
    if (!(extent > 0)) return 1;
    return Math.min(Math.abs(range[1] - range[0]) / extent, 1);
  }
}
