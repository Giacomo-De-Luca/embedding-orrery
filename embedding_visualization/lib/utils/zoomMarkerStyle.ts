import { calculateMarkerStyle, type MarkerStyle } from './plotUtils';

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

/** Factors on the full-view marker size and opacity. */
export type ZoomFactors = MarkerStyle;

const NO_ZOOM: ZoomFactors = Object.freeze({ size: 1, opacity: 1 });

/**
 * Zoom-dependent marker size and opacity for the 2D scatter plot.
 *
 * calculateMarkerStyle picks the base size and opacity from the number of points:
 * the more points, the smaller and fainter each one. Zooming in shows a smaller share
 * of the data box, so each trace's size and opacity are multiplied by
 * calculateMarkerStyle's values for that share of the points over its values for all
 * of them: the view looks like a plot of the points you can see. Panning keeps the
 * share, so it keeps the style.
 *
 * The plot applies it through the Plotly fork's `scatterglSizeScale` and
 * `scatterglOpacityScale` config options, which set each trace's size and opacity
 * before every draw: no recalc. See documentation/ZOOM_MARKER_STYLE.md.
 */
export class ZoomMarkerStyle {
  /** `meta` of traces that keep their opacity at every zoom (search glow, selected point). */
  static readonly FIXED_OPACITY_META = 'fixed-opacity';

  private readonly fullView: MarkerStyle;
  // Every trace of a draw asks with the same ranges
  private lastRanges: number[] = [];
  private lastFactors: ZoomFactors = NO_ZOOM;

  constructor(
    private readonly pointCount: number,
    private readonly bounds: DataBounds2D | null,
  ) {
    this.fullView = calculateMarkerStyle(pointCount);
  }

  /** Size and opacity factors for the given axis ranges, both 1 at full view or zoomed out. */
  multipliers(xRange: readonly number[], yRange: readonly number[]): ZoomFactors {
    const last = this.lastRanges;
    if (last[0] === xRange[0] && last[1] === xRange[1] && last[2] === yRange[0] && last[3] === yRange[1]) {
      return this.lastFactors;
    }
    this.lastRanges = [xRange[0], xRange[1], yRange[0], yRange[1]];
    this.lastFactors = this.computeFactors(xRange, yRange);
    return this.lastFactors;
  }

  /**
   * Factor for one trace's opacity, never past fully opaque. Fixed traces and traces
   * with per-point opacities keep theirs.
   */
  opacityScale(trace: OpacityTrace, xRange: readonly number[], yRange: readonly number[]): number {
    const opacity = trace.marker?.opacity;
    if (trace.meta === ZoomMarkerStyle.FIXED_OPACITY_META || typeof opacity !== 'number' || opacity <= 0) return 1;
    return Math.min(this.multipliers(xRange, yRange).opacity, 1 / opacity);
  }

  /** Factor for every trace's marker size, so the search glow stays sized relative to its neighbours. */
  sizeScale(xRange: readonly number[], yRange: readonly number[]): number {
    return this.multipliers(xRange, yRange).size;
  }

  private computeFactors(xRange: readonly number[], yRange: readonly number[]): ZoomFactors {
    if (!this.bounds) return NO_ZOOM;
    const share =
      ZoomMarkerStyle.axisShare(xRange, this.bounds.maxX - this.bounds.minX) *
      ZoomMarkerStyle.axisShare(yRange, this.bounds.maxY - this.bounds.minY);
    if (share >= 1) return NO_ZOOM;
    const zoomed = calculateMarkerStyle(this.pointCount * share);
    return { size: zoomed.size / this.fullView.size, opacity: zoomed.opacity / this.fullView.opacity };
  }

  /** Share of the data extent one axis range shows; an axis whose data has no extent counts as fully shown. */
  private static axisShare(range: readonly number[], extent: number): number {
    if (!(extent > 0)) return 1;
    return Math.min(Math.abs(range[1] - range[0]) / extent, 1);
  }
}
