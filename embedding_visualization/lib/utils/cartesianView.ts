/**
 * A 2D Plotly view: the axis ranges on screen and the plot area they fill, read
 * from Plotly's `_fullLayout`. Maps data coordinates to CSS pixels within the plot
 * div for the 2D overlays (labels, density map) and the zoom limit.
 */

export interface PlotArea {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface AxisRanges {
  xRange: [number, number];
  yRange: [number, number];
}

/** The parts of Plotly's `_fullLayout` a 2D view reads. */
export interface CartesianFullLayout {
  xaxis?: { range?: number[] };
  yaxis?: { range?: number[] };
  _size?: { l: number; t: number; w: number; h: number };
  margin?: { l?: number; r?: number; t?: number; b?: number };
  width?: number;
  height?: number;
}

export class CartesianView {
  constructor(readonly ranges: AxisRanges, readonly plotArea: PlotArea) {}

  /** The view Plotly shows now, or null before it has axis ranges. */
  static fromFullLayout(
    fullLayout: CartesianFullLayout | null | undefined,
    fallbackWidth: number,
    fallbackHeight: number,
  ): CartesianView | null {
    const ranges = CartesianView.rangesOf(fullLayout);
    if (!ranges) return null;
    return new CartesianView(ranges, CartesianView.plotAreaOf(fullLayout, fallbackWidth, fallbackHeight));
  }

  /** The current axis ranges, or null until Plotly has both. */
  static rangesOf(fullLayout: CartesianFullLayout | null | undefined): AxisRanges | null {
    const xRange = fullLayout?.xaxis?.range;
    const yRange = fullLayout?.yaxis?.range;
    if (!xRange || !yRange) return null;
    return { xRange: [xRange[0], xRange[1]], yRange: [yRange[0], yRange[1]] };
  }

  /**
   * The plot area in CSS px within the plot div. Prefers Plotly's exact `_size`,
   * falling back to the margins before the plot has been laid out.
   */
  static plotAreaOf(
    fullLayout: CartesianFullLayout | null | undefined,
    fallbackWidth: number,
    fallbackHeight: number,
  ): PlotArea {
    const size = fullLayout?._size;
    if (size && typeof size.l === 'number' && typeof size.w === 'number') {
      return { left: size.l, top: size.t, width: size.w, height: size.h };
    }
    const ml = fullLayout?.margin?.l ?? 50;
    const mr = fullLayout?.margin?.r ?? 50;
    const mt = fullLayout?.margin?.t ?? 50;
    const mb = fullLayout?.margin?.b ?? 50;
    return {
      left: ml,
      top: mt,
      width: Math.max((fullLayout?.width ?? fallbackWidth) - ml - mr, 1),
      height: Math.max((fullLayout?.height ?? fallbackHeight) - mt - mb, 1),
    };
  }

  static sameArea(a: PlotArea | null, b: PlotArea | null): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    return a.left === b.left && a.top === b.top && a.width === b.width && a.height === b.height;
  }

  /** CSS-pixel position of a data point (Plotly's y axis points up, the screen's down). */
  project(x: number, y: number): { x: number; y: number } {
    const { xRange, yRange } = this.ranges;
    const area = this.plotArea;
    const xSpan = xRange[1] - xRange[0];
    const ySpan = yRange[1] - yRange[0];
    if (xSpan === 0 || ySpan === 0) return { x: area.left, y: area.top };
    return {
      x: area.left + ((x - xRange[0]) / xSpan) * area.width,
      y: area.top + (1 - (y - yRange[0]) / ySpan) * area.height,
    };
  }

  /** Whether a screen point lies in the plot area, widened by `margin` px on every side. */
  contains(point: { x: number; y: number }, margin = 0): boolean {
    const area = this.plotArea;
    return point.x >= area.left - margin && point.x <= area.left + area.width + margin
      && point.y >= area.top - margin && point.y <= area.top + area.height + margin;
  }

  /**
   * Zoom relative to `initial`: the geometric mean of the span ratios (1 = the
   * initial view, below 1 = zoomed in). This is the dynamic label placement's scale.
   */
  scaleRelativeTo(initial: AxisRanges): number {
    const span = (range: [number, number]) => Math.abs(range[1] - range[0]) || 1;
    const scaleX = span(this.ranges.xRange) / span(initial.xRange);
    const scaleY = span(this.ranges.yRange) / span(initial.yRange);
    return Math.sqrt(scaleX * scaleY);
  }
}
