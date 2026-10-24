/** The parts of a Plotly graph div that the 2D export touches. */
export interface PlotlyGraph {
  _context: { plotGlPixelRatio?: number };
}

/** The part of Plotly that the 2D export calls. */
export interface PlotlyImageExport {
  toImage(gd: PlotlyGraph, opts: { format: 'png'; width: number; height: number; scale: number }): Promise<string>;
}

/** The parts of gl-plot3d's scene that the 3D export touches. */
export interface GlPlot3d {
  pixelRatio: number;
  redraw(): void;
}

/**
 * Renders a plot's WebGL layer at a screenshot's pixel ratio for one call, then
 * puts the live ratio back whatever the call does. Neither ratio ever drops below
 * the live one, which already supersamples a 1× screen. The Plotly and gl-plot3d
 * behaviour this relies on is pinned by forked/__tests__/screenshotScaleContract.test.ts.
 */
export class ExportPixelRatio {
  /**
   * Plotly.toImage at `scale`, with the clone's WebGL points rendered at that ratio
   * too. The clone draws its points at the graph's `plotGlPixelRatio` (2 by
   * default) and the SVG holding them is then scaled by `scale`, so a 4K export
   * would upscale them. toImage copies the graph's config before it returns, so
   * the ratio is raised for the call only and the live plot never sees it.
   */
  static plotToImage(
    plotlyLib: PlotlyImageExport,
    gd: PlotlyGraph,
    width: number,
    height: number,
    scale: number,
  ): Promise<string> {
    const context = gd._context;
    const livePixelRatio = context.plotGlPixelRatio;
    context.plotGlPixelRatio = Math.max(scale, livePixelRatio ?? 1);
    try {
      return plotlyLib.toImage(gd, { format: 'png', width, height, scale });
    } finally {
      context.plotGlPixelRatio = livePixelRatio;
    }
  }

  /**
   * Redraws the 3D scene at `pixelRatio`, runs `read` on the result and puts the
   * live ratio back, all in this task. gl-plot3d sizes its canvas from the
   * container's CSS size × its ratio on every redraw and scales glyphs and line
   * widths by the same ratio, so this is the on-screen frame at a higher
   * resolution. The browser composites only after the task, so the export-size
   * frame never reaches the screen.
   */
  static redrawAt<T>(glplot: GlPlot3d, pixelRatio: number, read: () => T): T {
    const livePixelRatio = glplot.pixelRatio;
    const ratio = Math.max(pixelRatio, livePixelRatio);
    glplot.pixelRatio = ratio;
    try {
      glplot.redraw();
      return read();
    } finally {
      glplot.pixelRatio = livePixelRatio;
      if (ratio !== livePixelRatio) glplot.redraw();
    }
  }
}
