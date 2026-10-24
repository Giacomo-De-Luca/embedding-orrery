import { describe, expect, it } from 'vitest';

import { ExportPixelRatio, type GlPlot3d, type PlotlyGraph } from '../exportPixelRatio';

/** A Plotly stand-in whose toImage records the graph's WebGL ratio at call time. */
function fakePlotly(throws = false) {
  const seen: (number | undefined)[] = [];
  const options: unknown[] = [];
  return {
    seen,
    options,
    toImage(gd: PlotlyGraph, opts: unknown): Promise<string> {
      seen.push(gd._context.plotGlPixelRatio);
      options.push(opts);
      if (throws) throw new Error('toImage failed');
      return Promise.resolve('data:image/png;base64,');
    },
  };
}

/** A gl-plot3d stand-in recording the ratio of every redraw. */
function fakeGlPlot(pixelRatio = 2, throws = false): GlPlot3d & { redraws: number[] } {
  return {
    pixelRatio,
    redraws: [],
    redraw() {
      this.redraws.push(this.pixelRatio);
      if (throws) throw new Error('redraw failed');
    },
  };
}

describe('ExportPixelRatio.plotToImage', () => {
  it("renders the clone's WebGL layer at the export scale and puts the live ratio back", async () => {
    const plotly = fakePlotly();
    const gd: PlotlyGraph = { _context: { plotGlPixelRatio: 2 } };
    await ExportPixelRatio.plotToImage(plotly, gd, 1440, 900, 2.67);
    expect(plotly.seen).toEqual([2.67]);
    expect(plotly.options).toEqual([{ format: 'png', width: 1440, height: 900, scale: 2.67 }]);
    expect(gd._context.plotGlPixelRatio).toBe(2);
  });

  it('never lowers the live ratio, and restores an unset one', async () => {
    const plotly = fakePlotly();
    const set: PlotlyGraph = { _context: { plotGlPixelRatio: 2 } };
    await ExportPixelRatio.plotToImage(plotly, set, 2560, 1440, 1.5);
    const unset: PlotlyGraph = { _context: {} };
    await ExportPixelRatio.plotToImage(plotly, unset, 2560, 1440, 1.5);
    expect(plotly.seen).toEqual([2, 1.5]);
    expect(unset._context.plotGlPixelRatio).toBeUndefined();
  });

  it('puts the live ratio back when toImage throws', () => {
    const gd: PlotlyGraph = { _context: { plotGlPixelRatio: 2 } };
    expect(() => ExportPixelRatio.plotToImage(fakePlotly(true), gd, 800, 600, 4.8)).toThrow('toImage failed');
    expect(gd._context.plotGlPixelRatio).toBe(2);
  });
});

describe('ExportPixelRatio.redrawAt', () => {
  it('reads a frame drawn at the export ratio, then redraws at the live one', () => {
    const glplot = fakeGlPlot();
    const read = ExportPixelRatio.redrawAt(glplot, 4.8, () => glplot.redraws.slice());
    expect(read).toEqual([4.8]);
    expect(glplot.redraws).toEqual([4.8, 2]);
    expect(glplot.pixelRatio).toBe(2);
  });

  it('redraws once when the ratio is already the live one', () => {
    const glplot = fakeGlPlot();
    ExportPixelRatio.redrawAt(glplot, 2, () => null);
    expect(glplot.redraws).toEqual([2]);
  });

  it('puts the live ratio back and redraws when the read throws', () => {
    const glplot = fakeGlPlot();
    expect(() => ExportPixelRatio.redrawAt(glplot, 4.8, () => {
      throw new Error('context lost');
    })).toThrow('context lost');
    expect(glplot.pixelRatio).toBe(2);
    expect(glplot.redraws).toEqual([4.8, 2]);
  });

  it('puts the live ratio back when the redraw throws', () => {
    const glplot = fakeGlPlot(2, true);
    expect(() => ExportPixelRatio.redrawAt(glplot, 4.8, () => null)).toThrow('redraw failed');
    expect(glplot.pixelRatio).toBe(2);
  });
});
