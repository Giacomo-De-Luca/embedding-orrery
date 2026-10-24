import { describe, expect, it } from 'vitest';

import { CartesianView, type AxisRanges, type PlotArea } from '../cartesianView';

const ranges: AxisRanges = { xRange: [0, 10], yRange: [0, 20] };
const area: PlotArea = { left: 40, top: 8, width: 200, height: 100 };

describe('CartesianView', () => {
  it('reads the plot area from _size before the margins', () => {
    const fullLayout = {
      _size: { l: 12, t: 5, w: 300, h: 150 },
      margin: { l: 99, r: 99, t: 99, b: 99 },
      width: 800,
      height: 600,
    };
    expect(CartesianView.plotAreaOf(fullLayout, 800, 600)).toEqual({ left: 12, top: 5, width: 300, height: 150 });
  });

  it('falls back to the margins and the given size before Plotly has laid out', () => {
    expect(CartesianView.plotAreaOf({ margin: { l: 40, r: 8, t: 8, b: 30 } }, 800, 600))
      .toEqual({ left: 40, top: 8, width: 752, height: 562 });
    expect(CartesianView.plotAreaOf(undefined, 800, 600)).toEqual({ left: 50, top: 50, width: 700, height: 500 });
    expect(CartesianView.plotAreaOf({ margin: { l: 0, r: 0, t: 0, b: 0 }, width: 0, height: 0 }, 0, 0))
      .toEqual({ left: 0, top: 0, width: 1, height: 1 });
  });

  it('reads the axis ranges, or null before Plotly has both', () => {
    expect(CartesianView.rangesOf({ xaxis: { range: [1, 2] }, yaxis: { range: [3, 4] } }))
      .toEqual({ xRange: [1, 2], yRange: [3, 4] });
    expect(CartesianView.rangesOf({ xaxis: { range: [1, 2] } })).toBeNull();
    expect(CartesianView.rangesOf(undefined)).toBeNull();
  });

  it('builds the current view from _fullLayout', () => {
    const view = CartesianView.fromFullLayout(
      { xaxis: { range: [0, 10] }, yaxis: { range: [0, 20] }, _size: { l: 40, t: 8, w: 200, h: 100 } },
      800,
      600,
    );
    expect(view?.ranges).toEqual(ranges);
    expect(view?.plotArea).toEqual(area);
    expect(CartesianView.fromFullLayout({}, 800, 600)).toBeNull();
  });

  it('projects data to screen pixels with the y axis pointing up', () => {
    const view = new CartesianView(ranges, area);
    expect(view.project(0, 0)).toEqual({ x: 40, y: 108 });
    expect(view.project(10, 20)).toEqual({ x: 240, y: 8 });
    expect(view.project(5, 10)).toEqual({ x: 140, y: 58 });
  });

  it('projects to the plot corner when a range has no span', () => {
    expect(new CartesianView({ xRange: [3, 3], yRange: [0, 1] }, area).project(3, 0.5)).toEqual({ x: 40, y: 8 });
  });

  it('tests screen points against the plot area, optionally widened by a margin', () => {
    const view = new CartesianView(ranges, area);
    expect(view.contains({ x: 40, y: 8 })).toBe(true);
    expect(view.contains({ x: 240, y: 108 })).toBe(true);
    expect(view.contains({ x: 39, y: 50 })).toBe(false);
    expect(view.contains({ x: 25, y: 50 }, 20)).toBe(true);
    expect(view.contains({ x: 261, y: 50 }, 20)).toBe(false);
  });

  it('measures zoom as the geometric mean of the span ratios', () => {
    expect(new CartesianView({ xRange: [0, 5], yRange: [0, 5] }, area).scaleRelativeTo(ranges))
      .toBeCloseTo(Math.sqrt(0.5 * 0.25));
    expect(new CartesianView({ xRange: [10, 0], yRange: [20, 0] }, area).scaleRelativeTo(ranges)).toBeCloseTo(1);
    // A zero span counts as 1, as before
    expect(new CartesianView({ xRange: [1, 1], yRange: [0, 20] }, area).scaleRelativeTo({ xRange: [0, 1], yRange: [0, 20] }))
      .toBeCloseTo(1);
  });

  it('compares plot areas', () => {
    expect(CartesianView.sameArea(area, { ...area })).toBe(true);
    expect(CartesianView.sameArea(area, { ...area, left: 0 })).toBe(false);
    expect(CartesianView.sameArea(null, area)).toBe(false);
    expect(CartesianView.sameArea(null, null)).toBe(true);
  });
});
