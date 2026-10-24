import { describe, expect, it } from 'vitest';

import { ScreenshotScale } from '../screenshotScale';

/** The export size the capture allocates: the CSS size × the scale, rounded. */
function exportSize(cssWidth: number, cssHeight: number, devicePixelRatio: number): [number, number] {
  const scale = ScreenshotScale.of(cssWidth, cssHeight, devicePixelRatio);
  return [Math.round(cssWidth * scale), Math.round(cssHeight * scale)];
}

describe('ScreenshotScale', () => {
  it('scales a landscape plot to a 3840 px long side on a 1× screen, keeping its aspect ratio', () => {
    expect(exportSize(1440, 900, 1)).toEqual([3840, 2400]);
  });

  it('scales a retina plot past its 2× backing store to 4K', () => {
    expect(exportSize(1280, 720, 2)).toEqual([3840, 2160]);
  });

  it('measures the long side of a portrait plot', () => {
    expect(exportSize(390, 700, 3)).toEqual([2139, 3840]);
  });

  it("keeps the screen's own resolution when it is already above 4K", () => {
    expect(ScreenshotScale.of(2560, 1300, 2)).toBe(2);
    expect(exportSize(2560, 1300, 2)).toEqual([5120, 2600]);
  });

  it('falls back to the screen ratio for an unmeasured plot, and to 1 without one', () => {
    expect(ScreenshotScale.of(0, 0, 2)).toBe(2);
    expect(ScreenshotScale.of(0, 0, 0)).toBe(1);
    expect(ScreenshotScale.of(1920, 1080, 0)).toBe(2);
  });
});
