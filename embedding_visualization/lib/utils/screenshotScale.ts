/**
 * How many export pixels a plot screenshot gives each CSS pixel: enough for a 4K
 * long side on any screen, or the screen's own ratio when that is higher. The image
 * keeps the plot's on-screen aspect ratio.
 */
export class ScreenshotScale {
  /** The long side of a 4K UHD frame, in pixels. */
  static readonly LONG_SIDE = 3840;

  static of(cssWidth: number, cssHeight: number, devicePixelRatio: number): number {
    const screenRatio = devicePixelRatio > 0 ? devicePixelRatio : 1;
    const longSide = Math.max(cssWidth, cssHeight);
    return longSide > 0 ? Math.max(screenRatio, ScreenshotScale.LONG_SIDE / longSide) : screenRatio;
  }
}
