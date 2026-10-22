/**
 * Pixels read back with `gl.readPixels` from a WebGL canvas created with
 * `premultipliedAlpha: true` (Plotly's 3D scene).
 *
 * gl-plot3d blends with (ONE, ONE_MINUS_SRC_ALPHA), but gl-scatter3d writes
 * `colour × opacity` without multiplying RGB by the colour's own alpha, so a
 * translucent rgba() marker or line (the search glow layers, the connection
 * lines) leaves RGB above alpha in the drawing buffer. The browser composites
 * such pixels additively (rgb + (1 − a) × backdrop), which is what makes the
 * glow glow. Converting them to straight alpha, as Plotly's `scene.toImage`
 * does, has to clamp RGB/alpha at 255 and throws most of that light away.
 * Flattening onto the export background with the compositor's own formula
 * keeps it.
 */

export type Rgb255 = readonly [number, number, number];

export class PremultipliedPixels {
  /**
   * Composite premultiplied RGBA rows in `gl.readPixels` order (bottom row
   * first) onto an opaque background, returning top-down opaque RGBA ready for
   * `ImageData`. Channels are clamped at 255, as the browser does.
   */
  static flattenOnto(pixels: Uint8Array, width: number, height: number, background: Rgb255): Uint8ClampedArray<ArrayBuffer> {
    const rowBytes = width * 4;
    if (pixels.length !== rowBytes * height) {
      throw new Error(`Expected ${rowBytes * height} bytes for a ${width}×${height} image, got ${pixels.length}`);
    }
    const [bgR, bgG, bgB] = background;
    const out = new Uint8ClampedArray(pixels.length);
    for (let row = 0; row < height; row++) {
      const src = (height - 1 - row) * rowBytes;
      const dst = row * rowBytes;
      for (let i = 0; i < rowBytes; i += 4) {
        const uncovered = (255 - pixels[src + i + 3]) / 255;
        out[dst + i] = pixels[src + i] + uncovered * bgR;
        out[dst + i + 1] = pixels[src + i + 1] + uncovered * bgG;
        out[dst + i + 2] = pixels[src + i + 2] + uncovered * bgB;
        out[dst + i + 3] = 255;
      }
    }
    return out;
  }
}
