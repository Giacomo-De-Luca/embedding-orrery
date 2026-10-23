/**
 * WebGL2 float render-target capability, shared by both density renderers:
 * the 2D overlay (`densityRenderer/`) and the 3D density haze (`densityHaze/`).
 *
 * Two tiers:
 * - `float32` — RGBA32F targets that blend additively and sample with LINEAR
 *   filtering (EXT_color_buffer_float + EXT_float_blend +
 *   OES_texture_float_linear). Desktop GPUs.
 * - `float16` — RGBA16F targets (EXT_color_buffer_float alone): WebGL2 filters
 *   half floats natively and EXT_float_blend only governs 32-bit targets.
 *   This is what phones run: Apple's iPhone GPUs neither blend nor filter
 *   RGBA32F, and many Android GPUs lack 32-bit filtering.
 *
 * Three-free and shader-free, so UI code can gate controls cheaply.
 */

export type FloatPrecision = 'float32' | 'float16';

const RENDERABLE_EXTENSION = 'EXT_color_buffer_float';
const FLOAT32_EXTENSIONS = ['EXT_float_blend', 'OES_texture_float_linear'];

let probed: FloatPrecision | null | undefined;

export class FloatRenderTargets {
  /**
   * The best precision `gl` supports, or null without renderable float
   * textures. `getExtension` also enables what it returns, so renderers call
   * this on their own context before allocating targets.
   */
  static enable(gl: { getExtension(name: string): unknown }): FloatPrecision | null {
    if (!gl.getExtension(RENDERABLE_EXTENSION)) return null;
    const blendsAndFilters = FLOAT32_EXTENSIONS.every((ext) => gl.getExtension(ext) != null);
    return blendsAndFilters ? 'float32' : 'float16';
  }

  /** This browser's precision, probed once on a throwaway context; null during SSR. */
  static probe(): FloatPrecision | null {
    if (probed !== undefined) return probed;
    let precision: FloatPrecision | null = null;
    try {
      if (typeof document !== 'undefined') {
        const gl = document.createElement('canvas').getContext('webgl2');
        if (gl) {
          precision = FloatRenderTargets.enable(gl);
          // Browsers cap live contexts (Safari especially); free the probe's.
          gl.getExtension('WEBGL_lose_context')?.loseContext();
        }
      }
    } catch {
      precision = null;
    }
    probed = precision;
    return precision;
  }

  /** True when either tier is available. */
  static available(): boolean {
    return FloatRenderTargets.probe() !== null;
  }
}
