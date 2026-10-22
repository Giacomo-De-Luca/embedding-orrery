/**
 * Capability probe for the density haze renderer: WebGL2 with renderable float
 * textures (EXT_color_buffer_float). Cached after the first call; always false
 * during SSR. Three-free so UI code can gate controls without importing the
 * renderer.
 */

import type { HazeRendererKind } from '../../types/types';

let cached: boolean | null = null;

export function isDensityHazeSupported(): boolean {
  if (cached !== null) return cached;
  let ok = false;
  try {
    if (typeof document !== 'undefined') {
      const gl = document.createElement('canvas').getContext('webgl2');
      ok = !!gl && !!gl.getExtension('EXT_color_buffer_float');
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    }
  } catch {
    ok = false;
  }
  cached = ok;
  return ok;
}

/** The renderer this browser can run: `density` degrades to sprites without float targets. */
export function supportedHazeRenderer(kind: HazeRendererKind): HazeRendererKind {
  return kind === 'density' && !isDensityHazeSupported() ? 'sprites' : kind;
}
