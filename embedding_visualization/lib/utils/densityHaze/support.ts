/**
 * Capability probe for the density haze renderer: WebGL2 with renderable float
 * textures (EXT_color_buffer_float; the haze renders into half-float targets).
 * Delegates to the shared, cached `FloatRenderTargets` probe; always false
 * during SSR. Three-free so UI code can gate controls without importing the
 * renderer.
 */

import type { HazeRendererKind } from '../../types/types';
import { FloatRenderTargets } from '../floatRenderTargets';

export function isDensityHazeSupported(): boolean {
  return FloatRenderTargets.available();
}

/** The renderer this browser can run: `density` degrades to sprites without float targets. */
export function supportedHazeRenderer(kind: HazeRendererKind): HazeRendererKind {
  return kind === 'density' && !isDensityHazeSupported() ? 'sprites' : kind;
}
