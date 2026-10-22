/**
 * Haze renderer selection for ScatterPlot3D's nebula overlay.
 *
 * Two implementations share one lifecycle contract (`HazeRendererLike`):
 *   - `HazeRenderer` (lib/utils/hazeRenderer.ts): feathered sprites at sampled points.
 *   - `DensityHazeRenderer` (lib/utils/densityHaze/): per-point density splats,
 *     experimental; needs WebGL2 float targets and falls back to sprites otherwise.
 */

import type { ClusterData } from './clusterGeometry';
import type { HazeRendererKind } from '../types/types';
import { HazeRenderer } from './hazeRenderer';
import { DensityHazeRenderer } from './densityHaze/DensityHazeRenderer';
import { supportedHazeRenderer } from './densityHaze/support';
import { DEFAULT_DENSITY_HAZE_TUNING, type DensityHazeTuning } from './densityHaze/densityHazeMath';

export interface HazeRendererLike {
  updateClusters(clusterDataMap: Map<string, ClusterData>, totalPoints: number): void;
  resize(width: number, height: number): void;
  render(projection: Float32Array, view: Float32Array, model: Float32Array): void;
  dispose(): void;
  /** Renderers that can retune without a rebuild (density only). */
  setTuning?(tuning: Partial<DensityHazeTuning>): void;
}

/** User-facing knobs from the store: strength scales the exposure target, equalize is gamma. */
export interface HazeTuningPrefs {
  strength: number;
  equalize: number;
}

export function hazeTuningFromPrefs(prefs: HazeTuningPrefs): Partial<DensityHazeTuning> {
  return {
    exposureTarget: DEFAULT_DENSITY_HAZE_TUNING.exposureTarget * prefs.strength,
    gamma: prefs.equalize,
  };
}

/**
 * Instantiate the requested renderer. `density` needs float render targets; the
 * Controls panel disables that option (and resets the preference) when the probe
 * fails, so the fallback here is defence in depth, not a user-facing path.
 */
export function createHazeRenderer(
  kind: HazeRendererKind,
  canvas: HTMLCanvasElement,
  isDark: boolean,
  prefs?: HazeTuningPrefs,
): HazeRendererLike {
  if (supportedHazeRenderer(kind) === 'density') {
    return new DensityHazeRenderer(canvas, isDark, prefs ? hazeTuningFromPrefs(prefs) : undefined);
  }
  return new HazeRenderer(canvas, isDark);
}
