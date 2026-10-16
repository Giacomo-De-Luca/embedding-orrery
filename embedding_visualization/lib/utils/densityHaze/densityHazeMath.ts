/**
 * Density-haze math — pure, unit-tested (`__tests__/densityHazeMath.test.ts`).
 *
 * The density renderer splats every clustered point with a Gaussian kernel and
 * accumulates the light additively, so the brightness rule is explicit instead
 * of emerging from sprite counts:
 *
 *   kernel sigma   = kernelFrac × (projected rms radius of the cluster)
 *   expected peak  = n · σ² / (s² + σ²)   with s = per-axis std of the cluster
 *                    (a Gaussian cloud convolved with a Gaussian kernel; zoom-invariant
 *                     because kernel area and point density scale inversely with zoom)
 *   point weight   = peak^(−gamma)   gamma 0 → raw density, gamma 1 → every cluster peaks equally
 *   exposure       = exposureTarget / median(peak · weight)
 *
 * Everything here is in data units; the shader converts sigma to pixels per frame.
 */

import type { ClusterData } from '../clusterGeometry';
import { hexToRgb } from '../plotUtils';

export interface DensityHazeTuning {
  /** Kernel sigma as a fraction of the cluster's projected rms radius. */
  kernelFrac: number;
  /** Cluster-brightness equalisation exponent (0 = raw density, 1 = equalised). */
  gamma: number;
  /** Pre-knee intensity the median cluster peak is mapped to. */
  exposureTarget: number;
}

export const DEFAULT_DENSITY_HAZE_TUNING: DensityHazeTuning = {
  kernelFrac: 0.8,
  gamma: 0.5,
  exposureTarget: 0.45,
};

/** Clusters below this size get no haze (matches the sprite renderer). */
export const MIN_HAZE_CLUSTER_POINTS = 10;

/** Isotropic 3D rms radius → rms radius of a 2D projection. */
const RMS2D_FACTOR = Math.sqrt(2 / 3);
/** Isotropic 3D rms radius → per-axis std. */
const AXIS_STD_FACTOR = Math.sqrt(1 / 3);

/** Point-sprite width in sigmas: a sprite spans ±(span/2)σ. Shared by both shaders. */
export const SPRITE_SIGMA_SPAN = 6;

/**
 * GLSL twin of {@link pointSizeCompensation} — injected into the splat vertex
 * shader so the rule lives in one place next to its TypeScript mirror.
 */
export const CLAMP_COMPENSATION_GLSL = /* glsl */ `
  float clampCompensation(float ideal, float size) {
    float r = ideal / size;
    return min(1.0, r * r);
  }
`;

/** Degenerate clusters (all points coincident) still get a kernel this fraction of the largest cluster's rms. */
const SIGMA_FLOOR_FRACTION = 1e-3;

const FALLBACK_RGB: readonly [number, number, number] = [0.5, 0.5, 0.5];

export interface ClusterStats {
  n: number;
  rms3d: number;
}

export interface ClusterHazeParams {
  /** Kernel sigma in data units. */
  sigma: number;
  /** Expected accumulated peak of the cluster with unit weights. */
  peak: number;
  /** Per-point weight. */
  weight: number;
}

export interface ClusterBufferInfo {
  key: string;
  /** First point index in the shared buffers. */
  offset: number;
  count: number;
  stats: ClusterStats;
  params: ClusterHazeParams;
}

export interface DensityHazeBuffers {
  positions: Float32Array;
  colors: Float32Array;
  weights: Float32Array;
  sigmas: Float32Array;
  count: number;
  exposure: number;
  clusters: ClusterBufferInfo[];
}

export function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function clusterStatsOf(cluster: ClusterData): ClusterStats {
  const { x, y, z } = cluster.std;
  return { n: cluster.points.length, rms3d: Math.sqrt(x * x + y * y + z * z) };
}

export function clusterHazeParams(
  stats: ClusterStats,
  tuning: DensityHazeTuning,
  sigmaFloor = 0,
): ClusterHazeParams {
  const rms2d = stats.rms3d * RMS2D_FACTOR;
  const axisStd = stats.rms3d * AXIS_STD_FACTOR;
  const sigma = Math.max(tuning.kernelFrac * rms2d, sigmaFloor);
  const sigma2 = sigma * sigma;
  const peak = sigma2 > 0 ? (stats.n * sigma2) / (axisStd * axisStd + sigma2) : stats.n;
  return { sigma, peak, weight: weightForPeak(peak, tuning.gamma) };
}

function weightForPeak(peak: number, gamma: number): number {
  return peak > 0 ? Math.pow(peak, -gamma) : 1;
}

/** Global exposure so the median cluster peak lands on `target`; 1 when nothing to normalise. */
export function computeExposure(peakTimesWeight: number[], target: number): number {
  if (peakTimesWeight.length === 0) return 1;
  const m = median(peakTimesWeight);
  return m > 0 && Number.isFinite(m) ? target / m : 1;
}

/**
 * Weight factor that keeps accumulated brightness zoom-stable when the point
 * sprite is enlarged by the minimum pixel clamp. The maximum clamp is not
 * compensated: haze is allowed to fade as the camera zooms into a cluster.
 * Mirrors the formula in the splat vertex shader.
 */
export function pointSizeCompensation(idealPx: number, clampedPx: number): number {
  if (clampedPx <= idealPx || clampedPx <= 0) return 1;
  const ratio = idealPx / clampedPx;
  return ratio * ratio;
}

function colorToRgb01(color: string): readonly [number, number, number] {
  try {
    const { r, g, b } = hexToRgb(color);
    return [r / 255, g / 255, b / 255];
  } catch {
    return FALLBACK_RGB;
  }
}

/** Flatten a cluster map into GPU attribute arrays plus the exposure for the current tuning. */
export function buildDensityHazeBuffers(
  clusterDataMap: Map<string, ClusterData>,
  tuning: DensityHazeTuning,
): DensityHazeBuffers {
  const included: { key: string; cluster: ClusterData; stats: ClusterStats }[] = [];
  let total = 0;
  let maxRms = 0;
  for (const [key, cluster] of clusterDataMap) {
    if (cluster.points.length < MIN_HAZE_CLUSTER_POINTS) continue;
    const stats = clusterStatsOf(cluster);
    included.push({ key, cluster, stats });
    total += stats.n;
    maxRms = Math.max(maxRms, stats.rms3d);
  }

  const sigmaFloor = maxRms * SIGMA_FLOOR_FRACTION;
  const positions = new Float32Array(total * 3);
  const colors = new Float32Array(total * 3);
  const weights = new Float32Array(total);
  const sigmas = new Float32Array(total);
  const clusters: ClusterBufferInfo[] = [];

  let offset = 0;
  for (const { key, cluster, stats } of included) {
    const params = clusterHazeParams(stats, tuning, sigmaFloor);
    const [r, g, b] = colorToRgb01(cluster.color);
    for (let i = 0; i < stats.n; i++) {
      const p = cluster.points[i];
      const j = offset + i;
      positions[j * 3] = p.x;
      positions[j * 3 + 1] = p.y;
      positions[j * 3 + 2] = p.z;
      colors[j * 3] = r;
      colors[j * 3 + 1] = g;
      colors[j * 3 + 2] = b;
    }
    weights.fill(params.weight, offset, offset + stats.n);
    sigmas.fill(params.sigma, offset, offset + stats.n);
    clusters.push({ key, offset, count: stats.n, stats, params });
    offset += stats.n;
  }

  return {
    positions,
    colors,
    weights,
    sigmas,
    count: total,
    exposure: exposureOf(clusters, tuning),
    clusters,
  };
}

/** Recompute weights + exposure for a new gamma/target; geometry and kernels are reused. */
export function reweightDensityHazeBuffers(
  buffers: DensityHazeBuffers,
  tuning: DensityHazeTuning,
): DensityHazeBuffers {
  const weights = new Float32Array(buffers.count);
  const clusters = buffers.clusters.map((c) => {
    const weight = weightForPeak(c.params.peak, tuning.gamma);
    weights.fill(weight, c.offset, c.offset + c.count);
    return { ...c, params: { ...c.params, weight } };
  });
  return { ...buffers, weights, clusters, exposure: exposureOf(clusters, tuning) };
}

function exposureOf(clusters: ClusterBufferInfo[], tuning: DensityHazeTuning): number {
  return computeExposure(
    clusters.map((c) => c.params.peak * c.params.weight),
    tuning.exposureTarget,
  );
}
