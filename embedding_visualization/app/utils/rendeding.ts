import type { Point2D, Point3D } from '../../lib/types/types';

// --- Animation Helpers ---
export const easeInOutCubic = (t: number): number => {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
};

export const lerp = (start: number, end: number, t: number) => {
  return start + (end - start) * t;
};

export function cartesianToSpherical(x: number, y: number, z: number) {
  const r = Math.sqrt(x * x + y * y + z * z);
  const theta = Math.atan2(y, x);
  const phi = Math.acos(z / (r || 1));
  return { r, theta, phi };
}

export function sphericalToCartesian(r: number, theta: number, phi: number) {
  return {
    x: r * Math.sin(phi) * Math.cos(theta),
    y: r * Math.sin(phi) * Math.sin(theta),
    z: r * Math.cos(phi),
  };
}

export const getZoomLevel = (
  eye: { x: number; y: number; z: number },
  center: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 }
): number => {
  const dx = eye.x - center.x;
  const dy = eye.y - center.y;
  const dz = eye.z - center.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

export const getZoomMultiplier = (
  eye: { x: number; y: number; z: number },
  center: { x: number; y: number; z: number } = { x: 0, y: 0, z: 0 },
  defaultDistance: number = Math.sqrt(0.9**2 * 3) // ~1.56 for your defaults
): number => {
  const distance = getZoomLevel(eye, center);
  return defaultDistance / distance;
};


// --- Data → Scene Coordinate Transforms ---
// Plotly's 3D scene with `aspectmode: 'data'` uses per-axis aspect ratios
// (range[i] / geoMean(ranges)) that, combined with gl-plot3d's model matrix,
// simplify to a uniform divisor: geoMean(paddedRanges). These utilities
// read the actual padded axis ranges from _fullLayout.scene for accuracy.

export interface DataBounds {
  minX: number; maxX: number;
  minY: number; maxY: number;
  minZ: number; maxZ: number;
}

export interface SceneNormalization {
  centerX: number;
  centerY: number;
  centerZ: number;
  /** Geometric mean of padded axis ranges */
  normRange: number;
}

/**
 * Read Plotly's padded axis ranges and compute the correct data→scene
 * normalization parameters. Falls back to raw bounds if layout isn't ready.
 */
export function getSceneNormalization(
  graphDiv: any,
  fallbackBounds?: DataBounds,
): SceneNormalization | null {
  const scene = graphDiv?._fullLayout?.scene;
  const xRange = scene?.xaxis?.range;
  const yRange = scene?.yaxis?.range;
  const zRange = scene?.zaxis?.range;

  if (xRange && yRange && zRange) {
    return {
      centerX: (xRange[0] + xRange[1]) / 2,
      centerY: (yRange[0] + yRange[1]) / 2,
      centerZ: (zRange[0] + zRange[1]) / 2,
      normRange: Math.cbrt(
        (xRange[1] - xRange[0]) *
        (yRange[1] - yRange[0]) *
        (zRange[1] - zRange[0])
      ) || 1,
    };
  }

  if (fallbackBounds) {
    return {
      centerX: (fallbackBounds.minX + fallbackBounds.maxX) / 2,
      centerY: (fallbackBounds.minY + fallbackBounds.maxY) / 2,
      centerZ: (fallbackBounds.minZ + fallbackBounds.maxZ) / 2,
      normRange: Math.cbrt(
        (fallbackBounds.maxX - fallbackBounds.minX) *
        (fallbackBounds.maxY - fallbackBounds.minY) *
        (fallbackBounds.maxZ - fallbackBounds.minZ)
      ) || 1,
    };
  }

  return null;
}

/**
 * Build a column-major 4x4 model matrix that transforms raw data coordinates
 * to scene coordinates: Scale(1/normRange) * Translate(-center).
 */
export function buildDataToSceneMatrix(
  graphDiv: any,
  fallbackBounds?: DataBounds,
): Float32Array {
  const norm = getSceneNormalization(graphDiv, fallbackBounds);
  if (!norm) {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  const s = 1 / norm.normRange;
  return new Float32Array([
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    -norm.centerX * s, -norm.centerY * s, -norm.centerZ * s, 1,
  ]);
}

export function formatHoverText(point: Point3D | Point2D): string {
  const label = point.label || point.id;
  const doc = point.document || '';
  const truncatedDoc = doc.length > 100 ? doc.substring(0, 100) + '...' : doc;
  return `${label}<br>${truncatedDoc}`;
}
