import type { Point3D, NestedColorMap } from '../types/types';

/**
 * Cluster geometry utilities for nebula effects.
 * Shared between volume (Plan A) and WebGL particle (Plan B) approaches.
 */

export interface ClusterData {
  points: Point3D[];
  centroid: { x: number; y: number; z: number };
  std: { x: number; y: number; z: number };
  color: string;
}

export interface DensityGrid {
  x: number[];
  y: number[];
  z: number[];
  value: number[];
  maxValue: number;
}

export interface NebulaParticles {
  positions: Float32Array;
  opacities: Float32Array;
  sizes: Float32Array;
}

/**
 * Group points by cluster (category field value), computing centroid + std per cluster.
 * Skips noise cluster (topic_id = -1 / "Unclustered").
 * Returns a Map from cluster label to ClusterData.
 */
export function groupPointsByCluster(
  points: Point3D[],
  categoryField: string | null,
  colorMap: Record<string, string>,
  nestedColorMap?: NestedColorMap | null,
): Map<string, ClusterData> {
  if (!categoryField || points.length === 0) return new Map();

  // Group points by category
  const groups = new Map<string, Point3D[]>();
  for (const p of points) {
    const raw = p.metadata?.[categoryField];
    const cat = (raw !== null && raw !== undefined && raw !== '') ? String(raw) : 'unknown';

    // Skip noise cluster
    if (cat === '-1' || cat === 'Unclustered' || cat === 'unclustered') continue;

    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat)!.push(p);
  }

  const result = new Map<string, ClusterData>();

  for (const [cat, pts] of groups) {
    if (pts.length < 3) continue; // Need minimum points for meaningful geometry

    // Centroid
    let sx = 0, sy = 0, sz = 0;
    for (const p of pts) { sx += p.x; sy += p.y; sz += p.z; }
    const n = pts.length;
    const cx = sx / n, cy = sy / n, cz = sz / n;

    // Standard deviation
    let vx = 0, vy = 0, vz = 0;
    for (const p of pts) {
      vx += (p.x - cx) ** 2;
      vy += (p.y - cy) ** 2;
      vz += (p.z - cz) ** 2;
    }
    const stdX = Math.sqrt(vx / n);
    const stdY = Math.sqrt(vy / n);
    const stdZ = Math.sqrt(vz / n);

    // Resolve color: prefer nested topic colors, then colorMap
    const color = (nestedColorMap?.topicColors?.[cat])
      ?? colorMap[cat]
      ?? '#7f7f7f';

    result.set(cat, {
      points: pts,
      centroid: { x: cx, y: cy, z: cz },
      std: { x: stdX || 0.01, y: stdY || 0.01, z: stdZ || 0.01 },
      color,
    });
  }

  return result;
}

/**
 * Plan A: Compute a 3D density grid using Gaussian KDE.
 * Uses Silverman's rule for bandwidth and a 3-sigma cutoff.
 */
export function computeDensityGrid(
  cluster: ClusterData,
  gridSize: number = 20,
  paddingFactor: number = 1.5,
  bandwidthScale: number = 1.0,
): DensityGrid {
  const { points, centroid, std } = cluster;

  // Subsample for large clusters to keep KDE fast
  const maxKdePoints = 2000;
  const samplePoints = points.length > maxKdePoints
    ? points.filter((_, i) => i % Math.ceil(points.length / maxKdePoints) === 0)
    : points;

  // Silverman's bandwidth per axis, scaled for smoother clouds
  const n = samplePoints.length;
  const factor = Math.pow(n, -1 / 7) * bandwidthScale; // Silverman's for 3D: n^(-1/(d+4))
  const hx = std.x * factor;
  const hy = std.y * factor;
  const hz = std.z * factor;

  // Grid bounds: centroid ± paddingFactor * std
  const xMin = centroid.x - paddingFactor * std.x;
  const xMax = centroid.x + paddingFactor * std.x;
  const yMin = centroid.y - paddingFactor * std.y;
  const yMax = centroid.y + paddingFactor * std.y;
  const zMin = centroid.z - paddingFactor * std.z;
  const zMax = centroid.z + paddingFactor * std.z;

  const dx = (xMax - xMin) / (gridSize - 1);
  const dy = (yMax - yMin) / (gridSize - 1);
  const dz = (zMax - zMin) / (gridSize - 1);

  // 3-sigma cutoff squared (per axis, normalized)
  const cutoffSq = 9; // 3^2

  const x: number[] = [];
  const y: number[] = [];
  const z: number[] = [];
  const value: number[] = [];
  let maxValue = 0;

  for (let ix = 0; ix < gridSize; ix++) {
    const gx = xMin + ix * dx;
    for (let iy = 0; iy < gridSize; iy++) {
      const gy = yMin + iy * dy;
      for (let iz = 0; iz < gridSize; iz++) {
        const gz = zMin + iz * dz;

        let density = 0;
        for (const p of samplePoints) {
          const rx = (gx - p.x) / hx;
          const ry = (gy - p.y) / hy;
          const rz = (gz - p.z) / hz;
          const r2 = rx * rx + ry * ry + rz * rz;
          if (r2 < cutoffSq) {
            density += Math.exp(-0.5 * r2);
          }
        }

        x.push(gx);
        y.push(gy);
        z.push(gz);
        value.push(density);
        if (density > maxValue) maxValue = density;
      }
    }
  }

  return { x, y, z, value, maxValue };
}

/**
 * Plan B: Sample billboard sprite particles from a Gaussian distribution around the cluster centroid.
 * Opacity and size decrease with distance from centroid.
 */
export function sampleNebulaParticles(
  cluster: ClusterData,
  count: number = 300,
  spread: number = 1.5,
): NebulaParticles {
  const { centroid, std } = cluster;
  const positions = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const sizes = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    // Box-Muller transform for Gaussian samples
    // Clamp away from zero to avoid Math.log(0) = -Infinity
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    const u3 = Math.random() || 1e-10;
    const u4 = Math.random();
    const u5 = Math.random() || 1e-10;
    const u6 = Math.random();

    const gx = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const gy = Math.sqrt(-2 * Math.log(u3)) * Math.cos(2 * Math.PI * u4);
    const gz = Math.sqrt(-2 * Math.log(u5)) * Math.cos(2 * Math.PI * u6);

    const px = centroid.x + gx * std.x * spread;
    const py = centroid.y + gy * std.y * spread;
    const pz = centroid.z + gz * std.z * spread;

    positions[i * 3] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    // Distance from centroid (normalized by std)
    const dist = Math.sqrt(gx * gx + gy * gy + gz * gz);

    // Opacity: peaks at center, fades with distance
    opacities[i] = Math.max(0, Math.exp(-0.5 * dist * dist) * 0.6);

    // Size: larger near center, smaller at edges
    sizes[i] = Math.max(2, 12 * Math.exp(-0.3 * dist));
  }

  return { positions, opacities, sizes };
}

/**
 * Convert hex color to normalized RGB [0-1] array for WebGL.
 */
export function hexToRgbNormalized(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const r = parseInt(h.substring(0, 2), 16) / 255;
  const g = parseInt(h.substring(2, 4), 16) / 255;
  const b = parseInt(h.substring(4, 6), 16) / 255;
  return [r, g, b];
}
