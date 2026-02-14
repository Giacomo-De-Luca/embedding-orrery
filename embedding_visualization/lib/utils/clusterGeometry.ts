import type { Point3D, NestedColorMap } from '../types/types';

/**
 * Cluster geometry utilities for nebula haze effects and cluster labels.
 * Groups points by category, computes centroid + std per cluster.
 */

export interface ClusterData {
  points: Point3D[];
  centroid: { x: number; y: number; z: number };
  std: { x: number; y: number; z: number };
  color: string;
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
