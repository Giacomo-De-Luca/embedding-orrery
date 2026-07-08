/**
 * 2D label placement utilities using Apple's dynamic map labeling algorithm.
 *
 * Bridges Plotly 2D axis ranges to the scale-aware placement system from
 * `dynamicMapPlacement.ts`. Labels smoothly appear/disappear as the user
 * zooms, with larger clusters visible at lower zoom levels.
 */

import { dynamicLabelPlacement, type Label, type Placement } from './dynamicMapPlacement';
import type { ClusterData } from '../../lib/utils/clusterGeometry';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlotArea {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface AxisRanges {
  xRange: [number, number];
  yRange: [number, number];
}

export interface ClusterLabelPlacement {
  label: string;
  color: string;
  /** Data-space centroid */
  dataX: number;
  dataY: number;
  /** Text width in CSS pixels (at rendering font) */
  textWidth: number;
  /** Apple placement result (null = never visible) */
  placement: Placement | null;
  /** Priority (cluster point count) */
  priority: number;
  /** Font size in px; renderers default to 13 when absent */
  fontSize?: number;
  /** Topic label backing this text, when it differs from `label` semantics
   *  (density labels use it to keep topic-click behavior) */
  topicLabel?: string | null;
}

// ---------------------------------------------------------------------------
// Plot area
// ---------------------------------------------------------------------------

/**
 * Extract the plot-area rect (CSS px within the plot div) from Plotly's
 * internal `_fullLayout`. Prefers the exact `_size` object, falling back to
 * margins when the plot hasn't fully laid out yet.
 */
export function plotAreaFromFullLayout(
  fl: any,
  fallbackWidth: number,
  fallbackHeight: number,
): PlotArea {
  const size = fl?._size;
  if (size && typeof size.l === 'number' && typeof size.w === 'number') {
    return { left: size.l, top: size.t, width: size.w, height: size.h };
  }
  const ml = fl?.margin?.l ?? 50;
  const mr = fl?.margin?.r ?? 50;
  const mt = fl?.margin?.t ?? 50;
  const mb = fl?.margin?.b ?? 50;
  return {
    left: ml,
    top: mt,
    width: Math.max((fl?.width ?? fallbackWidth) - ml - mr, 1),
    height: Math.max((fl?.height ?? fallbackHeight) - mt - mb, 1),
  };
}

// ---------------------------------------------------------------------------
// Coordinate projection
// ---------------------------------------------------------------------------

/**
 * Project a data-space coordinate to CSS pixel coordinates within the plot area.
 */
export function projectDataToScreen(
  dataX: number,
  dataY: number,
  ranges: AxisRanges,
  plotArea: PlotArea,
): { x: number; y: number } {
  const xSpan = ranges.xRange[1] - ranges.xRange[0];
  const ySpan = ranges.yRange[1] - ranges.yRange[0];
  if (xSpan === 0 || ySpan === 0) return { x: plotArea.left, y: plotArea.top };

  const normX = (dataX - ranges.xRange[0]) / xSpan;
  // Plotly Y axis goes bottom-to-top, screen Y goes top-to-bottom
  const normY = 1 - (dataY - ranges.yRange[0]) / ySpan;

  return {
    x: plotArea.left + normX * plotArea.width,
    y: plotArea.top + normY * plotArea.height,
  };
}

// ---------------------------------------------------------------------------
// Scale computation
// ---------------------------------------------------------------------------

/**
 * Compute the Apple algorithm's "scale" from the current view ranges vs. the
 * initial (full-extent) ranges.
 *
 * In Apple's model, scale controls label size on screen:
 *   scale = 0 → labels are points (no overlap, all visible)
 *   scale = 1 → labels at full size (max overlap, fewest visible)
 *
 * We map: scale = currentSpan / initialSpan
 *   Zoomed all the way out → scale ≈ 1 (full overlap, few labels)
 *   Zoomed in             → scale < 1 (less overlap, more labels appear)
 */
export function computeCurrentScale(
  currentRanges: AxisRanges,
  initialRanges: AxisRanges,
): number {
  const curXSpan = Math.abs(currentRanges.xRange[1] - currentRanges.xRange[0]) || 1;
  const curYSpan = Math.abs(currentRanges.yRange[1] - currentRanges.yRange[0]) || 1;
  const iniXSpan = Math.abs(initialRanges.xRange[1] - initialRanges.xRange[0]) || 1;
  const iniYSpan = Math.abs(initialRanges.yRange[1] - initialRanges.yRange[0]) || 1;

  const scaleX = curXSpan / iniXSpan;
  const scaleY = curYSpan / iniYSpan;
  return Math.sqrt(scaleX * scaleY);
}

// ---------------------------------------------------------------------------
// Placement computation
// ---------------------------------------------------------------------------

/**
 * Build Apple `Label[]` descriptors from cluster data and run the placement
 * algorithm. Returns an array of `ClusterLabelPlacement` objects, one per
 * cluster, with pre-computed placement visibility ranges.
 *
 * @param clusterDataMap - Map of cluster name → ClusterData (from groupPointsByCluster)
 * @param ctx - A canvas 2D context for measuring text
 * @param font - The CSS font string used for cluster labels
 * @param initialRanges - The full-extent axis ranges (at scale=1)
 * @param plotArea - The plot's pixel area within the container
 * @param globalMaxScale - The maximum zoom scale the user can reach
 */
export function computeClusterLabelPlacements(
  clusterDataMap: Map<string, ClusterData>,
  ctx: CanvasRenderingContext2D,
  font: string,
  initialRanges: AxisRanges,
  plotArea: PlotArea,
  globalMaxScale: number = 2,
): ClusterLabelPlacement[] {
  if (clusterDataMap.size === 0) return [];

  ctx.font = font;
  const pad = 4; // px padding around text

  // Build label descriptors
  const entries: { label: string; color: string; dataX: number; dataY: number; textWidth: number; priority: number }[] = [];
  const appleLabels: Label[] = [];

  for (const [name, cluster] of clusterDataMap) {
    const textWidth = ctx.measureText(name).width;
    const halfW = textWidth / 2 + pad;
    const halfH = 8 + pad; // ~fontSize * 0.6 + padding

    // Project centroid to screen at scale=1 (initial full view)
    const screen = projectDataToScreen(
      cluster.centroid.x,
      cluster.centroid.y,
      initialRanges,
      plotArea,
    );

    entries.push({
      label: name,
      color: cluster.color,
      dataX: cluster.centroid.x,
      dataY: cluster.centroid.y,
      textWidth,
      priority: cluster.points.length,
    });

    appleLabels.push({
      bounds: {
        xMin: screen.x - halfW,
        yMin: screen.y - halfH,
        xMax: screen.x + halfW,
        yMax: screen.y + halfH,
      },
      locationAtZero: { x: screen.x, y: screen.y },
      priority: cluster.points.length,
    });
  }

  // Run Apple's dynamic placement algorithm
  const placements = dynamicLabelPlacement(appleLabels, { globalMaxScale });

  // Combine into result
  return entries.map((entry, i) => ({
    ...entry,
    placement: placements[i],
  }));
}
