/**
 * Density-based cluster labels for the 2D scatter plot — port of
 * embedding-atlas's auto-label pipeline (EmbeddingViewImpl.svelte
 * generateClusters/generateLabels + labels.ts layoutLabels, Apple MIT),
 * adapted to Plotly axis ranges and this app's hybrid label-text rule:
 * majority topic_label of the cluster's member points when the collection
 * has topics, else client-side c-TF-IDF via TextSummarizer.
 *
 * Pipeline (once per data change, in useDensityClusterLabels):
 *   computeDensityMap (CPU, 500², σ 5 / 2.5 → levels 0 / 1)
 *   → findClusters (WASM) → clustersFromFindClustersResult (pixel → data)
 *   → resolveClusterLabelTexts (hybrid text)
 *   → computeDensityLabelPlacements (per-level zoom bands + dynamic placement)
 * The result plugs into ScatterPlot2D's existing cluster-label canvas.
 */

import { computeDensityMap, type DensityMapResult } from './densityUtils';
import { findClusters, type Cluster } from '../../lib/density-clustering';
import { TextSummarizer, type Rectangle } from './text_summarizer';
import { dynamicLabelPlacement, type Label } from './dynamicMapPlacement';
import {
  projectDataToScreen,
  type AxisRanges,
  type PlotArea,
  type ClusterLabelPlacement,
} from './labelPlacement2D';

export interface DensityCluster {
  /** Density-weighted cluster center, data space. */
  x: number;
  y: number;
  sumDensity: number;
  /** Rectangle approximation of the cluster boundary, data space. */
  rects: Rectangle[];
  /** 0 = coarse (labels shown zoomed out), 1 = fine (shown zoomed in). */
  level: 0 | 1;
}

export type LabeledDensityCluster = DensityCluster & {
  text: string;
  /** Set when the text came from a topic label (keeps labels clickable). */
  topicLabel: string | null;
};

export type DensityLabelPlacement = ClusterLabelPlacement & {
  fontSize: number;
  level: 0 | 1;
  topicLabel: string | null;
};

// Grid is half of embedding-atlas's 1000², so σ 5/2.5 ≙ their 10/5 —
// identical smoothing in data units at a quarter of the blur cost.
const LABEL_GRID_SIZE = 500;
const LEVEL_SIGMAS: [number, number] = [5, 2.5];
const DEFAULT_DENSITY_THRESHOLD = 0.005;

// ---------------------------------------------------------------------------
// Cluster generation
// ---------------------------------------------------------------------------

/**
 * Pure post-processing of a findClusters result: converts pixel-space
 * centers/rects into data space via the density map's bounds and drops
 * clusters below the relative density threshold (per generate call, matching
 * embedding-atlas). Split from the WASM call so it is unit-testable.
 */
export function clustersFromFindClustersResult(
  rawClusters: Cluster[],
  map: Pick<DensityMapResult, 'width' | 'height' | 'xMin' | 'xMax' | 'yMin' | 'yMax'>,
  level: 0 | 1,
  densityThreshold: number,
): DensityCluster[] {
  // Inverse of computeDensityMap's binning: gx = round((x − xMin)·(w−1)/span)
  const xStep = (map.xMax - map.xMin) / (map.width - 1);
  const yStep = (map.yMax - map.yMin) / (map.height - 1);
  const toDataX = (gx: number) => map.xMin + gx * xStep;
  const toDataY = (gy: number) => map.yMin + gy * yStep;

  const maxSumDensity = rawClusters.reduce((max, c) => Math.max(max, c.sumDensity), 0);

  const clusters: DensityCluster[] = [];
  for (const c of rawClusters) {
    if (!c.boundaryRectApproximation || c.boundaryRectApproximation.length === 0) continue;
    if (maxSumDensity > 0 && c.sumDensity / maxSumDensity <= densityThreshold) continue;
    clusters.push({
      x: toDataX(c.meanX),
      y: toDataY(c.meanY),
      sumDensity: c.sumDensity,
      rects: c.boundaryRectApproximation.map(([x1, y1, x2, y2]) => ({
        xMin: toDataX(Math.min(x1, x2)),
        xMax: toDataX(Math.max(x1, x2)),
        yMin: toDataY(Math.min(y1, y2)),
        yMax: toDataY(Math.max(y1, y2)),
      })),
      level,
    });
  }
  return clusters;
}

/**
 * Two-bandwidth cluster generation: a coarse pass (level 0) and a fine pass
 * (level 1), each CPU density map → WASM findClusters. Yields to the event
 * loop between passes.
 */
export async function generateDensityClusters(
  points: { x: number; y: number }[],
  opts?: { gridSize?: number; densityThreshold?: number },
): Promise<DensityCluster[]> {
  const gridSize = opts?.gridSize ?? LABEL_GRID_SIZE;
  const densityThreshold = opts?.densityThreshold ?? DEFAULT_DENSITY_THRESHOLD;

  const result: DensityCluster[] = [];
  for (let level = 0 as 0 | 1; level <= 1; level++) {
    // computeDensityMap only reads x/y off its Point2D[] parameter.
    const map = computeDensityMap(points as Parameters<typeof computeDensityMap>[0], gridSize, gridSize, LEVEL_SIGMAS[level]);
    const raw = await findClusters(map.density, map.width, map.height);
    result.push(...clustersFromFindClustersResult(raw, map, level, densityThreshold));
    // Yield between the two blur+cluster passes to keep the main thread live.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Hybrid label text
// ---------------------------------------------------------------------------

/** Spatial index: 200×200 bin grid over the union bbox of all cluster rects. */
class ClusterMembership {
  private binToClusters = new Map<number, number[]>();
  private xMin = 0;
  private yMin = 0;
  private xStep = 1;
  private yStep = 1;
  private clusters: DensityCluster[];

  constructor(clusters: DensityCluster[]) {
    this.clusters = clusters;
    let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
    for (const c of clusters) {
      for (const r of c.rects) {
        if (r.xMin < xMin) xMin = r.xMin;
        if (r.xMax > xMax) xMax = r.xMax;
        if (r.yMin < yMin) yMin = r.yMin;
        if (r.yMax > yMax) yMax = r.yMax;
      }
    }
    if (!(xMin < xMax && yMin < yMax)) return;
    this.xMin = xMin;
    this.yMin = yMin;
    this.xStep = (xMax - xMin) / 200;
    this.yStep = (yMax - yMin) / 200;
    for (let i = 0; i < clusters.length; i++) {
      for (const r of clusters[i].rects) {
        const x0 = Math.floor((r.xMin - xMin) / this.xStep);
        const x1 = Math.floor((r.xMax - xMin) / this.xStep);
        const y0 = Math.floor((r.yMin - yMin) / this.yStep);
        const y1 = Math.floor((r.yMax - yMin) / this.yStep);
        for (let xi = x0; xi <= x1; xi++) {
          for (let yi = y0; yi <= y1; yi++) {
            const key = yi * 32768 + xi;
            const list = this.binToClusters.get(key);
            if (list) {
              if (list[list.length - 1] !== i) list.push(i);
            } else {
              this.binToClusters.set(key, [i]);
            }
          }
        }
      }
    }
  }

  /** Cluster indices whose rects contain the point (exact test on candidates). */
  clustersAt(x: number, y: number): number[] {
    const xi = Math.floor((x - this.xMin) / this.xStep);
    const yi = Math.floor((y - this.yMin) / this.yStep);
    const candidates = this.binToClusters.get(yi * 32768 + xi);
    if (!candidates) return [];
    return candidates.filter((i) =>
      this.clusters[i].rects.some(
        (r) => x >= r.xMin && x <= r.xMax && y >= r.yMin && y <= r.yMax,
      ),
    );
  }
}

/**
 * Hybrid text resolution. With `topicOf`, each cluster gets the majority
 * topic_label among its member points (ties break lexicographically); clusters
 * with no topic-labeled members fall back to the c-TF-IDF summarizer. Without
 * `topicOf`, every cluster is summarized.
 */
export function resolveClusterLabelTexts(
  clusters: DensityCluster[],
  points: { x: number; y: number; text: string }[],
  topicOf: ((pointIndex: number) => string | null) | null,
): { text: string; topicLabel: string | null }[] {
  const results: { text: string; topicLabel: string | null }[] =
    clusters.map(() => ({ text: '', topicLabel: null }));
  if (clusters.length === 0) return results;

  const needsSummary = new Set<number>();

  if (topicOf) {
    const membership = new ClusterMembership(clusters);
    const topicCounts: Map<string, number>[] = clusters.map(() => new Map());
    for (let i = 0; i < points.length; i++) {
      const topic = topicOf(i);
      if (topic == null || topic === '') continue;
      for (const ci of membership.clustersAt(points[i].x, points[i].y)) {
        const counts = topicCounts[ci];
        counts.set(topic, (counts.get(topic) ?? 0) + 1);
      }
    }
    for (let ci = 0; ci < clusters.length; ci++) {
      let best: string | null = null;
      let bestCount = 0;
      for (const [topic, count] of topicCounts[ci]) {
        if (count > bestCount || (count === bestCount && best !== null && topic < best)) {
          best = topic;
          bestCount = count;
        }
      }
      if (best !== null) {
        results[ci] = { text: best, topicLabel: best };
      } else {
        needsSummary.add(ci);
      }
    }
  } else {
    for (let ci = 0; ci < clusters.length; ci++) needsSummary.add(ci);
  }

  if (needsSummary.size > 0) {
    const summarizer = new TextSummarizer({ regions: clusters.map((c) => c.rects) });
    summarizer.add({
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      text: points.map((p) => p.text),
    });
    const summaries = summarizer.summarize(4);
    for (const ci of needsSummary) {
      // Single-line canvas labels: top 3 keywords, dash-joined like the atlas.
      results[ci] = { text: summaries[ci].slice(0, 3).join('-'), topicLabel: null };
    }
  }

  return results;
}

/**
 * Collapse labeled clusters that resolve to the same text, keeping the densest
 * (highest sumDensity) blob per unique label.
 *
 * The pipeline finds blobs *by density* at two bandwidths, so one topic can
 * yield several blobs — surfaced at both the coarse and fine level, or split
 * across a spatially multimodal region. In the hybrid path those all take the
 * same majority topic_label, so without this step the same title is drawn more
 * than once. c-TF-IDF (no-topic) labels differ per region, so only genuine
 * duplicates collapse. Because coarse blobs integrate more mass, the surviving
 * blob is usually the coarse (level-0) one, which also keeps the label visible
 * across the whole zoom range instead of vanishing past the fine-level band.
 */
export function dedupeLabeledClusters(labeled: LabeledDensityCluster[]): LabeledDensityCluster[] {
  const bestByText = new Map<string, LabeledDensityCluster>();
  for (const c of labeled) {
    if (!c.text) continue;
    const existing = bestByText.get(c.text);
    if (!existing || c.sumDensity > existing.sumDensity) {
      bestByText.set(c.text, c);
    }
  }
  return Array.from(bestByText.values());
}

// ---------------------------------------------------------------------------
// Placement (labels.ts layoutLabels port)
// ---------------------------------------------------------------------------

const LEVEL_FONT_SIZES: [number, number] = [14, 12];
const GLOBAL_MAX_SCALE = 2;
const PAD = 4;

/**
 * Run dynamic label placement over the (deduped) cluster labels.
 *
 * Every label is a candidate at every zoom; `dynamicLabelPlacement` alone
 * declutters — at the overview it shows the highest-priority (densest) labels
 * and suppresses those that would overlap, revealing more as you zoom in and
 * the overlaps clear. We deliberately do NOT impose embedding-atlas's two-level
 * scale bands: those exist to swap coarse keyword text for finer text on
 * zoom-in, but our hybrid labels reuse the same topic name at both bandwidths,
 * so the band's only effect would be to hard-hide the fine-level labels at the
 * overview. Level still drives font size for a coarse/fine visual hierarchy.
 * (computeCurrentScale = current span / initial span; 1 = full view.)
 */
export function computeDensityLabelPlacements(
  labeled: LabeledDensityCluster[],
  ctx: CanvasRenderingContext2D,
  initialRanges: AxisRanges,
  plotArea: PlotArea,
  colorFor: (topicLabel: string | null) => string,
): DensityLabelPlacement[] {
  const visible = labeled.filter((c) => c.text.length > 0);
  if (visible.length === 0) return [];

  const entries: DensityLabelPlacement[] = [];
  const appleLabels: Label[] = [];

  for (const cluster of visible) {
    const fontSize = LEVEL_FONT_SIZES[cluster.level];
    ctx.font = `bold ${fontSize}px Geist Mono, monospace`;
    const textWidth = ctx.measureText(cluster.text).width;
    const halfW = textWidth / 2 + PAD;
    const halfH = fontSize * 0.6 + PAD;

    const screen = projectDataToScreen(cluster.x, cluster.y, initialRanges, plotArea);

    entries.push({
      label: cluster.text,
      color: colorFor(cluster.topicLabel),
      dataX: cluster.x,
      dataY: cluster.y,
      textWidth,
      placement: null,
      priority: cluster.sumDensity,
      fontSize,
      level: cluster.level,
      topicLabel: cluster.topicLabel,
    });

    appleLabels.push({
      bounds: {
        xMin: screen.x - halfW,
        yMin: screen.y - halfH,
        xMax: screen.x + halfW,
        yMax: screen.y + halfH,
      },
      locationAtZero: { x: screen.x, y: screen.y },
      priority: cluster.sumDensity,
    });
  }

  const placements = dynamicLabelPlacement(appleLabels, { globalMaxScale: GLOBAL_MAX_SCALE });
  return entries.map((entry, i) => ({ ...entry, placement: placements[i] }));
}
