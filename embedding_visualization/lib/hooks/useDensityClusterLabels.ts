/**
 * Async pipeline driver for density-based cluster labels (see
 * app/utils/densityClusterLabels.ts). Clustering is debounced and
 * generation-token cancelled; on any failure (e.g. WASM fetch) it warns once,
 * permanently falls back, and the caller keeps using category-centroid placements.
 * Placement runs separately on the latest clusters, so a new plot area, label font,
 * palette or theme re-places and recolours the labels without clustering again.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { Point2D, NestedColorMap } from '../types/types';
import {
  computeDensityLabelPlacements,
  dedupeLabeledClusters,
  DensityLabelColors,
  generateDensityClusters,
  resolveClusterLabelTexts,
  type DensityLabelPlacement,
  type LabeledDensityCluster,
} from '../../app/utils/densityClusterLabels';
import { CartesianView, type AxisRanges, type PlotArea } from '../utils/cartesianView';
import type { TextMeasurer } from '../utils/labels/labelStyle';

const DEBOUNCE_MS = 300;

interface UseDensityClusterLabelsArgs {
  /** densityMode && showClusterLabels && plot ready. */
  enabled: boolean;
  visiblePoints: Point2D[];
  /** Selected categorical field, topic fallback, or null for keyword summaries. */
  labelField: string | null;
  colorMap: Record<string, string>;
  nestedColorMap?: NestedColorMap | null;
  isDark: boolean;
  initialRanges: AxisRanges | null;
  /** The plot area Plotly has laid out, or null before it has. */
  plotArea: PlotArea | null;
  /** `useLabelMeasurer`: replaced once the label fonts load, so labels are measured again. */
  measurer: TextMeasurer | null;
}

/** Titled clusters, with the points and field they were made from. */
interface TitledClusters {
  points: Point2D[];
  field: string | null;
  clusters: LabeledDensityCluster[];
}

export function useDensityClusterLabels({
  enabled,
  visiblePoints,
  labelField,
  colorMap,
  nestedColorMap,
  isDark,
  initialRanges,
  plotArea,
  measurer,
}: UseDensityClusterLabelsArgs): DensityLabelPlacement[] | null {
  const [titled, setTitled] = useState<TitledClusters | null>(null);
  const generationRef = useRef(0);
  const failedRef = useRef(false);

  // Cluster the visible points and title each blob; the result lands asynchronously
  useEffect(() => {
    const generation = ++generationRef.current;
    if (!enabled || failedRef.current || visiblePoints.length === 0) return;

    const timer = setTimeout(async () => {
      try {
        const clusters = await generateDensityClusters(visiblePoints);
        if (generationRef.current !== generation) return;

        const textPoints = visiblePoints.map((p) => ({
          x: p.x,
          y: p.y,
          text: p.document || p.label || '',
          metadata: p.metadata,
        }));
        const texts = resolveClusterLabelTexts(clusters, textPoints, labelField);
        // Blobs sharing a resolved title (same category surfaced at both
        // bandwidths or across a multimodal region) collapse to their densest.
        const labeled = dedupeLabeledClusters(clusters.map((c, i) => ({ ...c, ...texts[i] })));
        setTitled({ points: visiblePoints, field: labelField, clusters: labeled });
      } catch (err) {
        if (!failedRef.current) {
          failedRef.current = true;
          console.warn('Density cluster labels unavailable, falling back to category centroids:', err);
        }
        if (generationRef.current === generation) setTitled(null);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [enabled, visiblePoints, labelField]);

  // Titles made from other points or another field are not shown while their
  // rebuild runs: the caller falls back to the centroid labels meanwhile.
  const current = enabled && titled?.points === visiblePoints && titled.field === labelField ? titled : null;

  // Place and colour the titled clusters for the current plot area and fonts
  return useMemo(() => {
    if (!current || !initialRanges || !plotArea || !measurer) return null;
    return computeDensityLabelPlacements(
      current.clusters,
      measurer,
      new CartesianView(initialRanges, plotArea),
      (category) => DensityLabelColors.resolve(category, current.field, colorMap, nestedColorMap, isDark),
    );
  }, [current, initialRanges, plotArea, measurer, colorMap, nestedColorMap, isDark]);
}
