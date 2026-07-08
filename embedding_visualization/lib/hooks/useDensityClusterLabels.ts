/**
 * Async pipeline driver for density-based cluster labels (see
 * app/utils/densityClusterLabels.ts). Debounced and generation-token
 * cancelled; on any failure (e.g. WASM fetch) it warns once, permanently
 * falls back, and the caller keeps using topic-centroid placements.
 */

import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import type { Point2D, NestedColorMap } from '../types/types';
import {
  computeDensityLabelPlacements,
  dedupeLabeledClusters,
  generateDensityClusters,
  resolveClusterLabelTexts,
  type DensityLabelPlacement,
} from '../../app/utils/densityClusterLabels';
import { plotAreaFromFullLayout, type AxisRanges } from '../../app/utils/labelPlacement2D';

const DEBOUNCE_MS = 300;

interface UseDensityClusterLabelsArgs {
  /** densityMode && showClusterLabels && plot ready. */
  enabled: boolean;
  visiblePoints: Point2D[];
  /** Metadata field holding topic labels, or null when the collection has none. */
  topicLabelField: string | null;
  colorMap: Record<string, string>;
  nestedColorMap?: NestedColorMap | null;
  isDark: boolean;
  initialRanges: AxisRanges | null;
  graphDivRef: RefObject<any>;
  width: number;
  height: number;
}

export function useDensityClusterLabels({
  enabled,
  visiblePoints,
  topicLabelField,
  colorMap,
  nestedColorMap,
  isDark,
  initialRanges,
  graphDivRef,
  width,
  height,
}: UseDensityClusterLabelsArgs): DensityLabelPlacement[] | null {
  const [placements, setPlacements] = useState<DensityLabelPlacement[] | null>(null);
  const generationRef = useRef(0);
  const failedRef = useRef(false);

  // Neutral color for summarizer-derived (non-topic) labels.
  const neutralColor = isDark ? '#94a3b8' : '#475569';

  // Latest lookup values via refs so the async effect doesn't re-fire on
  // every colorMap identity change (colors are applied at compute time).
  const lookupRef = useRef({ colorMap, nestedColorMap, neutralColor });
  lookupRef.current = { colorMap, nestedColorMap, neutralColor };

  const measureCtx = useMemo(() => {
    if (typeof document === 'undefined') return null;
    return document.createElement('canvas').getContext('2d');
  }, []);

  useEffect(() => {
    const generation = ++generationRef.current;

    if (!enabled || failedRef.current || !initialRanges || !measureCtx || visiblePoints.length === 0) {
      setPlacements(null);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const clusters = await generateDensityClusters(visiblePoints);
        if (generationRef.current !== generation) return;

        const textPoints = visiblePoints.map((p) => ({
          x: p.x,
          y: p.y,
          text: p.document || p.label || '',
        }));
        const topicOf = topicLabelField
          ? (i: number) => {
              const v = visiblePoints[i].metadata?.[topicLabelField];
              return v === null || v === undefined || v === '' ? null : String(v);
            }
          : null;
        const texts = resolveClusterLabelTexts(clusters, textPoints, topicOf);
        // Blobs sharing a resolved title (same topic surfaced at both
        // bandwidths or across a multimodal region) collapse to their densest.
        const labeled = dedupeLabeledClusters(clusters.map((c, i) => ({ ...c, ...texts[i] })));

        const { colorMap: cm, nestedColorMap: ncm, neutralColor: neutral } = lookupRef.current;
        const plotArea = plotAreaFromFullLayout(graphDivRef.current?._fullLayout, width, height);
        const result = computeDensityLabelPlacements(
          labeled,
          measureCtx,
          initialRanges,
          plotArea,
          (topicLabel) =>
            topicLabel
              ? (ncm?.topicColors?.[topicLabel] ?? cm[topicLabel]) || neutral
              : neutral,
        );
        if (generationRef.current !== generation) return;
        setPlacements(result);
      } catch (err) {
        if (!failedRef.current) {
          failedRef.current = true;
          console.warn('Density cluster labels unavailable, falling back to topic centroids:', err);
        }
        if (generationRef.current === generation) setPlacements(null);
      }
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
    // graphDivRef is a ref; width/height changes only affect screen-space
    // measurement, which the debounce re-run covers.
  }, [enabled, visiblePoints, topicLabelField, initialRanges, measureCtx, graphDivRef, width, height]);

  return placements;
}
