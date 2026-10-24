'use client';

import { useMemo } from 'react';
import { PointLabelRanking, type LabelPoint, type RankedPointLabels } from '../utils/labels/pointLabels';
import type { TextMeasurer } from '../utils/labels/labelStyle';

/**
 * The search-result labels a plot draws, or null for none: the highlighted
 * points ranked once per change of the highlights or the selection (not per
 * frame), the selected point's label first. With a measurer (`useLabelMeasurer`)
 * their text is measured here too, again once the label fonts have loaded.
 */
export function useSearchLabels<P extends LabelPoint>(
  showLabels: boolean,
  highlightedPoints: readonly P[],
  highlights: ReadonlyMap<number, number> | null | undefined,
  selectedIndex: number | null | undefined,
  measurer: TextMeasurer | null,
): RankedPointLabels<P> | null {
  return useMemo(() => {
    if (!showLabels || !highlights || highlightedPoints.length === 0) return null;
    return PointLabelRanking.rank(highlightedPoints, highlights, selectedIndex, measurer);
  }, [showLabels, highlightedPoints, highlights, selectedIndex, measurer]);
}
