import { useDeferredValue, useMemo } from 'react';
import type { MetadataPoint } from '../types/types';
import { ContingencyTable } from '../utils/contingencyTable';
import { getUnclusteredValues } from '../utils/categoryColors';

/** Most values per heatmap axis; smaller ones fold into "Other". */
export const MAX_ASSOCIATION_VALUES = 50;

/**
 * Cross-tabulates `groupField` (rows) against `acrossField` (columns) over the
 * points left visible by the text-search and temporal filters. Unclustered
 * topic noise is left out, as in the category list. Returns null until two
 * different fields are chosen; pass null fields to skip the work entirely
 * (e.g. while the section is collapsed).
 */
export function useFieldAssociation(
  points: MetadataPoint[],
  groupField: string | null,
  acrossField: string | null,
  mutedIndices: Set<number> | null | undefined,
): ContingencyTable | null {
  // Filter changes rebuild the table in a deferred render, so typing a search stays responsive.
  const deferredMuted = useDeferredValue(mutedIndices);

  return useMemo(() => {
    if (!groupField || !acrossField || groupField === acrossField || points.length === 0) return null;
    return ContingencyTable.fromPoints(points, groupField, acrossField, {
      exclude: deferredMuted,
      rowExclude: getUnclusteredValues(groupField),
      colExclude: getUnclusteredValues(acrossField),
      maxValues: MAX_ASSOCIATION_VALUES,
    });
  }, [points, groupField, acrossField, deferredMuted]);
}
