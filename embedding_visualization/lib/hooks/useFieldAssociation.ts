import { useDeferredValue, useMemo } from 'react';
import type { MetadataPoint } from '../types/types';
import { ContingencyTable } from '../utils/contingencyTable';
import {
  AssociationFields,
  AssociationStatistics,
  FieldAxis,
  type AssociationFieldOption,
  type AssociationStatistic,
  type BinSettings,
} from '../utils/fieldAxis';


export interface FieldAssociation {
  table: ContingencyTable;
  group: FieldAxis;
  across: FieldAxis;
  statistic: AssociationStatistic;
}

/**
 * Cross-tabulates `group` (rows) against `across` (columns) over the points
 * left visible by the text-search and temporal filters, with the pair's
 * statistic on the raw values. Numeric fields are binned per `bins` over all
 * points, so filtering never moves the bin edges (the label style is display
 * only and rebuilds nothing); unclustered topic noise is
 * left out, as in the category list. Pass null fields to skip the work (e.g.
 * while the section is collapsed).
 */
export function useFieldAssociation(
  points: MetadataPoint[],
  group: AssociationFieldOption | null,
  across: AssociationFieldOption | null,
  mutedIndices: Set<number> | null | undefined,
  bins: BinSettings,
): FieldAssociation | null {
  // Filter changes rebuild the table in a deferred render, so typing a search stays responsive.
  const deferredMuted = useDeferredValue(mutedIndices);
  const { count, method } = bins;

  const groupAxis = useMemo(
    () => (group ? FieldAxis.of(group, points, { count, method }) : null),
    [group, points, count, method],
  );
  const acrossAxis = useMemo(
    () => (across ? FieldAxis.of(across, points, { count, method }) : null),
    [across, points, count, method],
  );

  // Spearman between two numeric fields reads raw values only, so new bin settings must not recompute it.
  const numericPair = useMemo(
    () => (group?.kind === 'numeric' && across?.kind === 'numeric' && group.field !== across.field
      ? AssociationStatistics.numericPair(points, group.field, across.field, deferredMuted)
      : null),
    [group, across, points, deferredMuted],
  );

  return useMemo(() => {
    if (!groupAxis || !acrossAxis || groupAxis.field === acrossAxis.field || points.length === 0) return null;
    const table = ContingencyTable.fromPoints(points, groupAxis.field, acrossAxis.field, {
      exclude: deferredMuted,
      rowKey: groupAxis.keyOf,
      colKey: acrossAxis.keyOf,
      rowExclude: groupAxis.excluded,
      colExclude: acrossAxis.excluded,
      maxValues: AssociationFields.MAX_AXIS_VALUES,
    });
    const statistic = numericPair ?? AssociationStatistics.of(points, groupAxis, acrossAxis, table, deferredMuted);
    return { table, group: groupAxis, across: acrossAxis, statistic };
  }, [points, groupAxis, acrossAxis, deferredMuted, numericPair]);
}
