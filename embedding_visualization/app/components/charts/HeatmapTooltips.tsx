'use client';

import { AssociationView } from '@/lib/utils/associationView';
import type { CorrelationMatrixView } from '@/lib/utils/correlationView';
import { FieldCorrelation, type CorrelationMethod } from '@/lib/utils/fieldCorrelation';

function formatPercent(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}

/**
 * Cross-tab cell: the mode's value first, then points, expected count and both
 * shares. `rowNote` / `colNote` add detail to an axis label (a numeric bin's
 * value range while bins are labelled by percentile).
 */
export function AssociationCellTooltip({
  view,
  i,
  j,
  rowNote,
  colNote,
}: {
  view: AssociationView;
  i: number;
  j: number;
  rowNote?: string;
  colNote?: string;
}) {
  const { table, mode } = view;
  const rowLabel = view.rowLabels[i];
  const colLabel = view.colLabels[j];
  const count = table.counts[i][j];
  const headline =
    mode === 'enrichment'
      ? `${AssociationView.formatRatio(table.ratio(i, j))} expected`
      : mode === 'rowShare'
        ? `${formatPercent(table.rowShare(i, j))} of ${rowLabel}`
        : mode === 'colShare'
          ? `${formatPercent(table.colShare(i, j))} of ${colLabel}`
          : `${count.toLocaleString()} points`;

  return (
    <>
      <div className="text-sm font-semibold tabular-nums">{headline}</div>
      <div className="mt-0.5 text-xs break-words opacity-70">
        {rowLabel}{rowNote ? ` (${rowNote})` : ''} · {colLabel}{colNote ? ` (${colNote})` : ''}
      </div>
      <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <span className="opacity-60">Points</span>
        <span className="tabular-nums">{count.toLocaleString()}</span>
        <span className="opacity-60">Expected if unrelated</span>
        <span className="tabular-nums">{table.expected(i, j).toFixed(1)}</span>
        <span className="min-w-0 truncate opacity-60">Share of {rowLabel}</span>
        <span className="tabular-nums">{formatPercent(table.rowShare(i, j))}</span>
        <span className="min-w-0 truncate opacity-60">Share of {colLabel}</span>
        <span className="tabular-nums">{formatPercent(table.colShare(i, j))}</span>
      </div>
    </>
  );
}

/** Correlation cell: the coefficient first, then the pair and the points it rests on. */
export function CorrelationCellTooltip({
  view,
  method,
  i,
  j,
}: {
  view: CorrelationMatrixView;
  method: CorrelationMethod;
  i: number;
  j: number;
}) {
  const symbol = method === 'spearman' ? 'ρ' : 'r';
  const value = view.value(i, j);
  const n = view.matrix.counts[i][j];
  return (
    <>
      <div className="text-sm font-semibold tabular-nums">
        {value === null ? 'Not defined' : `${symbol} = ${FieldCorrelation.format(value)}${i === j ? ' (same field)' : ''}`}
      </div>
      <div className="mt-0.5 text-xs break-words opacity-70">
        {view.rowLabels[i]} · {view.colLabels[j]}
      </div>
      <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
        <span className="opacity-60">Points with both values</span>
        <span className="tabular-nums">{n.toLocaleString()}</span>
      </div>
      {i !== j && value === null && (
        <p className="mt-1 text-xs opacity-60">Fewer than three shared points, or one field never varies.</p>
      )}
    </>
  );
}
