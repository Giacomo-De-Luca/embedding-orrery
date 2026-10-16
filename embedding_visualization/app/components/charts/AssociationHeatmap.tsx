'use client';

import { Fragment, useCallback, useMemo, useRef, useState, type RefObject } from 'react';
import { AssociationView } from '@/lib/utils/associationView';
import { cn } from '@/lib/utils/utils';

const CELL_HEIGHT = 22;
const CELL_WIDTH = 22;
/** Wider cells leave room for "1.2K" / "2.5×" labels. */
const CELL_WIDTH_WITH_VALUES = 36;
const TOOLTIP_WIDTH = 280;

interface AssociationHeatmapProps {
  view: AssociationView;
  /** Display order of the table's rows / columns (index permutations). */
  rowOrder: number[];
  colOrder: number[];
  showValues: boolean;
  /** The full grid, including cells outside the scroll viewport, for PNG export. */
  exportRef?: RefObject<HTMLDivElement | null>;
}

/** Hovered or keyboard-focused cell: display position, highlight box (grid coordinates) and tooltip anchor (wrapper coordinates). */
interface ActiveCell {
  r: number;
  c: number;
  box: { left: number; top: number; width: number; height: number };
  x: number;
  y: number;
  /** Open the tooltip to the left of the anchor when it would overflow on the right. */
  flip: boolean;
}

function formatPercent(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}

/**
 * Cell grid of an {@link AssociationView}: sticky row and column labels, a
 * 2px surface gap between cells, and one tooltip that follows the pointer or
 * the arrow keys. Cells carry their display position in data attributes so a
 * single delegated handler serves the whole grid; the highlight ring and the
 * tooltip are overlays, so hovering never re-renders the cells.
 */
export function AssociationHeatmap({ view, rowOrder, colOrder, showValues, exportRef }: AssociationHeatmapProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState<ActiveCell | null>(null);
  const { table } = view;
  const cellWidth = showValues ? CELL_WIDTH_WITH_VALUES : CELL_WIDTH;

  const rowLabels = useMemo(
    () => table.rows.map(key => AssociationView.axisLabel(table.rowField, key)),
    [table],
  );
  const colLabels = useMemo(
    () => table.cols.map(key => AssociationView.axisLabel(table.colField, key)),
    [table],
  );

  const cellElement = useCallback(
    (r: number, c: number) => gridRef.current?.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`) ?? null,
    [],
  );

  /** Active-cell state for `element`, anchored at a viewport point (the pointer, or the cell's corner). */
  const activate = useCallback((element: HTMLElement, clientX: number, clientY: number): ActiveCell | null => {
    const wrapper = wrapperRef.current;
    if (!wrapper) return null;
    const bounds = wrapper.getBoundingClientRect();
    const x = clientX - bounds.left;
    return {
      r: Number(element.dataset.r),
      c: Number(element.dataset.c),
      box: {
        left: element.offsetLeft,
        top: element.offsetTop,
        width: element.offsetWidth,
        height: element.offsetHeight,
      },
      x,
      y: clientY - bounds.top,
      flip: x + TOOLTIP_WIDTH + 24 > wrapper.clientWidth,
    };
  }, []);

  const handlePointerMove = useCallback((event: React.PointerEvent) => {
    const cell = (event.target as HTMLElement).closest<HTMLElement>('[data-r]');
    setActive(cell ? activate(cell, event.clientX, event.clientY) : null);
  }, [activate]);

  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const move = moves[event.key];
    if (!move || rowOrder.length === 0 || colOrder.length === 0) return;
    event.preventDefault();
    const r = Math.min(rowOrder.length - 1, Math.max(0, (active?.r ?? 0) + (active ? move[0] : 0)));
    const c = Math.min(colOrder.length - 1, Math.max(0, (active?.c ?? 0) + (active ? move[1] : 0)));
    const element = cellElement(r, c);
    if (!element) return;
    element.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const rect = element.getBoundingClientRect();
    setActive(activate(element, rect.right, rect.top));
  }, [active, rowOrder.length, colOrder.length, cellElement, activate]);

  // The cells only change with the data or the layout, never with hover.
  const grid = useMemo(() => (
    <>
      <div className="sticky top-0 left-0 z-20 bg-background" />
      {colOrder.map((j, c) => (
        <div
          key={`col-${j}`}
          className="sticky top-0 z-10 flex h-32 items-end justify-center bg-background pb-1"
          data-col-header={c}
        >
          <span
            className="max-h-30 truncate text-[11px] text-muted-foreground [writing-mode:vertical-rl] rotate-180"
            title={colLabels[j]}
          >
            {colLabels[j]}
          </span>
        </div>
      ))}
      {rowOrder.map((i, r) => (
        <Fragment key={`row-${i}`}>
          <div
            className="sticky left-0 z-10 truncate bg-background pr-2 text-right text-[11px] text-muted-foreground"
            style={{ lineHeight: `${CELL_HEIGHT}px` }}
            title={rowLabels[i]}
          >
            {rowLabels[i]}
          </div>
          {colOrder.map((j, c) => (
            <div
              key={`cell-${i}-${j}`}
              data-r={r}
              data-c={c}
              className="flex items-center justify-center rounded-[2px]"
              style={{ backgroundColor: view.color(i, j), height: CELL_HEIGHT }}
            >
              {showValues && (
                <span
                  className={cn(
                    'pointer-events-none text-[10px] tabular-nums',
                    view.usesLightInk(i, j) ? 'text-white/90' : 'text-neutral-900/80',
                  )}
                >
                  {view.cellLabel(i, j)}
                </span>
              )}
            </div>
          ))}
        </Fragment>
      ))}
    </>
  ), [view, rowOrder, colOrder, rowLabels, colLabels, showValues]);

  // Positions past the end can linger for one render after the table shrinks.
  const i = active ? rowOrder[active.r] ?? -1 : -1;
  const j = active ? colOrder[active.c] ?? -1 : -1;

  return (
    <div ref={wrapperRef} className="relative h-full min-h-0">
      <div
        className="h-full overflow-auto rounded-md focus-visible:outline-2 focus-visible:outline-ring"
        role="grid"
        aria-label={`${table.rowField} by ${table.colField}. Use the arrow keys to read cells.`}
        tabIndex={0}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setActive(null)}
        onKeyDown={handleKeyDown}
        onBlur={() => setActive(null)}
      >
        <div
          ref={element => {
            gridRef.current = element;
            if (exportRef) exportRef.current = element;
          }}
          className="relative grid w-max gap-0.5 pr-2 pb-2"
          style={{ gridTemplateColumns: `minmax(8rem, 14rem) repeat(${colOrder.length}, ${cellWidth}px)` }}
        >
          {grid}
          {active && (
            <div
              aria-hidden
              data-export-exclude
              className="pointer-events-none absolute z-30 rounded-[3px] ring-2 ring-foreground"
              style={active.box}
            />
          )}
        </div>
      </div>

      {active && i >= 0 && j >= 0 && (
        <div
          className="frosted-tooltip pointer-events-none absolute z-40"
          style={{
            left: active.flip ? active.x - TOOLTIP_WIDTH - 14 : active.x + 14,
            top: active.y + 14,
            width: TOOLTIP_WIDTH,
            backdropFilter: 'blur(12px) saturate(150%)',
            WebkitBackdropFilter: 'blur(12px) saturate(150%)',
          }}
          role="status"
        >
          <CellTooltip view={view} i={i} j={j} rowLabel={rowLabels[i]} colLabel={colLabels[j]} />
        </div>
      )}
    </div>
  );
}

function CellTooltip({
  view,
  i,
  j,
  rowLabel,
  colLabel,
}: {
  view: AssociationView;
  i: number;
  j: number;
  rowLabel: string;
  colLabel: string;
}) {
  const { table, mode } = view;
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
        {rowLabel} · {colLabel}
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
