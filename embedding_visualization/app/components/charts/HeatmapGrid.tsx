'use client';

import { Fragment, useCallback, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import type { HeatmapModel } from '@/lib/utils/heatmapModel';
import { cn } from '@/lib/utils/utils';

const CELL_HEIGHT = 22;
const CELL_WIDTH = 22;
/** Wider cells leave room for "1.2K" / "2.5×" / "−0.05" labels. */
const CELL_WIDTH_WITH_VALUES = 38;
const TOOLTIP_WIDTH = 280;
/** Rough tooltip height, to open it above the pointer near the bottom edge. */
const TOOLTIP_HEIGHT = 170;

interface HeatmapGridProps {
  model: HeatmapModel;
  /** Display order of the model's rows / columns (index permutations). */
  rowOrder: readonly number[];
  colOrder: readonly number[];
  showValues: boolean;
  /** Accessible name of the grid. */
  ariaLabel: string;
  /** Tooltip body of cell (i, j), in model indices. */
  renderTooltip: (i: number, j: number) => ReactNode;
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
  /** Open it above the anchor when it would overflow at the bottom; holds the wrapper height then. */
  above: number | null;
}

/**
 * Cell grid of a {@link HeatmapModel}: sticky row and column labels, a 2px
 * surface gap between cells, and one tooltip that follows the pointer or the
 * arrow keys. Cells carry their display position in data attributes so a
 * single delegated handler serves the whole grid; the highlight ring and the
 * tooltip are overlays, so hovering never re-renders the cells.
 */
export function HeatmapGrid({
  model,
  rowOrder,
  colOrder,
  showValues,
  ariaLabel,
  renderTooltip,
  exportRef,
}: HeatmapGridProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState<ActiveCell | null>(null);
  const cellWidth = showValues ? CELL_WIDTH_WITH_VALUES : CELL_WIDTH;
  const { rowLabels, colLabels } = model;

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
    const y = clientY - bounds.top;
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
      y,
      flip: x + TOOLTIP_WIDTH + 24 > wrapper.clientWidth,
      above: y + TOOLTIP_HEIGHT + 24 > wrapper.clientHeight ? wrapper.clientHeight : null,
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
              className="flex scroll-mt-[8.5rem] scroll-ml-[14.5rem] items-center justify-center rounded-[2px]"
              style={{ backgroundColor: model.color(i, j), height: CELL_HEIGHT }}
            >
              {showValues && (
                <span
                  className={cn(
                    'pointer-events-none text-[10px] tabular-nums',
                    model.usesLightInk(i, j) ? 'text-white/90' : 'text-neutral-900/80',
                  )}
                >
                  {model.cellLabel(i, j)}
                </span>
              )}
            </div>
          ))}
        </Fragment>
      ))}
    </>
  ), [model, rowOrder, colOrder, rowLabels, colLabels, showValues]);

  // Positions past the end can linger for one render after the model shrinks.
  const i = active ? rowOrder[active.r] ?? -1 : -1;
  const j = active ? colOrder[active.c] ?? -1 : -1;
  const hasCell = i >= 0 && j >= 0;

  return (
    <div ref={wrapperRef} className="relative h-full min-h-0">
      <div
        className="h-full overflow-auto rounded-md focus-visible:outline-2 focus-visible:outline-ring"
        role="group"
        aria-roledescription="heatmap"
        aria-label={`${ariaLabel}. Use the arrow keys to read cells.`}
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

      {/* Always mounted, so screen readers announce the first cell too. */}
      <div className="sr-only" aria-live="polite">
        {active && hasCell ? `${rowLabels[i]}, ${colLabels[j]}: ${model.cellLabel(i, j)}` : ''}
      </div>

      {active && hasCell && (
        <div
          aria-hidden
          className="frosted-tooltip pointer-events-none absolute z-40"
          style={{
            left: active.flip ? active.x - TOOLTIP_WIDTH - 14 : active.x + 14,
            ...(active.above === null ? { top: active.y + 14 } : { bottom: active.above - active.y + 14 }),
            width: TOOLTIP_WIDTH,
            backdropFilter: 'blur(12px) saturate(150%)',
            WebkitBackdropFilter: 'blur(12px) saturate(150%)',
          }}
        >
          {renderTooltip(i, j)}
        </div>
      )}
    </div>
  );
}
