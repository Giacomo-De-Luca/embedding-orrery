'use client';

import type { Ref } from 'react';
import type { AssociationLegend } from '@/lib/utils/associationView';

interface HeatmapLegendProps {
  ref?: Ref<HTMLDivElement>;
  /** What the colour encodes, e.g. "Enrichment" or "Spearman ρ". */
  title: string;
  legend: AssociationLegend;
  /** Swatch for cells drawn in the neutral colour (empty cells, the diagonal); omitted when null. */
  emptyColor: string | null;
  emptyLabel: string;
}

/** Scale strip of a heatmap: start and end labels, a centre tick on diverging scales, and the neutral swatch. */
export function HeatmapLegend({ ref, title, legend, emptyColor, emptyLabel }: HeatmapLegendProps) {
  return (
    <div ref={ref} className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      <span>{title}</span>
      <div className="flex items-center gap-2">
        <span>{legend.start}</span>
        <div className="relative h-2.5 w-44 rounded-full" style={{ background: legend.gradient }}>
          {legend.middle && (
            <span aria-hidden className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-foreground/40" />
          )}
        </div>
        <span>{legend.end}</span>
      </div>
      {legend.middle && <span>Centre: {legend.middle.toLowerCase()}</span>}
      {emptyColor && (
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ backgroundColor: emptyColor }} />
          {emptyLabel}
        </span>
      )}
    </div>
  );
}
