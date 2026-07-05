'use client';

import type { ReactNode } from 'react';
import { isHexColor, normalizeHex, formatMetadataValue } from '@/lib/utils/tooltipFormat';

interface TooltipData {
  x: number;
  y: number;
  label: string;
  document?: string;
  visible: boolean;
  metadata?: Record<string, unknown>;
  tooltipFields?: string[];
}

interface FrostedTooltipProps {
  data: TooltipData | null;
}

export type { TooltipData };

/** Convert snake_case or camelCase field names to Title Case */
function formatFieldName(field: string): string {
  return field
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/** Render a metadata value: color swatch for hex fields, formatted text otherwise. */
function renderFieldValue(value: unknown): ReactNode {
  if (isHexColor(value)) {
    const hex = normalizeHex(value);
    return (
      <span className="inline-flex items-center gap-1.5">
        <span
          className="inline-block h-3 w-3 shrink-0 rounded-sm ring-1 ring-black/20 dark:ring-white/20"
          style={{ backgroundColor: hex }}
        />
        <span className="font-mono tabular-nums">{hex}</span>
      </span>
    );
  }
  const isNumber = typeof value === 'number' && Number.isFinite(value);
  return (
    <span className={isNumber ? 'tabular-nums' : 'break-words'}>
      {formatMetadataValue(value)}
    </span>
  );
}

export function FrostedTooltip({ data }: FrostedTooltipProps) {
  if (!data?.visible) return null;

  const extraFields = data.tooltipFields && data.metadata
    ? data.tooltipFields.filter(f => data.metadata![f] !== undefined && data.metadata![f] !== null && data.metadata![f] !== '')
    : [];

  const hasFields = extraFields.length > 0;
  const hasDocument = Boolean(data.document);

  return (
    <div
      className="frosted-tooltip"
      style={{
        position: 'absolute',
        left: data.x + 12,
        top: data.y - 10,
        pointerEvents: 'none',
        zIndex: 1000,
        // Inline backdrop-filter to ensure it works over WebGL canvas
        backdropFilter: 'blur(12px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
      }}
    >
      <div className="text-sm font-semibold break-words">{data.label}</div>

      {hasFields && (
        <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
          {extraFields.map(field => (
            <div key={field} className="contents">
              <span className="opacity-60">{formatFieldName(field)}</span>
              <span className="min-w-0">{renderFieldValue(data.metadata![field])}</span>
            </div>
          ))}
        </div>
      )}

      {hasDocument && (
        <div
          className={`text-xs break-words opacity-70 ${hasFields ? 'mt-2 border-t border-foreground/15 pt-2' : 'mt-2'}`}
          style={{
            display: '-webkit-box',
            WebkitLineClamp: 4,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {data.document}
        </div>
      )}
    </div>
  );
}
