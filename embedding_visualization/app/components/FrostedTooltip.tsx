'use client';

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

function truncateValue(value: unknown, maxLen = 200): string {
  const str = String(value ?? '');
  return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

export function FrostedTooltip({ data }: FrostedTooltipProps) {
  if (!data?.visible) return null;

  const truncatedDoc = data.document && data.document.length > 200
    ? data.document.substring(0, 200) + '...'
    : data.document;

  const extraFields = data.tooltipFields && data.metadata
    ? data.tooltipFields.filter(f => data.metadata![f] !== undefined && data.metadata![f] !== null && data.metadata![f] !== '')
    : [];

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
        backdropFilter: 'blur(8px) saturate(150%)',
        WebkitBackdropFilter: 'blur(12px) saturate(150%)',
      }}
    >
      <div className="font-medium">{data.label}</div>
      {extraFields.length > 0 && (
        <div className="text-xs mt-1.5 space-y-0.5 opacity-80">
          {extraFields.map(field => (
            <div key={field}>
              <span className="font-medium">{formatFieldName(field)}:</span>{' '}
              {truncateValue(data.metadata![field])}
            </div>
          ))}
        </div>
      )}
      {truncatedDoc && <div className="text-sm mt-1 opacity-70">{truncatedDoc}</div>}
    </div>
  );
}
