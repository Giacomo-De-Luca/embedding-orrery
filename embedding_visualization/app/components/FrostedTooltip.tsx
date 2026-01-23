'use client';

interface TooltipData {
  x: number;
  y: number;
  label: string;
  document?: string;
  visible: boolean;
}

interface FrostedTooltipProps {
  data: TooltipData | null;
}

export type { TooltipData };

export function FrostedTooltip({ data }: FrostedTooltipProps) {
  if (!data?.visible) return null;

  const truncatedDoc = data.document && data.document.length > 200
    ? data.document.substring(0, 200) + '...'
    : data.document;

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
      {truncatedDoc && <div className="text-sm mt-1">{truncatedDoc}</div>}
    </div>
  );
}
