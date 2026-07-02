'use client';

import { useMemo, useState } from 'react';
import { cn } from '@/lib/utils/utils';

interface TokenStripProps {
  tokens: string[];
  values: number[];
  maxValueTokenIndex: number;
  /** Global max across all activations for consistent scaling */
  globalMax?: number;
  /** Called when hovering a token (value) or leaving (null) */
  onHoverActivation?: (value: number | null) => void;
}

/**
 * Renders a sequence of tokens with activation heatmap coloring.
 * Higher activation values get more intense orange/red backgrounds.
 * The max-activation token gets a highlighted border.
 */
export function TokenStrip({ tokens, values, maxValueTokenIndex, globalMax, onHoverActivation }: TokenStripProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const maxVal = globalMax ?? Math.max(...values);

  // Pre-compute colors for each token
  const tokenColors = useMemo(() => {
    return values.map((v) => {
      if (maxVal === 0) return 'transparent';
      const intensity = Math.max(0, v) / maxVal;
      // Orange-red heatmap: low = transparent, high = saturated orange
      const r = 255;
      const g = Math.round(165 - intensity * 100); // 165 → 65
      const b = Math.round(50 - intensity * 50);   // 50 → 0
      const a = Math.min(0.9, intensity * 0.85 + 0.05);
      return `rgba(${r}, ${g}, ${b}, ${a})`;
    });
  }, [values, maxVal]);

  return (
    <div className="relative leading-relaxed font-mono text-xs">
      <div className="flex flex-wrap">
        {tokens.map((token, i) => {
          const isMax = i === maxValueTokenIndex;
          const isHovered = i === hoveredIdx;

          return (
            <span
              key={i}
              className={cn(
                'px-[1px] py-[1px] cursor-default transition-shadow duration-75 whitespace-pre',
                isMax && 'ring-1 ring-orange-500 dark:ring-orange-400 rounded-sm font-semibold',
              )}
              style={{ backgroundColor: tokenColors[i] }}
              onMouseEnter={() => { setHoveredIdx(i); onHoverActivation?.(values[i]); }}
              onMouseLeave={() => { setHoveredIdx(null); onHoverActivation?.(null); }}
            >
              {/* Dark text when background is light, light text when intense */}
              <span className={values[i] / maxVal > 0.5 ? 'text-white dark:text-white' : 'text-foreground'}>
                {token}
              </span>
              {isHovered && (
                <span className="absolute z-10 -mt-8 ml-0 px-2 py-1 text-[10px] bg-popover text-popover-foreground border rounded shadow-md whitespace-nowrap pointer-events-none">
                  {values[i].toFixed(4)}
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}
