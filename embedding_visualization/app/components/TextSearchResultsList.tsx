'use client';

import type { CSSProperties } from 'react';
import { Badge } from '@/lib/ui-primitives/badge';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import type { Point2D, Point3D } from '../../lib/types/types';
import { HighlightedText } from '../utils/highlightedText';

// Max result rows rendered — an uncapped list can mean thousands of DOM nodes
const RESULTS_PREVIEW_LIMIT = 100;

interface TextSearchResultsListProps {
  results: (Point2D | Point3D)[];
  selectedPointId?: string | null;
  onResultClick?: (point: Point2D | Point3D) => void;
  maxHeight?: number;
  categoryField?: string | null;
  searchQuery?: string;
}

export function TextSearchResultsList({
  results,
  selectedPointId,
  onResultClick,
  maxHeight = 300,
  categoryField,
  searchQuery,
}: TextSearchResultsListProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-muted-foreground pb-2">
        Text matches ({results.length})
      </p>
      <ScrollArea
        className="rounded-md border [&>[data-radix-scroll-area-viewport]>div]:block!"
        viewportClassName="max-h-(--results-max-h)"
        style={{ '--results-max-h': `${maxHeight}px` } as CSSProperties}
      >
        <div className="p-1">
          {results.slice(0, RESULTS_PREVIEW_LIMIT).map((point) => (
            <button
              key={point.id}
              onClick={() => onResultClick?.(point)}
              className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors hover:bg-accent hover:text-accent-foreground ${selectedPointId === point.id
                  ? 'bg-accent text-accent-foreground'
                  : ''
                }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-medium flex-1">
                  <HighlightedText text={point.label} query={searchQuery ?? ''} />
                </span>
                {categoryField && point.category && (
                  <Badge variant="secondary" className="text-xs shrink-0">
                    {point.category}
                  </Badge>
                )}
              </div>
              {point.document && point.document !== point.label && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  <HighlightedText text={point.document} query={searchQuery ?? ''} />
                </p>
              )}
            </button>
          ))}
          {results.length > RESULTS_PREVIEW_LIMIT && (
            <p className="px-3 py-1.5 text-xs text-muted-foreground">
              Showing first {RESULTS_PREVIEW_LIMIT} of {results.length.toLocaleString()} matches
            </p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
