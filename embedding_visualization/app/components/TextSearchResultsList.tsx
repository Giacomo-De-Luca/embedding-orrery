'use client';

import * as React from 'react';
import { Badge } from '@/lib/ui-primitives/badge';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import type { Point2D, Point3D } from '../../lib/types/types';

interface TextSearchResultsListProps {
  results: (Point2D | Point3D)[];
  selectedPointId?: string | null;
  onResultClick?: (point: Point2D | Point3D) => void;
  maxHeight?: number;
  categoryField?: string | null;
}

export function TextSearchResultsList({
  results,
  selectedPointId,
  onResultClick,
  maxHeight = 300,
  categoryField,
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
        className="rounded-md border"
        style={{ height: maxHeight }}
      >
        <div className="p-1">
          {results.map((point) => (
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
                  {point.label}
                </span>
                {categoryField && point.category && (
                  <Badge variant="secondary" className="text-xs shrink-0">
                    {point.category}
                  </Badge>
                )}
              </div>
              {point.document && point.document !== point.label && (
                <p className="text-xs text-muted-foreground  mt-0.5">
                  {point.document}
                 {/*{point.document.slice(0, 80)}
                  {point.document.length > 80 ? '...' : ''}*/}
                </p>
              )}
            </button>
          ))}
        </div>
        <ScrollBar orientation="vertical" />
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
