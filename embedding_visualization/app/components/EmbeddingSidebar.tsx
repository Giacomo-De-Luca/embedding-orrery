'use client';

import * as React from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { VisualizationControls } from './VisualizationControls';
import { SelectedPointCard } from './SelectedPointCard';
import { TextSearchResultsList } from './TextSearchResultsList';
import type { VisualizationState, Point2D, Point3D, CategoryFieldOption } from '../../lib/types/types';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';

interface EmbeddingSidebarProps extends React.ComponentProps<typeof Sidebar> {
  state: VisualizationState;
  onStateChange: (newState: Partial<VisualizationState>) => void;
  embeddingDim: number;
  metadata: {
    pca_2d_variance?: number[];
    pca_3d_variance?: number[];
  };
  selectedPoint: Point2D | Point3D | null;
  searchQuery?: string;
  highlightedCount?: number;
  categoryField?: string | null;
  categoryFieldOptions?: CategoryFieldOption[];
  textSearchResults?: (Point2D | Point3D)[];
  onTextResultClick?: (point: Point2D | Point3D) => void;
}

export function EmbeddingSidebar({
  state,
  onStateChange,
  embeddingDim,
  metadata,
  selectedPoint,
  searchQuery,
  highlightedCount,
  categoryField,
  categoryFieldOptions,
  textSearchResults,
  onTextResultClick,
  ...props
}: EmbeddingSidebarProps) {
  const { className, ...rest } = props;
  const hasSearch = Boolean(searchQuery && searchQuery.trim().length > 0);
  const showSearchSummary = hasSearch && highlightedCount !== undefined;

  return (
    <Sidebar
      collapsible="offcanvas"
      className={`pt-8 lg:pt-8 ${className ?? ''}`}
      {...rest}
    >
      <SidebarHeader className="border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex size-6 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <span className="text-sm font-bold">E</span>
          </div>
          <span className="font-semibold">Controls</span>
        </div>
      </SidebarHeader>

      <SidebarContent className="gap-0">
        <div className="p-4 space-y-6">
          <VisualizationControls
            state={state}
            onStateChange={onStateChange}
            embeddingDim={embeddingDim}
            metadata={metadata}
            categoryFieldOptions={categoryFieldOptions}
            hasHighlights={Boolean(highlightedCount && highlightedCount > 0)}
          />

          {selectedPoint && (
            <>
              <Separator />
              {/*<div className="pt-3">
                <SelectedPointCard point={selectedPoint} />
              </div>*/}
            </>
          )}

          {showSearchSummary && textSearchResults && textSearchResults.length > 0 && (
            <>
              <Separator />
              <TextSearchResultsList
                results={textSearchResults}
                selectedPointId={selectedPoint?.id}
                onResultClick={onTextResultClick}
                categoryField={categoryField}
                maxHeight={280}
              />
            </>
          )}

        </div>
        <ScrollBar orientation="vertical" />

      </SidebarContent>

      <SidebarFooter className="border-t px-4 py-3">
        <div className="text-xs text-muted-foreground text-center">
          Press{' '}
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium">
            <span className="text-xs">⌘</span>B
          </kbd>{' '}
          to toggle
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
