'use client';

import * as React from 'react';
import { Search } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import { TextSearchResultsList } from './TextSearchResultsList';
import type { Point2D, Point3D } from '../../lib/types/types';
import { ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { cn } from '@/lib/utils/utils';

interface SearchSidebarProps extends React.ComponentProps<typeof Sidebar> {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  showOnlyHighlighted: boolean;
  onShowOnlyHighlightedChange: (checked: boolean) => void;
  showLabels: boolean;
  onShowLabelsChange: (checked: boolean) => void;
  hasHighlights: boolean;
  textSearchResults?: (Point2D | Point3D)[];
  selectedPointId?: string | null;
  onResultClick?: (point: Point2D | Point3D) => void;
  categoryField?: string | null;
}

export function SearchSidebar({
  searchQuery,
  onSearchChange,
  showOnlyHighlighted,
  onShowOnlyHighlightedChange,
  showLabels,
  onShowLabelsChange,
  hasHighlights,
  textSearchResults,
  selectedPointId,
  onResultClick,
  categoryField,
  className,
  ...props
}: SearchSidebarProps) {
  const hasSearch = Boolean(searchQuery && searchQuery.trim().length > 0);
  const showResults = hasSearch && textSearchResults && textSearchResults.length > 0;

  return (
    <Sidebar
      collapsible="offcanvas"
      className={className}
      {...props}
    >
      <SidebarHeader className="border-b px-4 py-3">
        <div className="flex items-center gap-2">
          {/*<div className="flex size-6 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Search className="size-3.5" />
          </div>*/}
          <span className="font-semibold">Search</span>
        </div>
      </SidebarHeader>

      <SidebarContent className="gap-0">
        <div className="p-4 space-y-6">
          {/* Search Input */}
          <div className="space-y-3">
            <Label htmlFor="sidebar-search" className="text-base">Search</Label>
            <DebouncedSearchInput
              id="sidebar-search"
              className="max-w-5/6"
              placeholder="Type to search..."
              value={searchQuery}
              onChange={onSearchChange}
              delay={300}
            />
            <p className="text-xs text-muted-foreground">
              Search will highlight matching words in the visualization
            </p>
          </div>

          {/* Show Only Highlighted */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="show-only-highlighted"
              checked={showOnlyHighlighted}
              onCheckedChange={(checked) => onShowOnlyHighlightedChange(checked === true)}
              disabled={!hasHighlights}
            />
            <Label
              htmlFor="show-only-highlighted"
              className={cn(
                "font-normal cursor-pointer",
                !hasHighlights && "text-muted-foreground"
              )}
            >
              Show only highlighted
            </Label>
          </div>

          {/* Show Labels */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="show-labels"
              checked={showLabels}
              onCheckedChange={(checked) => onShowLabelsChange(checked === true)}
              disabled={!hasHighlights}
            />
            <Label
              htmlFor="show-labels"
              className={cn(
                "font-normal cursor-pointer",
                !hasHighlights && "text-muted-foreground"
              )}
            >
              Show labels
            </Label>
          </div>

          {/* Search Results */}
          {showResults && (
            <TextSearchResultsList
              results={textSearchResults}
              selectedPointId={selectedPointId}
              onResultClick={onResultClick}
              categoryField={categoryField}
              maxHeight={400}
            />
          )}
        </div>
        <ScrollBar orientation="vertical" />
      </SidebarContent>

      <SidebarFooter className="border-t px-4 py-3">
        <div className="text-xs text-muted-foreground text-center">
          Press{' '}
          <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium">
            <span className="text-xs">⌘</span>K
          </kbd>{' '}
          to toggle
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
