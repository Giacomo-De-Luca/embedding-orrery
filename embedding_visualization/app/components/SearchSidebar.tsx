'use client';

import * as React from 'react';
import { ChevronDown, Settings2 } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Button } from '@/lib/ui-primitives/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { PromptCombobox } from '@/lib/ui-primitives/prompt-combobox';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import { TextSearchResultsList } from './TextSearchResultsList';
import { TopicSearchSection } from './TopicSearchSection';
import type { Point2D, Point3D, TopicInfo } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
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
  // Query prompt configuration
  queryPromptName?: string | null;
  onQueryPromptNameChange?: (value: string | null) => void;
  // Topic search props
  topics?: TopicInfo[];
  topicSearchMode?: TopicSearchMode;
  onTopicSearchModeChange?: (mode: TopicSearchMode) => void;
  topicDirectQuery?: string;
  onTopicDirectQueryChange?: (q: string) => void;
  topicFilteredTopics?: TopicInfo[];
  topicSemanticQuery?: string;
  onTopicSemanticQueryChange?: (q: string) => void;
  onTopicSemanticSearch?: () => void;
  topicSemanticResults?: TopicSearchResult[];
  topicSemanticLoading?: boolean;
  selectedTopicIds?: Set<number>;
  onToggleTopic?: (id: number) => void;
  onSelectAllTopics?: () => void;
  onClearAllTopics?: () => void;
  categoricalPalette?: string;
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
  queryPromptName,
  onQueryPromptNameChange,
  // Topic search props
  topics,
  topicSearchMode,
  onTopicSearchModeChange,
  topicDirectQuery,
  onTopicDirectQueryChange,
  topicFilteredTopics,
  topicSemanticQuery,
  onTopicSemanticQueryChange,
  onTopicSemanticSearch,
  topicSemanticResults,
  topicSemanticLoading,
  selectedTopicIds,
  onToggleTopic,
  onSelectAllTopics,
  onClearAllTopics,
  categoricalPalette,
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

          {/* Advanced Search Options */}
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="w-full justify-between px-0 h-8">
                <span className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Settings2 className="h-4 w-4" />
                  Advanced
                </span>
                <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-2">
              <div className="space-y-2">
                <Label htmlFor="query-prompt-name" className="text-sm">Query Prompt Name</Label>
                <PromptCombobox
                  id="query-prompt-name"
                  value={queryPromptName ?? ''}
                  onChange={(v) => onQueryPromptNameChange?.(v.trim() === '' ? null : v.trim())}
                  placeholder="None (type or select a prompt)"
                  className="h-8"
                />
                <p className="text-xs text-muted-foreground">
                  Task-specific prompt for models like Gemma Embedding. Type a custom prompt or select a preset.
                </p>
              </div>
            </CollapsibleContent>
          </Collapsible>

          {/* Search Results */}
          {showResults && (
            <TextSearchResultsList
              results={textSearchResults}
              selectedPointId={selectedPointId}
              onResultClick={onResultClick}
              categoryField={categoryField}
              searchQuery={searchQuery}
              maxHeight={400}
            />
          )}

          {/* Topic Search */}
          {topics && topics.length > 0 && topicSearchMode !== undefined && onTopicSearchModeChange && (
            <TopicSearchSection
              topics={topics}
              mode={topicSearchMode}
              onModeChange={onTopicSearchModeChange}
              directQuery={topicDirectQuery ?? ''}
              onDirectQueryChange={onTopicDirectQueryChange ?? (() => {})}
              filteredTopics={topicFilteredTopics ?? topics}
              semanticQuery={topicSemanticQuery ?? ''}
              onSemanticQueryChange={onTopicSemanticQueryChange ?? (() => {})}
              onSemanticSearch={onTopicSemanticSearch ?? (() => {})}
              semanticResults={topicSemanticResults ?? []}
              semanticLoading={topicSemanticLoading ?? false}
              selectedTopicIds={selectedTopicIds ?? new Set()}
              onToggleTopic={onToggleTopic ?? (() => {})}
              onSelectAll={onSelectAllTopics ?? (() => {})}
              onClearAll={onClearAllTopics ?? (() => {})}
              categoricalPalette={categoricalPalette}
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
