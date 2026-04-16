'use client';

import * as React from 'react';
import { ChevronDown, Settings2, Loader2 } from 'lucide-react';
import {
  Sidebar,
  SidebarContentPlain,
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
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
} from '@/lib/ui-primitives/dropdown-menu';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { PromptCombobox } from '@/lib/ui-primitives/prompt-combobox';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import { TextSearchResultsList } from './TextSearchResultsList';
import { TopicSearchSection } from './TopicSearchSection';
import type { Point2D, Point3D, TopicInfo, TextSearchConfig } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
import { cn } from '@/lib/utils/utils';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';

// Must match ChromaDBClient.DOCUMENT_SENTINEL in the backend
const DOCUMENT_SENTINEL = '__document__';

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
  // Text search config
  textSearchLoading?: boolean;
  availableFields?: string[];
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
  textSearchLoading,
  availableFields = [],
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

  // Read search config from store
  const textSearchConfig = useVisualizationStore((s) => s.textSearchConfig);
  const setTextSearchConfig = useVisualizationStore((s) => s.setTextSearchConfig);

  // Derive selected fields set (null = document only)
  const selectedFields = textSearchConfig.fields;
  const isAllFields = selectedFields !== null &&
    selectedFields.includes(DOCUMENT_SENTINEL) &&
    availableFields.every(f => selectedFields.includes(f));

  const toggleField = React.useCallback((field: string) => {
    const current = selectedFields ?? [DOCUMENT_SENTINEL];
    const next = current.includes(field)
      ? current.filter(f => f !== field)
      : [...current, field];
    // If nothing selected, revert to document only (null)
    setTextSearchConfig({
      ...textSearchConfig,
      fields: next.length === 0 ? null : next,
    });
  }, [selectedFields, textSearchConfig, setTextSearchConfig]);

  const toggleAllFields = React.useCallback(() => {
    if (isAllFields) {
      // Reset to document only
      setTextSearchConfig({ ...textSearchConfig, fields: null });
    } else {
      setTextSearchConfig({
        ...textSearchConfig,
        fields: [DOCUMENT_SENTINEL, ...availableFields],
      });
    }
  }, [isAllFields, availableFields, textSearchConfig, setTextSearchConfig]);

  const fieldSummary = React.useMemo(() => {
    if (selectedFields === null) return 'Document';
    const count = selectedFields.length;
    if (isAllFields) return 'All fields';
    if (count === 1) {
      return selectedFields[0] === DOCUMENT_SENTINEL ? 'Document' : selectedFields[0];
    }
    return `${count} fields`;
  }, [selectedFields, isAllFields]);

  return (
    <Sidebar
      collapsible="offcanvas"
      className={className}
      {...props}
    >
      <SidebarHeader className="border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold">Search</span>
        </div>
      </SidebarHeader>

      <SidebarContentPlain className="gap-0">

        <div className="p-4 space-y-6 ">
          {/* Search Input */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Label htmlFor="sidebar-search" className="text-base">Search</Label>
              {textSearchLoading && (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
              )}
            </div>
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
              {/* Search fields selector */}
              <div className="space-y-2">
                <Label className="text-sm">Search in</Label>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full justify-between h-8 text-xs">
                      {fieldSummary}
                      <ChevronDown className="h-3 w-3 ml-1 opacity-50" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56 max-h-64 overflow-y-auto">
                    <DropdownMenuCheckboxItem
                      checked={isAllFields}
                      onCheckedChange={toggleAllFields}
                    >
                      All fields
                    </DropdownMenuCheckboxItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuCheckboxItem
                      checked={(selectedFields ?? [DOCUMENT_SENTINEL]).includes(DOCUMENT_SENTINEL)}
                      onCheckedChange={() => toggleField(DOCUMENT_SENTINEL)}
                    >
                      Document
                    </DropdownMenuCheckboxItem>
                    {availableFields.map((field) => (
                      <DropdownMenuCheckboxItem
                        key={field}
                        checked={(selectedFields ?? []).includes(field)}
                        onCheckedChange={() => toggleField(field)}
                      >
                        {field}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              {/* Match mode toggle */}
              <div className="space-y-2">
                <Label className="text-sm">Match mode</Label>
                <ToggleGroup
                  type="single"
                  value={textSearchConfig.mode}
                  onValueChange={(value) => {
                    if (value) setTextSearchConfig({ ...textSearchConfig, mode: value as 'CONTAINS' | 'EXACT' });
                  }}
                  className="justify-start"
                >
                  <ToggleGroupItem value="CONTAINS" className="text-xs h-7 px-3">
                    Contains
                  </ToggleGroupItem>
                  <ToggleGroupItem value="EXACT" className="text-xs h-7 px-3">
                    Exact
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>

              {/* Case sensitivity toggle */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="case-sensitive"
                  checked={textSearchConfig.caseSensitive}
                  onCheckedChange={(checked) =>
                    setTextSearchConfig({ ...textSearchConfig, caseSensitive: checked === true })
                  }
                />
                <Label htmlFor="case-sensitive" className="text-sm font-normal cursor-pointer">
                  Case sensitive
                </Label>
              </div>

              {/* Query prompt name */}
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

      </SidebarContentPlain>

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
