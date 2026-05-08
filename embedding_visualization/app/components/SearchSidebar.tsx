'use client';

import * as React from 'react';
import { ChevronDown, Settings2, Loader2, X } from 'lucide-react';
import {
  Sidebar,
  SidebarContentPlain,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { Label } from '@/lib/ui-primitives/label';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import {
  Combobox,
  ComboboxChips,
  ComboboxChip,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxEmpty,
  useComboboxAnchor,
} from '@/lib/ui-primitives/combobox';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { PromptCombobox } from '@/lib/ui-primitives/prompt-combobox';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import { TextSearchResultsList } from './TextSearchResultsList';
import { TopicSearchSection } from './TopicSearchSection';
import { MetadataFilters } from './MetadataFilters';
import type { Point2D, Point3D, TopicInfo, TextSearchConfig } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
import { cn } from '@/lib/utils/utils';
import { fieldToDisplayName } from '../../lib/utils/fieldAnalysis';
import { MAX_HIGHLIGHTED_FEATURES } from '../../lib/hooks/usePromptHighlight';
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
  itemMetadata?: Record<string, unknown>[];
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
  // SAE prompt highlight
  saeInfo?: { modelId: string; saeId: string } | null;
  promptHighlightStatus?: 'idle' | 'loading_model' | 'running' | 'error';
  promptHighlightError?: string | null;
  promptHighlightActivePrompt?: string | null;
  onPromptHighlightSubmit?: (prompt: string) => void;
  onPromptHighlightClear?: () => void;
  promptMaxDensity?: number | null;
  onPromptMaxDensityChange?: (value: number | null) => void;
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
  itemMetadata,
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
  // SAE prompt highlight
  saeInfo,
  promptHighlightStatus,
  promptHighlightError,
  promptHighlightActivePrompt,
  onPromptHighlightSubmit,
  onPromptHighlightClear,
  promptMaxDensity,
  onPromptMaxDensityChange,
  className,
  ...props
}: SearchSidebarProps) {
  const hasSearch = Boolean(searchQuery && searchQuery.trim().length > 0);
  const showResults = hasSearch && textSearchResults && textSearchResults.length > 0;

  // SAE prompt activation state
  const [promptText, setPromptText] = React.useState('');
  const promptBusy = promptHighlightStatus === 'loading_model' || promptHighlightStatus === 'running';
  const handlePromptSubmit = React.useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (promptText.trim() && onPromptHighlightSubmit) {
      onPromptHighlightSubmit(promptText.trim());
    }
  }, [promptText, onPromptHighlightSubmit]);
  const handlePromptClear = React.useCallback(() => {
    setPromptText('');
    onPromptHighlightClear?.();
  }, [onPromptHighlightClear]);

  // Read search config from store
  const textSearchConfig = useVisualizationStore((s) => s.textSearchConfig);
  const setTextSearchConfig = useVisualizationStore((s) => s.setTextSearchConfig);

  // Derive selected fields for combobox (null = document only → [DOCUMENT_SENTINEL])
  const selectedFields = textSearchConfig.fields ?? [DOCUMENT_SENTINEL];
  const chipsRef = useComboboxAnchor();

  const handleFieldsChange = React.useCallback((newValue: string[] | undefined) => {
    const fields = newValue && newValue.length > 0 ? newValue : null;
    setTextSearchConfig({ ...textSearchConfig, fields });
  }, [textSearchConfig, setTextSearchConfig]);

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

          {/* SAE Prompt Activation */}
          {saeInfo && onPromptHighlightSubmit && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Label className="text-base">Prompt Activation</Label>
                {promptBusy && (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                )}
              </div>
              <form onSubmit={handlePromptSubmit} className="flex gap-2">
                <Input
                  placeholder="Enter a prompt..."
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  className="flex-1"
                  disabled={promptBusy}
                />
                <Button type="submit" size="sm" disabled={!promptText.trim() || promptBusy}>
                  Run
                </Button>
              </form>
              {promptHighlightStatus === 'loading_model' && (
                <p className="text-xs text-muted-foreground">Loading model...</p>
              )}
              {promptHighlightStatus === 'running' && (
                <p className="text-xs text-muted-foreground">Running SAE inference...</p>
              )}
              {promptHighlightError && (
                <p className="text-xs text-destructive">{promptHighlightError}</p>
              )}
              {promptHighlightActivePrompt && !promptBusy && (
                <div className="flex items-center gap-2">
                  <p className="text-xs text-muted-foreground truncate flex-1">
                    Top {MAX_HIGHLIGHTED_FEATURES} features for: &ldquo;{promptHighlightActivePrompt}&rdquo;
                  </p>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 shrink-0"
                    onClick={handlePromptClear}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              )}
              {!promptHighlightActivePrompt && !promptBusy && (
                <p className="text-xs text-muted-foreground">
                  Run a prompt through the SAE to highlight the top activated features
                </p>
              )}
              {/* Max density filter */}
              <div className="flex items-center gap-2">
                <Label htmlFor="max-density" className="text-xs whitespace-nowrap">
                  Max density
                </Label>
                <Input
                  id="max-density"
                  type="number"
                  step="any"
                  placeholder="e.g. 0.01"
                  value={promptMaxDensity ?? ''}
                  onChange={(e) => {
                    const val = e.target.value;
                    onPromptMaxDensityChange?.(val === '' ? null : parseFloat(val));
                  }}
                  className="h-7 w-28 text-xs"
                />
              </div>
            </div>
          )}

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
              {/* Search fields selector (combobox with chips) */}
              <div className="space-y-2">
                <Label className="text-sm">Search in</Label>
                <Combobox<string, true>
                  multiple
                  value={selectedFields}
                  onValueChange={handleFieldsChange}
                >
                  <ComboboxChips ref={chipsRef} className="min-h-8">
                    {selectedFields.map((field) => (
                      <ComboboxChip key={field}>
                        {field === DOCUMENT_SENTINEL ? 'Document' : fieldToDisplayName(field)}
                      </ComboboxChip>
                    ))}
                    <ComboboxChipsInput placeholder="Add fields..." className="text-xs" />
                  </ComboboxChips>
                  <ComboboxContent anchor={chipsRef}>
                    <ComboboxList>
                      <ComboboxItem value={DOCUMENT_SENTINEL}>Document</ComboboxItem>
                      {availableFields.map((field) => (
                        <ComboboxItem key={field} value={field}>
                          {fieldToDisplayName(field)}
                        </ComboboxItem>
                      ))}
                    </ComboboxList>
                    <ComboboxEmpty>No matching fields</ComboboxEmpty>
                  </ComboboxContent>
                </Combobox>
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

              {/* Metadata filters */}
              {availableFields.length > 0 && (
                <div className="space-y-2">
                  <Label className="text-sm">Filters</Label>
                  <MetadataFilters
                    filters={textSearchConfig.filters}
                    onChange={(filters) => setTextSearchConfig({ ...textSearchConfig, filters })}
                    availableFields={availableFields}
                    itemMetadata={itemMetadata ?? []}
                  />
                </div>
              )}

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
