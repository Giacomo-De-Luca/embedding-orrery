'use client';

import * as React from 'react';
import { ChevronDown, Settings2, Loader2, X } from 'lucide-react';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from '@/lib/ui-primitives/sidebar';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/lib/ui-primitives/select';
import { PromptCombobox } from '@/lib/ui-primitives/prompt-combobox';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import { TextSearchResultsList } from './TextSearchResultsList';
import { TopicSearchSection } from './TopicSearchSection';
import { MetadataFilters } from './MetadataFilters';
import type { Point2D, Point3D, TopicInfo, TextSearchConfig, DistanceMetric } from '../../lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '../../lib/hooks/useTopicSearch';
import type { UseDocumentFeatureSearchReturn, SelectedFeature, DocumentRankingMode } from '../../lib/hooks/useDocumentFeatureSearch';
import { fieldToDisplayName } from '../../lib/utils/fieldAnalysis';
import { MAX_HIGHLIGHTED_FEATURES } from '../../lib/hooks/usePromptHighlight';
import { useVisualizationStore } from '../../lib/stores/useVisualizationStore';

// Must match ChromaDBClient.DOCUMENT_SENTINEL in the backend
const DOCUMENT_SENTINEL = '__document__';

// Max feature-search result rows rendered in the sidebar preview list
const FEATURE_RESULTS_PREVIEW_LIMIT = 50;

interface SearchSidebarProps extends React.ComponentProps<typeof Sidebar> {
  searchQuery: string;
  onSearchChange: (query: string) => void;
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
  // Feature search (document activations) — combobox multi-select
  featureSearch?: UseDocumentFeatureSearchReturn | null;
  onFeatureSearchResultClick?: (rowIndex: number) => void;
}

export function SearchSidebar({
  searchQuery,
  onSearchChange,
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
  // Feature search (document activations)
  featureSearch,
  onFeatureSearchResultClick,
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

  // Feature search combobox state
  const featureChipsRef = useComboboxAnchor();
  const featureMapRef = React.useRef(new Map<string, SelectedFeature>());

  // Keep a map of value keys → feature objects for the combobox
  const featureValueKey = React.useCallback((f: SelectedFeature) => String(f.featureIndex), []);

  const selectedFeatureValues = React.useMemo(
    () => (featureSearch?.selectedFeatures ?? []).map((f) => featureValueKey(f)),
    [featureSearch?.selectedFeatures, featureValueKey],
  );

  // Update the map when suggestions or selected features change
  React.useEffect(() => {
    if (!featureSearch) return;
    for (const f of featureSearch.suggestions) {
      featureMapRef.current.set(featureValueKey(f), f);
    }
    for (const f of featureSearch.selectedFeatures) {
      featureMapRef.current.set(featureValueKey(f), f);
    }
  }, [featureSearch?.suggestions, featureSearch?.selectedFeatures, featureValueKey]);

  const handleFeatureSelectionChange = React.useCallback(
    (newValues: string[]) => {
      if (!featureSearch) return;
      const currentKeys = new Set(selectedFeatureValues);
      const nextKeys = new Set(newValues);

      // Find added
      for (const key of nextKeys) {
        if (!currentKeys.has(key)) {
          const feature = featureMapRef.current.get(key);
          if (feature) featureSearch.addFeature(feature);
        }
      }
      // Find removed
      for (const key of currentKeys) {
        if (!nextKeys.has(key)) {
          featureSearch.removeFeature(Number(key));
        }
      }
    },
    [featureSearch?.addFeature, featureSearch?.removeFeature, selectedFeatureValues],
  );

  const handleFeatureInputChange = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      featureSearch?.searchFeatures(e.target.value);
    },
    [featureSearch?.searchFeatures],
  );

  // Read search config from store
  const textSearchConfig = useVisualizationStore((s) => s.textSearchConfig);
  const setTextSearchConfig = useVisualizationStore((s) => s.setTextSearchConfig);
  const distanceMetric = useVisualizationStore((s) => s.distanceMetric);

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

      <SidebarContent className="gap-0">

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
              delay={600}
            />
            <p className="text-xs text-muted-foreground">
              Search will highlight matching words in the visualization
            </p>
          </div>

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

          {/* Feature Search (Document Activations — combobox multi-select) */}
          {saeInfo && featureSearch && featureSearch.hasActivations === true && (
            <div className="space-y-3">
              <Label className="text-base">Feature Search</Label>
              <Combobox<string, true>
                multiple
                value={selectedFeatureValues}
                onValueChange={handleFeatureSelectionChange}
              >
                <ComboboxChips ref={featureChipsRef} className="min-h-8">
                  {featureSearch.selectedFeatures.map((f) => (
                    <ComboboxChip key={f.featureIndex}>
                      #{f.featureIndex}: {f.label ? (f.label.length > 25 ? f.label.slice(0, 25) + '…' : f.label) : 'unlabeled'}
                    </ComboboxChip>
                  ))}
                  <ComboboxChipsInput
                    placeholder={featureSearch.selectedFeatures.length === 0 ? 'Search features...' : 'Add more...'}
                    className="text-xs"
                    onChange={handleFeatureInputChange}
                  />
                </ComboboxChips>
                <ComboboxContent anchor={featureChipsRef}>
                  <ComboboxList>
                    {featureSearch.suggestions.map((f) => (
                      <ComboboxItem key={f.featureIndex} value={featureValueKey(f)}>
                        <span className="truncate flex-1">
                          #{f.featureIndex}: {f.label ?? 'unlabeled'}
                        </span>
                        {f.density != null && (
                          <span className="text-muted-foreground text-[10px] ml-auto shrink-0">
                            {f.density.toExponential(1)}
                          </span>
                        )}
                      </ComboboxItem>
                    ))}
                  </ComboboxList>
                  <ComboboxEmpty>
                    {featureSearch.suggestionsLoading ? 'Searching...' : 'No matching features'}
                  </ComboboxEmpty>
                </ComboboxContent>
              </Combobox>

              {featureSearch.error && (
                <p className="text-xs text-destructive">{featureSearch.error}</p>
              )}

              {featureSearch.selectedFeatures.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {featureSearch.selectedFeatures.length} feature{featureSearch.selectedFeatures.length !== 1 ? 's' : ''} selected
                      {featureSearch.status !== 'searching' && (
                        <> · {featureSearch.totalResults} doc{featureSearch.totalResults !== 1 ? 's' : ''}</>
                      )}
                      {featureSearch.status === 'searching' && (
                        <> · <Loader2 className="inline h-3 w-3 animate-spin" /></>
                      )}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={featureSearch.clearFeatures}
                    >
                      <X className="h-3 w-3 mr-1" />Clear
                    </Button>
                  </div>

                  {/* Ranking mode — only meaningful with 2+ features */}
                  {featureSearch.selectedFeatures.length >= 2 && (
                    <div className="flex items-center gap-2">
                      <Label htmlFor="feature-ranking" className="text-xs whitespace-nowrap">
                        Ranking
                      </Label>
                      <Select
                        value={featureSearch.rankingMode}
                        onValueChange={(value) =>
                          featureSearch.setRankingMode(value as DocumentRankingMode)
                        }
                      >
                        <SelectTrigger id="feature-ranking" className="h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="SCALED_SUM">Scaled sum</SelectItem>
                          <SelectItem value="MAX">Max activation</SelectItem>
                          <SelectItem value="SUM">Sum</SelectItem>
                          <SelectItem value="MATCHING_FEATURES">Feature count</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {featureSearch.results.length > 0 && (
                    <ScrollArea
                      className="rounded-md border [&>[data-radix-scroll-area-viewport]>div]:block!"
                      viewportClassName="max-h-60"
                    >
                      <div className="p-1 space-y-0.5">
                      {featureSearch.results.slice(0, FEATURE_RESULTS_PREVIEW_LIMIT).map((r, i) => (
                        <button
                          key={r.itemId ?? i}
                          className="w-full text-left px-3 py-2 rounded-md text-sm transition-colors hover:bg-accent hover:text-accent-foreground"
                          onClick={() => {
                            if (r.rowIndex != null) {
                              onFeatureSearchResultClick?.(r.rowIndex);
                            }
                          }}
                        >
                          <span className="font-medium line-clamp-2">
                            {r.document ?? r.itemId}
                          </span>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            score: {r.score.toFixed(3)} · {r.matchingFeatures} feature{r.matchingFeatures !== 1 ? 's' : ''}
                          </p>
                        </button>
                      ))}
                      {featureSearch.totalResults > FEATURE_RESULTS_PREVIEW_LIMIT && (
                        <p className="px-3 py-1.5 text-xs text-muted-foreground">
                          Showing top {Math.min(FEATURE_RESULTS_PREVIEW_LIMIT, featureSearch.results.length)} of {featureSearch.totalResults.toLocaleString()} matches
                        </p>
                      )}
                      </div>
                    </ScrollArea>
                  )}
                </div>
              )}

              {featureSearch.selectedFeatures.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  Search and select SAE features to find matching documents
                </p>
              )}
            </div>
          )}
          {saeInfo && featureSearch && featureSearch.hasActivations === false && (
            <div className="space-y-2">
              <Label className="text-base text-muted-foreground">Feature Search</Label>
              <p className="text-xs text-muted-foreground">
                Document activations not computed. Use Collection Manager to compute them.
              </p>
            </div>
          )}

          {/* Advanced Search Options */}
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="group w-full justify-between px-0 h-8">
                <span className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Settings2 className="h-4 w-4" />
                  Advanced
                </span>
                <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-2">
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

              {/* Distance metric (semantic search) */}
              <div className="space-y-2">
                <Label htmlFor="distance-metric" className="text-sm">Distance Metric</Label>
                <Select
                  value={distanceMetric ?? 'COSINE'}
                  onValueChange={(value) =>
                    useVisualizationStore.getState().setDistanceMetric(value as DistanceMetric)
                  }
                >
                  <SelectTrigger id="distance-metric">
                    <SelectValue placeholder="Select metric" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="COSINE">Cosine Similarity</SelectItem>
                    <SelectItem value="L2">Euclidean (L2)</SelectItem>
                    <SelectItem value="IP">Inner Product</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Used for semantic search similarity calculations
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
