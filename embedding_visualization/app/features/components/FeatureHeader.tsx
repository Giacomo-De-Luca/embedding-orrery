'use client';

import { useState, useCallback, useEffect, type KeyboardEvent } from 'react';
import { Input } from '@/lib/ui-primitives/input';
import { Button } from '@/lib/ui-primitives/button';
import { Search, ArrowLeft, ArrowRight } from 'lucide-react';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import { SaeSelector } from './SaeSelector';
import type { SaeSelectors } from '../hooks/useSaeSelectors';

interface FeatureHeaderProps {
  // Selector state (from useSaeSelectors)
  selectors: SaeSelectors;
  modelOptions: string[];
  layerOptions: string[];
  hookTypeOptions: string[];
  widthOptions: string[];
  onModelChange: (v: string | null) => void;
  onLayerChange: (v: string | null) => void;
  onHookTypeChange: (v: string | null) => void;
  onWidthChange: (v: string | null) => void;
  resolvedCount: number;
  isSingleSae: boolean;

  // Feature navigation
  featureIndex: number | null;
  onFeatureIndexChange: (index: number) => void;
  maxFeatureIndex?: number;

  // Search
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSearch: () => void;
  searchMode?: 'text' | 'semantic' | 'prompt';
  onSearchModeChange?: (mode: 'text' | 'semantic' | 'prompt') => void;
  hasSemanticSearch?: boolean;
  hasPromptSearch?: boolean;

  // Links
  collectionLink?: string | null;
}

/**
 * Header bar with cascading SAE selectors, feature index input, and search.
 */
export function FeatureHeader({
  selectors,
  modelOptions,
  layerOptions,
  hookTypeOptions,
  widthOptions,
  onModelChange,
  onLayerChange,
  onHookTypeChange,
  onWidthChange,
  resolvedCount,
  isSingleSae,
  featureIndex,
  onFeatureIndexChange,
  maxFeatureIndex,
  searchQuery,
  onSearchQueryChange,
  onSearch,
  searchMode = 'text',
  onSearchModeChange,
  hasSemanticSearch = false,
  hasPromptSearch = false,
  collectionLink,
}: FeatureHeaderProps) {
  const [indexInput, setIndexInput] = useState(featureIndex?.toString() ?? '');

  const handleGoToFeature = useCallback(() => {
    const idx = parseInt(indexInput, 10);
    if (!isNaN(idx) && idx >= 0) {
      onFeatureIndexChange(idx);
    }
  }, [indexInput, onFeatureIndexChange]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleGoToFeature();
  }, [handleGoToFeature]);

  const handleSearchKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') onSearch();
  }, [onSearch]);

  const handlePrev = useCallback(() => {
    if (featureIndex != null && featureIndex > 0) {
      const newIdx = featureIndex - 1;
      onFeatureIndexChange(newIdx);
      setIndexInput(newIdx.toString());
    }
  }, [featureIndex, onFeatureIndexChange]);

  const handleNext = useCallback(() => {
    if (featureIndex != null && (maxFeatureIndex == null || featureIndex < maxFeatureIndex - 1)) {
      const newIdx = featureIndex + 1;
      onFeatureIndexChange(newIdx);
      setIndexInput(newIdx.toString());
    }
  }, [featureIndex, maxFeatureIndex, onFeatureIndexChange]);

  // Sync local input when parent changes feature index (e.g. clicking a search result)
  useEffect(() => {
    if (featureIndex != null) {
      setIndexInput(featureIndex.toString());
    }
  }, [featureIndex]);

  const indexNavDisabled = !isSingleSae;

  return (
    <div className="space-y-2">
      {/* Row 1: SAE selectors */}
      <SaeSelector
        selectors={selectors}
        modelOptions={modelOptions}
        layerOptions={layerOptions}
        hookTypeOptions={hookTypeOptions}
        widthOptions={widthOptions}
        onModelChange={onModelChange}
        onLayerChange={onLayerChange}
        onHookTypeChange={onHookTypeChange}
        onWidthChange={onWidthChange}
        resolvedCount={resolvedCount}
      />

      {/* Row 2: Feature navigation + search */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Feature index navigation */}
        <div className="flex items-center gap-1" title={indexNavDisabled ? 'Select a single SAE to browse by index' : undefined}>
          <Button
            variant="outline" size="icon" className="h-8 w-8"
            onClick={handlePrev}
            disabled={indexNavDisabled || featureIndex == null || featureIndex <= 0}
          >
            <ArrowLeft className="h-3.5 w-3.5" />
          </Button>
          <Input
            type="number"
            min={0}
            max={maxFeatureIndex ? maxFeatureIndex - 1 : undefined}
            value={indexNavDisabled ? '' : indexInput}
            onChange={(e) => setIndexInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleGoToFeature}
            className="w-24 h-8 text-center font-mono text-sm"
            placeholder={indexNavDisabled ? 'Multi' : 'Index'}
            disabled={indexNavDisabled}
          />
          <Button
            variant="outline" size="icon" className="h-8 w-8"
            onClick={handleNext}
            disabled={indexNavDisabled || featureIndex == null || (maxFeatureIndex != null && featureIndex >= maxFeatureIndex - 1)}
          >
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Search mode toggle + search input */}
        {(hasSemanticSearch || hasPromptSearch) && onSearchModeChange && (
          <ToggleGroup
            type="single"
            value={searchMode}
            onValueChange={(v) => v && onSearchModeChange(v as 'text' | 'semantic' | 'prompt')}
            variant="outline"
            className="shrink-0"
          >
            <ToggleGroupItem value="text" className="text-xs h-8 px-2">Text</ToggleGroupItem>
            {hasSemanticSearch && (
              <ToggleGroupItem value="semantic" className="text-xs h-8 px-2">Semantic</ToggleGroupItem>
            )}
            {hasPromptSearch && (
              <ToggleGroupItem value="prompt" className="text-xs h-8 px-2">Prompt</ToggleGroupItem>
            )}
          </ToggleGroup>
        )}
        <div className="flex items-center gap-1 flex-1 min-w-48">
          <Input
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            placeholder={
              searchMode === 'prompt'
                ? 'Run a prompt to find activated features...'
                : searchMode === 'semantic'
                  ? 'Search by meaning...'
                  : 'Search features by label...'
            }
            className="h-8 text-sm"
          />
          <Button variant="outline" size="icon" className="h-8 w-8 shrink-0" onClick={onSearch}>
            <Search className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* Link back to visualization */}
        {collectionLink && (
          <a
            href={`/?collection=${encodeURIComponent(collectionLink)}`}
            className="text-xs text-blue-500 hover:underline shrink-0"
          >
            View in scatter plot
          </a>
        )}
      </div>
    </div>
  );
}
