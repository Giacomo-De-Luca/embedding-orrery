'use client';

import React, { useCallback } from 'react';
import { Search, CheckSquare, XSquare, ChevronDown } from 'lucide-react';
import { Button } from '@/lib/ui-primitives/button';
import { Input } from '@/lib/ui-primitives/input';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { cn } from '@/lib/utils/utils';
import { ScrollArea, ScrollBar } from '@/lib/ui-primitives/scroll-area';
import { buildCategoryColorMap } from '@/lib/utils/categoryColors';
import { DebouncedSearchInput } from './DebouncedSearchInput';
import type { TopicInfo } from '@/lib/types/types';
import type { TopicSearchMode, TopicSearchResult } from '@/lib/hooks/useTopicSearch';

interface TopicSearchSectionProps {
  topics: TopicInfo[];
  mode: TopicSearchMode;
  onModeChange: (mode: TopicSearchMode) => void;
  // Direct search
  directQuery: string;
  onDirectQueryChange: (q: string) => void;
  filteredTopics: TopicInfo[];
  // Semantic search
  semanticQuery: string;
  onSemanticQueryChange: (q: string) => void;
  onSemanticSearch: () => void;
  semanticResults: TopicSearchResult[];
  semanticLoading: boolean;
  // Selection
  selectedTopicIds: Set<number>;
  onToggleTopic: (topicId: number) => void;
  onSelectAll: () => void;
  onClearAll: () => void;
  // Optional: palette for color dots
  categoricalPalette?: string;
}

export function TopicSearchSection({
  topics,
  mode,
  onModeChange,
  directQuery,
  onDirectQueryChange,
  filteredTopics,
  semanticQuery,
  onSemanticQueryChange,
  onSemanticSearch,
  semanticResults,
  semanticLoading,
  selectedTopicIds,
  onToggleTopic,
  onSelectAll,
  onClearAll,
  categoricalPalette,
}: TopicSearchSectionProps) {
  // Build color map for topic labels
  const topicColorMap = React.useMemo(() => {
    const values = topics
      .filter(t => t.topicId !== -1)
      .map(t => String(t.topicId));
    return buildCategoryColorMap('topic_id', ['-1', ...values], categoricalPalette);
  }, [topics, categoricalPalette]);

  const handleSemanticKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      onSemanticSearch();
    }
  }, [onSemanticSearch]);

  return (
    <div className="space-y-3">
      <Separator />
      <Label className="text-base">Topics</Label>

      {/* Mode Toggle */}
      <div className="flex gap-1">
        <Button
          variant={mode === 'direct' ? 'default' : 'outline'}
          size="sm"
          className="flex-1 h-7 text-xs"
          onClick={() => onModeChange('direct')}
        >
          Browse
        </Button>
        <Button
          variant={mode === 'semantic' ? 'default' : 'outline'}
          size="sm"
          className="flex-1 h-7 text-xs"
          onClick={() => onModeChange('semantic')}
        >
          Semantic
        </Button>
      </div>

      {/* Direct Mode */}
      {mode === 'direct' && (
        <div className="space-y-2">
          <DebouncedSearchInput
            value={directQuery}
            onChange={onDirectQueryChange}
            placeholder="Filter topics..."
            delay={200}
            className="h-8"
          />
          <TopicList
            topics={filteredTopics}
            selectedTopicIds={selectedTopicIds}
            onToggleTopic={onToggleTopic}
            topicColorMap={topicColorMap}
          />
        </div>
      )}

      {/* Semantic Mode */}
      {mode === 'semantic' && (
        <div className="space-y-2">
          <div className="flex gap-1">
            <Input
              value={semanticQuery}
              onChange={(e) => onSemanticQueryChange(e.target.value)}
              onKeyDown={handleSemanticKeyDown}
              placeholder="Describe a topic..."
              className="h-8 text-sm"
            />
            <Button
              size="sm"
              className="h-8 px-2"
              onClick={onSemanticSearch}
              disabled={semanticLoading || !semanticQuery.trim()}
            >
              <Search className="h-3.5 w-3.5" />
            </Button>
          </div>
          {semanticLoading && (
            <p className="text-xs text-muted-foreground">Searching...</p>
          )}
          {semanticResults.length > 0 && (
            <TopicResultsList
              results={semanticResults}
              selectedTopicIds={selectedTopicIds}
              onToggleTopic={onToggleTopic}
              topicColorMap={topicColorMap}
            />
          )}
        </div>
      )}

      {/* Select All / Clear All */}
      {selectedTopicIds.size > 0 && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {selectedTopicIds.size} topic{selectedTopicIds.size !== 1 ? 's' : ''} selected
          </span>
          <div className="flex gap-1">
            <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={onSelectAll}>
              <CheckSquare className="h-3 w-3 mr-1" />All
            </Button>
            <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={onClearAll}>
              <XSquare className="h-3 w-3 mr-1" />Clear
            </Button>
          </div>
        </div>
      )}
      {selectedTopicIds.size === 0 && topics.length > 0 && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {topics.length} topic{topics.length !== 1 ? 's' : ''}
          </span>
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={onSelectAll}>
            <CheckSquare className="h-3 w-3 mr-1" />Select All
          </Button>
        </div>
      )}
    </div>
  );
}

// ============ Sub-components ============

function TopicList({
  topics,
  selectedTopicIds,
  onToggleTopic,
  topicColorMap,
}: {
  topics: TopicInfo[];
  selectedTopicIds: Set<number>;
  onToggleTopic: (id: number) => void;
  topicColorMap: Record<string, string>;
}) {
  return (
    <div className="max-h-64 overflow-y-auto space-y-0.5 pr-1">
      {topics.map(topic => (
        <TopicRow
          key={topic.topicId}
          topic={topic}
          selected={selectedTopicIds.has(topic.topicId)}
          onToggle={() => onToggleTopic(topic.topicId)}
          color={topicColorMap[String(topic.topicId)] ?? '#888'}
        />
      ))}
      {topics.length === 0 && (
        <p className="text-xs text-muted-foreground py-2">No matching topics</p>
      )}
    </div>
  );
}

function TopicRow({
  topic,
  selected,
  onToggle,
  color,
  relevanceBar,
}: {
  topic: TopicInfo;
  selected: boolean;
  onToggle: () => void;
  color: string;
  relevanceBar?: number; // 0-1 for semantic results
}) {
  const hasSubtopics = topic.subtopics && topic.subtopics.length > 0;
  const hasKeywords = topic.keywords.length > 0;

  return (
    <Collapsible>
      <div className={cn(
        "flex items-center gap-2 px-2 py-1.5 rounded-sm hover:bg-accent/50 cursor-pointer transition-colors",
        selected && "bg-accent"
      )}>
        <Checkbox
          checked={selected}
          onCheckedChange={() => onToggle()}
          className="h-3.5 w-3.5"
        />
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: color }}
        />
        <div className="flex-1 min-w-0" onClick={onToggle}>
          <div className="flex items-center justify-between gap-1">
            <ScrollArea className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
              <span className="text-xs font-medium whitespace-nowrap" onClick={onToggle}>
                {topic.label || `Topic ${topic.topicId}`}
              </span>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
            <span className="text-[10px] text-muted-foreground flex-shrink-0">
              {topic.count}
            </span>
          </div>
          {relevanceBar !== undefined && (
            <div className="mt-0.5 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${Math.round(relevanceBar * 100)}%` }}
              />
            </div>
          )}
        </div>
        {(hasKeywords || hasSubtopics) && (
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="h-5 w-5 p-0">
              <ChevronDown className="h-3 w-3 text-muted-foreground transition-transform duration-200 group-data-[state=open]:rotate-180" />
            </Button>
          </CollapsibleTrigger>
        )}
      </div>
      <CollapsibleContent>
        <div className="pl-10 pr-2 pb-1 space-y-1">
          {hasKeywords && (
            <div className="flex flex-wrap gap-1">
              {topic.keywords.slice(0, 8).map(k => (
                <span key={k.word} className="text-[10px] px-1.5 py-0.5 bg-muted rounded-sm text-muted-foreground">
                  {k.word}
                </span>
              ))}
            </div>
          )}
          {hasSubtopics && (
            <div className="space-y-0.5">
              <span className="text-[10px] text-muted-foreground font-medium">Subtopics:</span>
              {topic.subtopics!.map((st, i) => (
                <div key={i} className="text-[10px] text-muted-foreground pl-2">
                  {st}
                </div>
              ))}
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function TopicResultsList({
  results,
  selectedTopicIds,
  onToggleTopic,
  topicColorMap,
}: {
  results: TopicSearchResult[];
  selectedTopicIds: Set<number>;
  onToggleTopic: (id: number) => void;
  topicColorMap: Record<string, string>;
}) {
  return (
    <div className="max-h-64 overflow-y-auto space-y-0.5 pr-1">
      {results.map(r => (
        <TopicRow
          key={r.topic.topicId}
          topic={r.topic}
          selected={selectedTopicIds.has(r.topic.topicId)}
          onToggle={() => onToggleTopic(r.topic.topicId)}
          color={topicColorMap[String(r.topic.topicId)] ?? '#888'}
          relevanceBar={r.relevance}
        />
      ))}
    </div>
  );
}
