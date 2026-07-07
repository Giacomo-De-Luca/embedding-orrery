'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button, buttonVariants } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Input } from '@/lib/ui-primitives/input';
import { ScrollArea } from '@/lib/ui-primitives/scroll-area';
import { X, Pencil, Check, RotateCw } from 'lucide-react';
import type { ExtractTopicsResult } from '@/lib/graphql/mutations';

interface TopicListSectionProps {
  collectionName: string;
  result: ExtractTopicsResult;
  renameTopicLabel: (collectionName: string, topicId: number, newLabel: string, isSubtopic?: boolean) => Promise<{ error?: string | null } | null>;
  regenerateTopicLabel: (collectionName: string, topicId: number, llmConfig?: string) => Promise<{ error?: string | null; newLabel?: string } | null>;
}

/**
 * Extraction results: summary badges + the topic list with inline
 * rename/regenerate actions and a link into the visualization.
 */
export function TopicListSection({
  collectionName,
  result,
  renameTopicLabel,
  regenerateTopicLabel,
}: TopicListSectionProps) {
  const [editingTopicId, setEditingTopicId] = useState<number | null>(null);
  const [editingLabel, setEditingLabel] = useState('');
  const [renamingSaving, setRenamingSaving] = useState(false);
  const [regeneratingTopicId, setRegeneratingTopicId] = useState<number | null>(null);
  const [showAllTopics, setShowAllTopics] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary">{result.numTopics} topics</Badge>
        <Badge variant="outline">{result.numNoisePoints} unclustered</Badge>
        {!!result.durationSeconds && (
          <Badge variant="outline">{result.durationSeconds.toFixed(1)}s</Badge>
        )}
        {result.reductionApplied && result.numTopicsBeforeReduction != null && (
          <Badge variant="outline">reduced from {result.numTopicsBeforeReduction}</Badge>
        )}
      </div>

      <ScrollArea className="[&>[data-radix-scroll-area-viewport]>div]:block!" viewportClassName="max-h-[400px]">
      <div className="space-y-2">
        {(showAllTopics ? result.topics : result.topics.slice(0, 5)).map((topic) => (
          <div key={topic.topicId} className="text-sm border rounded-md p-2 group">
            <div className="flex items-center justify-between mb-1">
              {editingTopicId === topic.topicId ? (
                <form
                  className="flex items-center gap-1 flex-1 mr-2"
                  onSubmit={async (e) => {
                    e.preventDefault();
                    if (!editingLabel.trim()) return;
                    setRenamingSaving(true);
                    await renameTopicLabel(collectionName, topic.topicId, editingLabel.trim());
                    setRenamingSaving(false);
                    setEditingTopicId(null);
                  }}
                >
                  <Input
                    value={editingLabel}
                    onChange={(e) => setEditingLabel(e.target.value)}
                    className="h-6 text-sm py-0 px-1"
                    autoFocus
                    disabled={renamingSaving}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') setEditingTopicId(null);
                    }}
                  />
                  <Button
                    type="submit"
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5"
                    disabled={renamingSaving || !editingLabel.trim()}
                  >
                    <Check className="h-3 w-3" />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5"
                    onClick={() => setEditingTopicId(null)}
                    disabled={renamingSaving}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </form>
              ) : (
                <span className="font-medium flex items-center gap-1">
                  {regeneratingTopicId === topic.topicId ? (
                    <RotateCw className="h-3 w-3 animate-spin text-muted-foreground" />
                  ) : null}
                  {topic.label || `Topic ${topic.topicId}`}
                  {topic.topicId !== -1 && regeneratingTopicId !== topic.topicId && (
                    <>
                      <button
                        className="opacity-40 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-muted"
                        onClick={() => {
                          setEditingTopicId(topic.topicId);
                          setEditingLabel(topic.label || `Topic ${topic.topicId}`);
                        }}
                        title="Rename topic"
                      >
                        <Pencil className="h-3 w-3 text-muted-foreground" />
                      </button>
                      <button
                        className="opacity-40 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-muted"
                        onClick={async () => {
                          setRegeneratingTopicId(topic.topicId);
                          await regenerateTopicLabel(collectionName, topic.topicId);
                          setRegeneratingTopicId(null);
                        }}
                        title="Regenerate LLM label"
                      >
                        <RotateCw className="h-3 w-3 text-muted-foreground" />
                      </button>
                    </>
                  )}
                </span>
              )}
              <Badge variant="secondary" className="text-xs shrink-0">
                {topic.count} pts
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              {topic.keywords.slice(0, 5).map(k => k.word).join(', ')}
            </p>
          </div>
        ))}
        {!showAllTopics && result.topics.length > 5 && (
          <button
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowAllTopics(true)}
          >
            +{result.topics.length - 5} more topics
          </button>
        )}
        {showAllTopics && result.topics.length > 5 && (
          <button
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowAllTopics(false)}
          >
            Show less
          </button>
        )}
      </div>
      </ScrollArea>

      <Link
        href={`/?collection=${encodeURIComponent(collectionName)}&colorBy=topic_label`}
        className={buttonVariants({ variant: 'outline', size: 'sm' })}
      >
        View in Visualization
      </Link>
    </div>
  );
}
