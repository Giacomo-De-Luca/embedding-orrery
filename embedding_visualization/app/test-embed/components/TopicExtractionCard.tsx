'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/lib/ui-primitives/collapsible';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { X, ChevronDown, ChevronRight } from 'lucide-react';
import type { TopicConfigInput, ExtractTopicsResult } from '@/lib/graphql/mutations';
import { TopicConfigForm, DEFAULT_TOPIC_CONFIG, toTopicConfigInput, type TopicConfigState } from './TopicConfigForm';

interface TopicExtractionCardProps {
  collectionName: string;
  hasTopics: boolean;
  topicCount: number | null;
  extractTopics: (collectionName: string, config?: TopicConfigInput) => Promise<ExtractTopicsResult | null>;
  topicsLoading: boolean;
  lastTopicsResult: ExtractTopicsResult | null;
  error: string | null;
  clearError: () => void;
  onTopicsExtracted: () => void;
}

export function TopicExtractionCard({
  collectionName,
  hasTopics,
  topicCount,
  extractTopics,
  topicsLoading,
  lastTopicsResult,
  error,
  clearError,
  onTopicsExtracted,
}: TopicExtractionCardProps) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<TopicConfigState>(DEFAULT_TOPIC_CONFIG);

  const handleExtract = async () => {
    const result = await extractTopics(collectionName, toTopicConfigInput(config));
    if (result && !result.error) {
      onTopicsExtracted();
    }
  };

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card>
        <CardHeader className="pb-3">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="p-0 h-auto hover:bg-transparent justify-start">
              <div className="flex items-center gap-2">
                {open ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
                <CardTitle>Topic Extraction</CardTitle>
                {hasTopics && topicCount != null && (
                  <Badge variant="secondary" className="text-xs">{topicCount} topics</Badge>
                )}
              </div>
            </Button>
          </CollapsibleTrigger>
          <CardDescription>
            Cluster points and extract topic keywords
          </CardDescription>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="space-y-5 pt-0">
            <TopicConfigForm value={config} onChange={setConfig} />

            <Separator />

            {/* Extract Button */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleExtract}
              disabled={topicsLoading}
            >
              {topicsLoading ? (
                <>
                  <Spinner className="h-4 w-4 mr-2" />
                  Extracting Topics...
                </>
              ) : hasTopics ? (
                'Re-extract Topics'
              ) : (
                'Extract Topics'
              )}
            </Button>

            {/* Error Display */}
            {error && (
              <div className="flex items-start gap-2 p-3 border border-destructive rounded-md bg-destructive/5">
                <p className="text-sm text-destructive flex-1">{error}</p>
                <button onClick={clearError} className="text-destructive hover:text-destructive/80">
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            {/* Results Display */}
            {lastTopicsResult && !lastTopicsResult.error && (
              <div className="space-y-3">
                <Separator />
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">{lastTopicsResult.numTopics} topics</Badge>
                  <Badge variant="outline">{lastTopicsResult.numNoisePoints} unclustered</Badge>
                  <Badge variant="outline">{lastTopicsResult.durationSeconds.toFixed(1)}s</Badge>
                </div>

                <div className="space-y-2">
                  {lastTopicsResult.topics.slice(0, 5).map((topic) => (
                    <div key={topic.topicId} className="text-sm border rounded-md p-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">
                          {topic.label || `Topic ${topic.topicId}`}
                        </span>
                        <Badge variant="secondary" className="text-xs">
                          {topic.count} pts
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {topic.keywords.slice(0, 5).map(k => k.word).join(', ')}
                      </p>
                    </div>
                  ))}
                  {lastTopicsResult.topics.length > 5 && (
                    <p className="text-xs text-muted-foreground">
                      +{lastTopicsResult.topics.length - 5} more topics
                    </p>
                  )}
                </div>

                <a
                  href={`/?collection=${encodeURIComponent(collectionName)}&colorBy=topic_label`}
                  className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-4"
                >
                  View in Visualization
                </a>
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
