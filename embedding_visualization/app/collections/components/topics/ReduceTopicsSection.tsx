'use client';

import { useState } from 'react';
import { useQuery } from '@apollo/client/react';
import { Button } from '@/lib/ui-primitives/button';
import { dotControlIndentClassName } from '@/lib/ui-primitives/dot-control';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { TOPIC_MERGE_TREE } from '@/lib/graphql/queries';
import type { ReduceTopicsInput, ReduceTopicsResult, TopicMergeTreeData } from '@/lib/graphql/mutations';
import { DEFAULT_REDUCTION_CHOICE, ReductionInputs, type ReductionChoice } from '../../lib/topicConfigInput';
import { ReductionStrategyFields } from './ReductionStrategyFields';

/** Slider maximum while the collection's base topic count is unknown. */
const FALLBACK_MAX_TOPICS = 50;

interface TopicTreeSummaryData {
  topicMergeTree: Pick<TopicMergeTreeData, 'availableLinkages' | 'leaves'> | null;
}

interface ReduceTopicsSectionProps {
  collectionName: string;
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  /** Extraction running — reduce is disabled meanwhile */
  topicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  onReduced: () => void;
}

/**
 * "Reduce Topics" controls + result summary. Owns its own form state. A
 * reduction always cuts the extraction's base topics (its original clusters),
 * so the "Topics" slider goes up to their count (from the merge-tree query)
 * and a second reduction never compounds the first.
 */
export function ReduceTopicsSection({
  collectionName,
  reduceTopics,
  reduceTopicsLoading,
  topicsLoading,
  lastReduceResult,
  onReduced,
}: ReduceTopicsSectionProps) {
  const [choice, setChoice] = useState<ReductionChoice>(DEFAULT_REDUCTION_CHOICE);
  const [reduceRegenerateLabels, setReduceRegenerateLabels] = useState<boolean>(false);
  const [reduceLlmProvider, setReduceLlmProvider] = useState<string>('gemini');
  const [reduceLlmModel, setReduceLlmModel] = useState<string>('gemini-3-flash-preview');

  // Base topics and the linkages this collection offers (density needs an
  // HDBSCAN extraction made with density trees).
  const { data: treeData, refetch: refetchTree } = useQuery<TopicTreeSummaryData>(TOPIC_MERGE_TREE, {
    variables: { collectionName, linkage: 'average', includeNoise: false },
    fetchPolicy: 'network-only',
  });
  const tree = treeData?.topicMergeTree ?? null;
  const densityAvailable = tree?.availableLinkages.includes('density') ?? false;
  const maxTopics = tree?.leaves.length ?? FALLBACK_MAX_TOPICS;

  const handleReduce = async () => {
    const input = ReductionInputs.reduceTopics(collectionName, choice, {
      regenerateLabels: reduceRegenerateLabels,
      llmProvider: reduceLlmProvider,
      llmModel: reduceLlmModel,
    }, densityAvailable);
    const result = await reduceTopics(input);
    if (result && !result.error) {
      onReduced();
      void refetchTree();
    }
  };

  // The hook's last result is shared by every collection's pane: only show ours.
  const result = lastReduceResult?.collectionName === collectionName && !lastReduceResult.error
    ? lastReduceResult
    : null;

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium">Reduce Topics</h4>
      <p className="text-xs text-muted-foreground">
        Merge similar topics. Reductions start from the original clusters, which stay available as subtopics.
      </p>

      <div className="space-y-3">
        <ReductionStrategyFields
          value={choice}
          onChange={setChoice}
          densityAvailable={densityAvailable}
          maxTopics={maxTopics}
        />

        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Checkbox
              id="reduce-regenerate-labels"
              checked={reduceRegenerateLabels}
              onCheckedChange={(checked) => setReduceRegenerateLabels(checked === true)}
            />
            <Label htmlFor="reduce-regenerate-labels" className="cursor-pointer">
              LLM labels for merged topics
            </Label>
          </div>

          {reduceRegenerateLabels && (
            <div className={`space-y-3 ${dotControlIndentClassName}`}>
              <div className="space-y-2">
                <Label htmlFor="reduce-llm-provider">Provider</Label>
                <Select value={reduceLlmProvider} onValueChange={setReduceLlmProvider}>
                  <SelectTrigger id="reduce-llm-provider">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="gemini">Gemini</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="reduce-llm-model">Model</Label>
                <Input
                  id="reduce-llm-model"
                  value={reduceLlmModel}
                  onChange={(e) => setReduceLlmModel(e.target.value)}
                  placeholder="gemini-3-flash-preview"
                />
              </div>
            </div>
          )}
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={handleReduce}
          disabled={reduceTopicsLoading || topicsLoading}
        >
          {reduceTopicsLoading ? (
            <>
              <Spinner className="h-4 w-4 mr-2" />
              Reducing Topics...
            </>
          ) : (
            'Reduce Topics'
          )}
        </Button>
      </div>

      {/* Reduction Results */}
      {result && (
        <div className="space-y-3">
          <Separator />
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              {result.numTopicsBefore} → {result.numTopicsAfter} topics
            </Badge>
            {result.absorbedNoise > 0 && (
              <Badge variant="secondary">{result.absorbedNoise.toLocaleString()} noise documents absorbed</Badge>
            )}
            <Badge variant="outline">{result.durationSeconds.toFixed(1)}s</Badge>
          </div>

          <div className="space-y-2">
            {result.topics.slice(0, 5).map((topic) => (
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
            {result.topics.length > 5 && (
              <p className="text-xs text-muted-foreground">
                +{result.topics.length - 5} more topics
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
