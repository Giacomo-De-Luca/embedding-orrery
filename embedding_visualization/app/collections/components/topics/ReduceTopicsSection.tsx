'use client';

import { useState } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Slider } from '@/lib/ui-primitives/slider';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { ReduceTopicsInput, ReduceTopicsResult } from '@/lib/graphql/mutations';

interface ReduceTopicsSectionProps {
  collectionName: string;
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  /** Extraction running — reduce is disabled meanwhile */
  topicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  onReduced: () => void;
}

/** "Reduce Topics" controls + result summary. Owns its own form state. */
export function ReduceTopicsSection({
  collectionName,
  reduceTopics,
  reduceTopicsLoading,
  topicsLoading,
  lastReduceResult,
  onReduced,
}: ReduceTopicsSectionProps) {
  const [reduceMethod, setReduceMethod] = useState<string>('fixed_n');
  const [reduceNTopics, setReduceNTopics] = useState<number>(10);
  const [reduceUseCtfidf, setReduceUseCtfidf] = useState<boolean>(true);
  const [reduceRegenerateLabels, setReduceRegenerateLabels] = useState<boolean>(false);
  const [reduceLlmProvider, setReduceLlmProvider] = useState<string>('gemini');
  const [reduceLlmModel, setReduceLlmModel] = useState<string>('gemini-3-flash-preview');

  const handleReduce = async () => {
    const result = await reduceTopics({
      collectionName,
      method: reduceMethod,
      nTopics: reduceMethod === 'fixed_n' ? reduceNTopics : undefined,
      useCtfidf: reduceUseCtfidf,
      regenerateLabels: reduceRegenerateLabels,
      llmProvider: reduceLlmProvider,
      llmModel: reduceLlmModel,
    });
    if (result && !result.error) {
      onReduced();
    }
  };

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium">Reduce Topics</h4>
      <p className="text-xs text-muted-foreground">
        Merge similar topics to reduce the total count.
      </p>

      <div className="space-y-3">
        <div className="space-y-2">
          <Label>Method</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            value={reduceMethod}
            onValueChange={(v) => { if (v) setReduceMethod(v); }}
          >
            <ToggleGroupItem value="fixed_n" className="text-xs">Fixed N</ToggleGroupItem>
            <ToggleGroupItem value="auto" className="text-xs">Auto</ToggleGroupItem>
          </ToggleGroup>
        </div>

        {reduceMethod === 'fixed_n' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="reduce-n-topics">Target Topics</Label>
              <span className="text-sm text-muted-foreground">{reduceNTopics}</span>
            </div>
            <Slider
              id="reduce-n-topics"
              min={2}
              max={50}
              step={1}
              value={[reduceNTopics]}
              onValueChange={([v]) => setReduceNTopics(v)}
            />
          </div>
        )}

        <div className="space-y-2">
          <Label>Similarity Method</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            value={reduceUseCtfidf ? 'ctfidf' : 'semantic'}
            onValueChange={(v) => {
              if (v) setReduceUseCtfidf(v === 'ctfidf');
            }}
          >
            <ToggleGroupItem value="ctfidf" className="text-xs">c-TF-IDF</ToggleGroupItem>
            <ToggleGroupItem value="semantic" className="text-xs">Semantic</ToggleGroupItem>
          </ToggleGroup>
          <p className="text-xs text-muted-foreground">
            c-TF-IDF is fast. Semantic uses embeddings for better quality but is slower.
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Checkbox
              id="reduce-regenerate-labels"
              checked={reduceRegenerateLabels}
              onCheckedChange={(checked) => setReduceRegenerateLabels(checked === true)}
            />
            <Label htmlFor="reduce-regenerate-labels" className="cursor-pointer">
              Regenerate LLM labels after merging
            </Label>
          </div>

          {reduceRegenerateLabels && (
            <div className="space-y-3 pl-6">
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
      {lastReduceResult && !lastReduceResult.error && (
        <div className="space-y-3">
          <Separator />
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              {lastReduceResult.numTopicsBefore} → {lastReduceResult.numTopicsAfter} topics
            </Badge>
            <Badge variant="outline">{lastReduceResult.durationSeconds.toFixed(1)}s</Badge>
          </div>

          <div className="space-y-2">
            {lastReduceResult.topics.slice(0, 5).map((topic) => (
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
            {lastReduceResult.topics.length > 5 && (
              <p className="text-xs text-muted-foreground">
                +{lastReduceResult.topics.length - 5} more topics
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
