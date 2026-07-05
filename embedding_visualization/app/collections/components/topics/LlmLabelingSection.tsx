'use client';

import { useState } from 'react';
import { Button } from '@/lib/ui-primitives/button';
import { Badge } from '@/lib/ui-primitives/badge';
import { Separator } from '@/lib/ui-primitives/separator';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Label } from '@/lib/ui-primitives/label';
import { Input } from '@/lib/ui-primitives/input';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import type { GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';

interface LlmLabelingSectionProps {
  collectionName: string;
  hasSubtopics: boolean;
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  /** Other topic operations running — labeling is disabled meanwhile */
  otherOpsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;
  onLabeled: () => void;
}

/** "Generate LLM Labels" controls + result summary. Owns its own form state. */
export function LlmLabelingSection({
  collectionName,
  hasSubtopics,
  generateLlmLabels,
  llmLabelsLoading,
  otherOpsLoading,
  lastLlmLabelsResult,
  onLabeled,
}: LlmLabelingSectionProps) {
  const [llmLabelScope, setLlmLabelScope] = useState<string>(hasSubtopics ? 'both' : 'topics_only');
  const [llmLabelProvider, setLlmLabelProvider] = useState<string>('gemini');
  const [llmLabelModel, setLlmLabelModel] = useState<string>('gemini-3-flash-preview');
  const [llmLabelResume, setLlmLabelResume] = useState<boolean>(true);

  const handleGenerateLlmLabels = async () => {
    const result = await generateLlmLabels({
      collectionName,
      llmProvider: llmLabelProvider,
      llmModel: llmLabelModel,
      labelScope: llmLabelScope,
      resume: llmLabelResume,
    });
    if (result && !result.error) {
      onLabeled();
    }
  };

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium">Generate LLM Labels</h4>
      <p className="text-xs text-muted-foreground">
        Add human-readable labels to existing topics using an LLM.
      </p>

      <div className="space-y-3">
        <div className="space-y-2">
          <Label>Label Scope</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            value={llmLabelScope}
            onValueChange={(v) => { if (v) setLlmLabelScope(v); }}
          >
            <ToggleGroupItem value="both" className="text-xs" disabled={!hasSubtopics}>Both</ToggleGroupItem>
            <ToggleGroupItem value="topics_only" className="text-xs">Topics Only</ToggleGroupItem>
            <ToggleGroupItem value="subtopics_only" className="text-xs" disabled={!hasSubtopics}>Subtopics Only</ToggleGroupItem>
          </ToggleGroup>
        </div>

        <div className="space-y-2">
          <Label htmlFor="llm-label-provider">Provider</Label>
          <Select value={llmLabelProvider} onValueChange={setLlmLabelProvider}>
            <SelectTrigger id="llm-label-provider">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="openai">OpenAI</SelectItem>
              <SelectItem value="gemini">Gemini</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="llm-label-model">Model</Label>
          <Input
            id="llm-label-model"
            value={llmLabelModel}
            onChange={(e) => setLlmLabelModel(e.target.value)}
            placeholder="gemini-3-flash-preview"
          />
        </div>

        <div className="flex items-center gap-2">
          <Checkbox
            id="llm-label-resume"
            checked={llmLabelResume}
            onCheckedChange={(checked) => setLlmLabelResume(checked === true)}
          />
          <Label htmlFor="llm-label-resume" className="cursor-pointer">
            Skip already-labeled topics
          </Label>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={handleGenerateLlmLabels}
          disabled={llmLabelsLoading || otherOpsLoading}
        >
          {llmLabelsLoading ? (
            <>
              <Spinner className="h-4 w-4 mr-2" />
              Generating Labels...
            </>
          ) : (
            'Generate LLM Labels'
          )}
        </Button>
      </div>

      {/* LLM Labels Results */}
      {lastLlmLabelsResult && !lastLlmLabelsResult.error && (
        <div className="space-y-3">
          <Separator />
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">
              {lastLlmLabelsResult.topicsLabeled}/{lastLlmLabelsResult.totalTopics} topics labeled
            </Badge>
            {lastLlmLabelsResult.totalSubtopics > 0 && (
              <Badge variant="secondary">
                {lastLlmLabelsResult.subtopicsLabeled}/{lastLlmLabelsResult.totalSubtopics} subtopics labeled
              </Badge>
            )}
            <Badge variant="outline">{lastLlmLabelsResult.durationSeconds.toFixed(1)}s</Badge>
          </div>
        </div>
      )}
    </div>
  );
}
