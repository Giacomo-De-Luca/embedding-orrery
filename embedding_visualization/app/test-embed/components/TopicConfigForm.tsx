'use client';

import { Label } from '@/lib/ui-primitives/label';
import { Separator } from '@/lib/ui-primitives/separator';
import { Slider } from '@/lib/ui-primitives/slider';
import { Checkbox } from '@/lib/ui-primitives/checkbox';
import { Input } from '@/lib/ui-primitives/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/lib/ui-primitives/select';
import { ToggleGroup, ToggleGroupItem } from '@/lib/ui-primitives/toggle-group';
import type { TopicConfigInput } from '@/lib/graphql/mutations';

export interface TopicConfigState {
  minTopicSize: number;
  nKeywords: number;
  projectionType: string;
  useLlmLabels: boolean;
  llmProvider: string;
  llmModel: string;
}

export const DEFAULT_TOPIC_CONFIG: TopicConfigState = {
  minTopicSize: 10,
  nKeywords: 10,
  projectionType: 'umap_2d',
  useLlmLabels: false,
  llmProvider: 'gemini',
  llmModel: 'gemini-3-flash-preview',
};

/** Convert TopicConfigState to the GraphQL input shape. */
export function toTopicConfigInput(state: TopicConfigState): TopicConfigInput {
  const config: TopicConfigInput = {
    minTopicSize: state.minTopicSize,
    nKeywords: state.nKeywords,
    projectionType: state.projectionType,
    useLlmLabels: state.useLlmLabels,
  };
  if (state.useLlmLabels) {
    config.llmProvider = state.llmProvider;
    config.llmModel = state.llmModel;
  }
  return config;
}

interface TopicConfigFormProps {
  value: TopicConfigState;
  onChange: (value: TopicConfigState) => void;
}

export function TopicConfigForm({ value, onChange }: TopicConfigFormProps) {
  const update = (patch: Partial<TopicConfigState>) => {
    onChange({ ...value, ...patch });
  };

  return (
    <div className="space-y-4">
      {/* Clustering Config */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="min-topic-size">Min Topic Size</Label>
          <span className="text-sm text-muted-foreground">{value.minTopicSize}</span>
        </div>
        <Slider
          id="min-topic-size"
          min={5}
          max={50}
          step={1}
          value={[value.minTopicSize]}
          onValueChange={([v]) => update({ minTopicSize: v })}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="n-keywords">Keywords per Topic</Label>
          <span className="text-sm text-muted-foreground">{value.nKeywords}</span>
        </div>
        <Slider
          id="n-keywords"
          min={5}
          max={20}
          step={1}
          value={[value.nKeywords]}
          onValueChange={([v]) => update({ nKeywords: v })}
        />
      </div>

      <div className="space-y-2">
        <Label>Projection Type</Label>
        <ToggleGroup
          type="single"
          variant="outline"
          value={value.projectionType}
          onValueChange={(v) => { if (v) update({ projectionType: v }); }}
        >
          <ToggleGroupItem value="umap_2d" className="text-xs">UMAP 2D</ToggleGroupItem>
          <ToggleGroupItem value="umap_3d" className="text-xs">UMAP 3D</ToggleGroupItem>
          <ToggleGroupItem value="pca_2d" className="text-xs">PCA 2D</ToggleGroupItem>
          <ToggleGroupItem value="pca_3d" className="text-xs">PCA 3D</ToggleGroupItem>
        </ToggleGroup>
      </div>

      <Separator />

      {/* LLM Labeling */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Checkbox
            id="use-llm"
            checked={value.useLlmLabels}
            onCheckedChange={(checked) => update({ useLlmLabels: checked === true })}
          />
          <Label htmlFor="use-llm" className="cursor-pointer">Generate LLM labels</Label>
        </div>

        {value.useLlmLabels && (
          <div className="space-y-3 pl-6">
            <div className="space-y-2">
              <Label htmlFor="llm-provider">Provider</Label>
              <Select value={value.llmProvider} onValueChange={(v) => update({ llmProvider: v })}>
                <SelectTrigger id="llm-provider">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="gemini">Gemini</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="llm-model">Model</Label>
              <Input
                id="llm-model"
                value={value.llmModel}
                onChange={(e) => update({ llmModel: e.target.value })}
                placeholder="gpt-4o-mini"
              />
            </div>

            <p className="text-xs text-muted-foreground">
              Requires {value.llmProvider === 'openai' ? 'CHROMA_OPENAI_API_KEY' : 'GEMINI_API_KEY'} environment variable on the backend.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
