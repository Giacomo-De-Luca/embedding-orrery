'use client';

import { Label } from '@/lib/ui-primitives/label';
import { dotControlIndentClassName } from '@/lib/ui-primitives/dot-control';
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
import {
  PROJECTION_SPACES,
  ReductionRules,
  type TopicConfigState,
} from '../lib/topicConfigInput';
import { ReductionStrategyFields } from './topics/ReductionStrategyFields';

// The state and its GraphQL builder live in ../lib/topicConfigInput (node-testable);
// re-exported so existing imports from this module keep working.
export { DEFAULT_TOPIC_CONFIG, toTopicConfigInput, type TopicConfigState } from '../lib/topicConfigInput';

/** Most topics the inline reduction offers (the base topics are not known before clustering). */
const INLINE_MAX_TOPICS = 50;

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
      {/* Clustering Method */}
      <div className="space-y-2">
        <Label>Clustering Method</Label>
        <ToggleGroup
          type="single"
          variant="outline"
          value={value.clusteringMethod}
          onValueChange={(v) => { if (v) update({ clusteringMethod: v }); }}
        >
          <ToggleGroupItem value="hdbscan" className="text-xs">HDBSCAN</ToggleGroupItem>
          <ToggleGroupItem value="kmeans" className="text-xs">KMeans</ToggleGroupItem>
          <ToggleGroupItem value="gmm" className="text-xs">GMM</ToggleGroupItem>
          <ToggleGroupItem value="spectral" className="text-xs">Spectral</ToggleGroupItem>
        </ToggleGroup>
        {value.clusteringMethod === 'spectral' && (
          <p className="text-xs text-muted-foreground">
            Spectral clustering uses O(n<sup>2</sup>) memory. Best for datasets under 20k points.
          </p>
        )}
      </div>

      {/* HDBSCAN: min topic size */}
      {value.clusteringMethod === 'hdbscan' && (
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
      )}

      {/* KMeans/GMM/Spectral: number of clusters */}
      {value.clusteringMethod !== 'hdbscan' && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="n-clusters">Number of Clusters</Label>
            <span className="text-sm text-muted-foreground">{value.nClusters}</span>
          </div>
          <Slider
            id="n-clusters"
            min={2}
            max={100}
            step={1}
            value={[value.nClusters]}
            onValueChange={([v]) => update({ nClusters: v })}
          />
        </div>
      )}

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

      {/* Clustering Space (merged: which space HDBSCAN & co. run in) */}
      <div className="space-y-2">
        <Label>Clustering Space</Label>
        <ToggleGroup
          type="single"
          variant="outline"
          className="flex-wrap justify-start"
          value={value.clusteringSpace}
          onValueChange={(v) => { if (v) update({ clusteringSpace: v }); }}
        >
          <ToggleGroupItem value="umap_3d" className="text-xs">UMAP 3D</ToggleGroupItem>
          <ToggleGroupItem value="umap_2d" className="text-xs">UMAP 2D</ToggleGroupItem>
          <ToggleGroupItem value="pca_3d" className="text-xs">PCA 3D</ToggleGroupItem>
          <ToggleGroupItem value="pca_2d" className="text-xs">PCA 2D</ToggleGroupItem>
          <ToggleGroupItem value="cluster_umap" className="text-xs">BERTopic 5D</ToggleGroupItem>
          <ToggleGroupItem value="embedding" className="text-xs">Raw embeddings</ToggleGroupItem>
        </ToggleGroup>
        {(PROJECTION_SPACES as readonly string[]).includes(value.clusteringSpace) && (
          <p className="text-xs text-muted-foreground">
            Clusters directly on the stored projection, so topics match the shapes you see in that
            view. The projection must already be computed for this collection.
          </p>
        )}
        {value.clusteringSpace === 'cluster_umap' && (
          <p className="text-xs text-muted-foreground">
            Runs a fresh UMAP (min_dist 0) on the raw vectors before clustering — slower than
            visualization coords, but usually sharper topics.
          </p>
        )}
        {value.clusteringSpace === 'embedding' && (
          <p className="text-xs text-muted-foreground">
            Clusters on the full-dimensional vectors. Advanced; quality varies with embedding dimension.
          </p>
        )}
      </div>

      {/* BERTopic UMAP params */}
      {value.clusteringSpace === 'cluster_umap' && (
        <div className="space-y-4 pl-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="cluster-n-components">Dimensions</Label>
              <span className="text-sm text-muted-foreground">{value.clusterNComponents}</span>
            </div>
            <Slider
              id="cluster-n-components"
              min={2}
              max={15}
              step={1}
              value={[value.clusterNComponents]}
              onValueChange={([v]) => update({ clusterNComponents: v })}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="cluster-min-dist">Min Distance</Label>
              <span className="text-sm text-muted-foreground">{value.clusterMinDist.toFixed(2)}</span>
            </div>
            <Slider
              id="cluster-min-dist"
              min={0}
              max={0.5}
              step={0.05}
              value={[value.clusterMinDist]}
              onValueChange={([v]) => update({ clusterMinDist: v })}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="cluster-n-neighbors">N Neighbors</Label>
              <span className="text-sm text-muted-foreground">{value.clusterNNeighbors}</span>
            </div>
            <Slider
              id="cluster-n-neighbors"
              min={5}
              max={50}
              step={1}
              value={[value.clusterNNeighbors]}
              onValueChange={([v]) => update({ clusterNNeighbors: v })}
            />
          </div>
        </div>
      )}

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
          <div className={`space-y-3 ${dotControlIndentClassName}`}>
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

      <Separator />

      {/* Topic Reduction */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Checkbox
            id="enable-reduction"
            checked={value.enableReduction}
            onCheckedChange={(checked) => update({ enableReduction: checked === true })}
          />
          <Label htmlFor="enable-reduction" className="cursor-pointer">
            Reduce topics after extraction
          </Label>
        </div>

        {value.enableReduction && (
          <div className={dotControlIndentClassName}>
            <ReductionStrategyFields
              value={value.reduction}
              onChange={(reduction) => update({ reduction })}
              densityAvailable={ReductionRules.densityAvailableFor(value.clusteringMethod)}
              maxTopics={INLINE_MAX_TOPICS}
            />
          </div>
        )}
      </div>
    </div>
  );
}
