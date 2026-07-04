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

/** Stored visualization projections that can double as the clustering space. */
const PROJECTION_SPACES = ['umap_3d', 'umap_2d', 'pca_3d', 'pca_2d'] as const;

/** Projection loaded for ids/documents (and used as fallback) when clustering on raw vectors. */
const RAW_MODE_PROJECTION = 'umap_3d';

export interface TopicConfigState {
  clusteringMethod: string;      // "hdbscan" | "kmeans" | "gmm" | "spectral"
  nClusters: number;
  minTopicSize: number;
  nKeywords: number;
  // Single space selector: a stored projection ("umap_3d" | "umap_2d" | "pca_3d" | "pca_2d")
  // or a raw-vector mode ("cluster_umap" | "embedding"). Split back into the GraphQL
  // projectionType + clusterOn pair in toTopicConfigInput.
  clusteringSpace: string;
  clusterNComponents: number;    // BERTopic UMAP dims (cluster_umap only)
  clusterMinDist: number;        // BERTopic UMAP min_dist (cluster_umap only)
  clusterNNeighbors: number;     // BERTopic UMAP n_neighbors (cluster_umap only)
  useLlmLabels: boolean;
  llmProvider: string;
  llmModel: string;
  // Reduction
  enableReduction: boolean;
  reductionMethod: string;       // "auto" or "fixed_n"
  reductionNTopics: number;
  reductionUseCtfidf: boolean;
}

export const DEFAULT_TOPIC_CONFIG: TopicConfigState = {
  clusteringMethod: 'hdbscan',
  nClusters: 10,
  minTopicSize: 10,
  nKeywords: 10,
  clusteringSpace: 'umap_3d',
  clusterNComponents: 5,
  clusterMinDist: 0.0,
  clusterNNeighbors: 15,
  useLlmLabels: false,
  llmProvider: 'gemini',
  llmModel: 'gemini-3-flash-preview',
  enableReduction: false,
  reductionMethod: 'fixed_n',
  reductionNTopics: 10,
  reductionUseCtfidf: true,
};

/** Convert TopicConfigState to the GraphQL input shape. */
export function toTopicConfigInput(state: TopicConfigState): TopicConfigInput {
  const isProjectionSpace = (PROJECTION_SPACES as readonly string[]).includes(state.clusteringSpace);
  const config: TopicConfigInput = {
    minTopicSize: state.minTopicSize,
    nKeywords: state.nKeywords,
    projectionType: isProjectionSpace ? state.clusteringSpace : RAW_MODE_PROJECTION,
    useLlmLabels: state.useLlmLabels,
    clusteringMethod: state.clusteringMethod,
    clusterOn: isProjectionSpace ? 'projection' : state.clusteringSpace,
  };
  if (state.clusteringMethod !== 'hdbscan') {
    config.nClusters = state.nClusters;
  }
  if (state.clusteringSpace === 'cluster_umap') {
    config.clusterNComponents = state.clusterNComponents;
    config.clusterMinDist = state.clusterMinDist;
    config.clusterNNeighbors = state.clusterNNeighbors;
  }
  if (state.useLlmLabels) {
    config.llmProvider = state.llmProvider;
    config.llmModel = state.llmModel;
  }
  if (state.enableReduction) {
    config.reduction = {
      enabled: true,
      method: state.reductionMethod,
      nTopics: state.reductionMethod === 'fixed_n' ? state.reductionNTopics : undefined,
      useCtfidf: state.reductionUseCtfidf,
    };
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
          <div className="space-y-3 pl-6">
            <div className="space-y-2">
              <Label>Reduction Method</Label>
              <ToggleGroup
                type="single"
                variant="outline"
                value={value.reductionMethod}
                onValueChange={(v) => { if (v) update({ reductionMethod: v }); }}
              >
                <ToggleGroupItem value="fixed_n" className="text-xs">Fixed N</ToggleGroupItem>
                <ToggleGroupItem value="auto" className="text-xs">Auto</ToggleGroupItem>
              </ToggleGroup>
            </div>

            {value.reductionMethod === 'fixed_n' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="reduction-n-topics">Target Topics</Label>
                  <span className="text-sm text-muted-foreground">{value.reductionNTopics}</span>
                </div>
                <Slider
                  id="reduction-n-topics"
                  min={2}
                  max={50}
                  step={1}
                  value={[value.reductionNTopics]}
                  onValueChange={([v]) => update({ reductionNTopics: v })}
                />
              </div>
            )}

            <div className="space-y-2">
              <Label>Similarity Method</Label>
              <ToggleGroup
                type="single"
                variant="outline"
                value={value.reductionUseCtfidf ? 'ctfidf' : 'semantic'}
                onValueChange={(v) => {
                  if (v) update({ reductionUseCtfidf: v === 'ctfidf' });
                }}
              >
                <ToggleGroupItem value="ctfidf" className="text-xs">c-TF-IDF</ToggleGroupItem>
                <ToggleGroupItem value="semantic" className="text-xs">Semantic</ToggleGroupItem>
              </ToggleGroup>
              <p className="text-xs text-muted-foreground">
                c-TF-IDF is fast. Semantic uses embeddings for better quality but is slower.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
