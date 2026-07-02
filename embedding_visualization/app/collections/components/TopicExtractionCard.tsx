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
import { X, ChevronDown, ChevronRight, Pencil, Check, RotateCw } from 'lucide-react';
import { ProgressModal } from './EmbeddingProgressModal';
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
import type { TopicConfigInput, ExtractTopicsResult, ReduceTopicsInput, ReduceTopicsResult, GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';
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
  // Standalone reduction
  reduceTopics: (input: ReduceTopicsInput) => Promise<ReduceTopicsResult | null>;
  reduceTopicsLoading: boolean;
  lastReduceResult: ReduceTopicsResult | null;
  // LLM label generation
  generateLlmLabels: (input: GenerateLlmLabelsInput) => Promise<GenerateLlmLabelsResult | null>;
  llmLabelsLoading: boolean;
  lastLlmLabelsResult: GenerateLlmLabelsResult | null;
  hasSubtopics: boolean;
  // Topic label renaming
  renameTopicLabel: (collectionName: string, topicId: number, newLabel: string, isSubtopic?: boolean) => Promise<{ error?: string | null } | null>;
  regenerateTopicLabel: (collectionName: string, topicId: number, llmConfig?: string) => Promise<{ error?: string | null; newLabel?: string } | null>;
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
  reduceTopics,
  reduceTopicsLoading,
  lastReduceResult,
  generateLlmLabels,
  llmLabelsLoading,
  lastLlmLabelsResult,
  hasSubtopics,
  renameTopicLabel,
  regenerateTopicLabel,
}: TopicExtractionCardProps) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<TopicConfigState>(DEFAULT_TOPIC_CONFIG);
  const [editingTopicId, setEditingTopicId] = useState<number | null>(null);
  const [editingLabel, setEditingLabel] = useState('');
  const [renamingSaving, setRenamingSaving] = useState(false);
  const [regeneratingTopicId, setRegeneratingTopicId] = useState<number | null>(null);
  const [showAllTopics, setShowAllTopics] = useState(false);

  // Standalone reduction state
  const [reduceMethod, setReduceMethod] = useState<string>('fixed_n');
  const [reduceNTopics, setReduceNTopics] = useState<number>(10);
  const [reduceUseCtfidf, setReduceUseCtfidf] = useState<boolean>(true);
  const [reduceRegenerateLabels, setReduceRegenerateLabels] = useState<boolean>(false);
  const [reduceLlmProvider, setReduceLlmProvider] = useState<string>('gemini');
  const [reduceLlmModel, setReduceLlmModel] = useState<string>('gemini-3-flash-preview');

  // LLM label generation state
  const [llmLabelScope, setLlmLabelScope] = useState<string>(hasSubtopics ? 'both' : 'topics_only');
  const [llmLabelProvider, setLlmLabelProvider] = useState<string>('gemini');
  const [llmLabelModel, setLlmLabelModel] = useState<string>('gemini-3-flash-preview');
  const [llmLabelResume, setLlmLabelResume] = useState<boolean>(true);

  const handleExtract = async () => {
    const result = await extractTopics(collectionName, toTopicConfigInput(config));
    if (result && !result.error) {
      onTopicsExtracted();
    }
  };

  const handleGenerateLlmLabels = async () => {
    const result = await generateLlmLabels({
      collectionName,
      llmProvider: llmLabelProvider,
      llmModel: llmLabelModel,
      labelScope: llmLabelScope,
      resume: llmLabelResume,
    });
    if (result && !result.error) {
      onTopicsExtracted();
    }
  };

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

            {/* Extraction Results Display */}
            {lastTopicsResult && !lastTopicsResult.error && (
              <div className="space-y-3">
                <Separator />
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">{lastTopicsResult.numTopics} topics</Badge>
                  <Badge variant="outline">{lastTopicsResult.numNoisePoints} unclustered</Badge>
                  {!!lastTopicsResult.durationSeconds && (
                    <Badge variant="outline">{lastTopicsResult.durationSeconds.toFixed(1)}s</Badge>
                  )}
                  {lastTopicsResult.reductionApplied && lastTopicsResult.numTopicsBeforeReduction != null && (
                    <Badge variant="outline">reduced from {lastTopicsResult.numTopicsBeforeReduction}</Badge>
                  )}
                </div>

                <div className="space-y-2 max-h-[400px] overflow-y-auto">
                  {(showAllTopics ? lastTopicsResult.topics : lastTopicsResult.topics.slice(0, 5)).map((topic) => (
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
                                  className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-muted"
                                  onClick={() => {
                                    setEditingTopicId(topic.topicId);
                                    setEditingLabel(topic.label || `Topic ${topic.topicId}`);
                                  }}
                                  title="Rename topic"
                                >
                                  <Pencil className="h-3 w-3 text-muted-foreground" />
                                </button>
                                <button
                                  className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-muted"
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
                  {!showAllTopics && lastTopicsResult.topics.length > 5 && (
                    <button
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                      onClick={() => setShowAllTopics(true)}
                    >
                      +{lastTopicsResult.topics.length - 5} more topics
                    </button>
                  )}
                  {showAllTopics && lastTopicsResult.topics.length > 5 && (
                    <button
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                      onClick={() => setShowAllTopics(false)}
                    >
                      Show less
                    </button>
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

            {/* Standalone Topic Reduction */}
            {hasTopics && (
              <>
                <Separator />
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

                {/* Generate LLM Labels */}
                <Separator />
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
                      disabled={llmLabelsLoading || topicsLoading || reduceTopicsLoading}
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
              </>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
      {/* Topic Extraction Progress Modal */}
      {topicsLoading && (
        <ProgressModal
          jobId={collectionName}
          title="Extracting Topics"
          subtitle="Clustering points and extracting keywords..."
          itemsLabel="topics"
        />
      )}
      {/* Topic Reduction Progress Modal */}
      {reduceTopicsLoading && (
        <ProgressModal
          jobId={`${collectionName}_reduce`}
          title="Reducing Topics"
          subtitle="Merging similar topics..."
          itemsLabel="topics"
        />
      )}
      {/* LLM Labeling Progress Modal */}
      {llmLabelsLoading && (
        <ProgressModal
          jobId={`${collectionName}_llm_labeling`}
          title="Generating LLM Labels"
          subtitle="Each topic is labeled individually via LLM API calls."
          itemsLabel="topics"
        />
      )}
    </Collapsible>
  );
}
