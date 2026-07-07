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
import { ProgressModal } from './EmbeddingProgressModal';
import type { TopicConfigInput, ExtractTopicsResult, ReduceTopicsInput, ReduceTopicsResult, GenerateLlmLabelsInput, GenerateLlmLabelsResult } from '@/lib/graphql/mutations';
import { TopicConfigForm, DEFAULT_TOPIC_CONFIG, toTopicConfigInput, type TopicConfigState } from './TopicConfigForm';
import { TopicListSection } from './topics/TopicListSection';
import { ReduceTopicsSection } from './topics/ReduceTopicsSection';
import { LlmLabelingSection } from './topics/LlmLabelingSection';
import { TopicQualitySection } from './topics/TopicQualitySection';

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

/**
 * Topic-extraction shell: config + extract action, with the results list,
 * reduction, and LLM-labeling flows delegated to the section components under
 * ./topics/. Progress modals for the three long-running operations live here.
 */
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

            {/* Extraction Results */}
            {lastTopicsResult && !lastTopicsResult.error && (
              <>
                <Separator />
                <TopicListSection
                  collectionName={collectionName}
                  result={lastTopicsResult}
                  renameTopicLabel={renameTopicLabel}
                  regenerateTopicLabel={regenerateTopicLabel}
                />
              </>
            )}

            {hasTopics && (
              <>
                <Separator />
                <ReduceTopicsSection
                  collectionName={collectionName}
                  reduceTopics={reduceTopics}
                  reduceTopicsLoading={reduceTopicsLoading}
                  topicsLoading={topicsLoading}
                  lastReduceResult={lastReduceResult}
                  onReduced={onTopicsExtracted}
                />

                <Separator />
                <LlmLabelingSection
                  collectionName={collectionName}
                  hasSubtopics={hasSubtopics}
                  generateLlmLabels={generateLlmLabels}
                  llmLabelsLoading={llmLabelsLoading}
                  otherOpsLoading={topicsLoading || reduceTopicsLoading}
                  lastLlmLabelsResult={lastLlmLabelsResult}
                  onLabeled={onTopicsExtracted}
                />

                <Separator />
                <TopicQualitySection
                  collectionName={collectionName}
                  hasSubtopics={hasSubtopics}
                  otherOpsLoading={topicsLoading || reduceTopicsLoading || llmLabelsLoading}
                  storedQualityMetrics={lastTopicsResult?.qualityMetrics}
                />
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
