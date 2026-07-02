'use client';

import type { EmbeddingJob } from '@/lib/graphql/mutations';
import { ProgressModal } from './EmbeddingProgressModal';
import { JobsPanel } from './JobsPanel';

interface EmbedProgressSectionProps {
  embedLoading: boolean;
  activeJobCollectionName?: string | null;
  llmResumeJobId: string | null;
  onResumeJob: (job: EmbeddingJob) => Promise<void>;
  onCancelActiveJob?: () => void;
  cancelLoading?: boolean;
  onCancelJob?: (job: EmbeddingJob) => void;
  onRemoveJob?: (job: EmbeddingJob) => void;
}

export function EmbedProgressSection({
  embedLoading,
  activeJobCollectionName,
  llmResumeJobId,
  onResumeJob,
  onCancelActiveJob,
  cancelLoading,
  onCancelJob,
  onRemoveJob,
}: EmbedProgressSectionProps) {
  return (
    <>
      {/* Progress Modal - shown during embedding */}
      {embedLoading && activeJobCollectionName && (
        <ProgressModal
          jobId={activeJobCollectionName}
          onCancel={onCancelActiveJob}
          cancelLoading={cancelLoading}
        />
      )}

      {/* Progress Modal - shown during LLM labeling resume */}
      {llmResumeJobId && (
        <ProgressModal
          jobId={llmResumeJobId}
          title="Generating LLM Labels"
          subtitle="Each topic is labeled individually via LLM API calls."
          itemsLabel="topics"
        />
      )}

      {/* Interrupted Jobs Panel */}
      {!embedLoading && (
        <JobsPanel
          statusFilter="interrupted"
          onResumeJob={onResumeJob}
          onCancelJob={onCancelJob}
          onRemoveJob={onRemoveJob}
        />
      )}
    </>
  );
}
