'use client';

import { useQuery } from '@apollo/client/react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/lib/ui-primitives/card';
import { Button } from '@/lib/ui-primitives/button';
import { Progress } from '@/lib/ui-primitives/progress';
import { Badge } from '@/lib/ui-primitives/badge';
import { GET_EMBEDDING_JOBS } from '@/lib/graphql/queries';
import type { EmbeddingJob, JobStatus } from '@/lib/graphql/mutations';
import { RefreshCw, Play, Square, X } from 'lucide-react';

interface JobsQueryData {
  embeddingJobs: EmbeddingJob[];
}

/** Poll the backend job registry. Pass null to fetch jobs of every status. */
export function useEmbeddingJobs(statusFilter: JobStatus | null) {
  const { data, loading, error, refetch } = useQuery<JobsQueryData>(GET_EMBEDDING_JOBS, {
    variables: statusFilter ? { status: statusFilter } : {},
    fetchPolicy: 'network-only',
    pollInterval: 5000,
  });
  return { jobs: data?.embeddingJobs ?? [], loading, error, refetch };
}

interface ActiveJobsStripProps {
  onResumeJob: (job: EmbeddingJob) => void | Promise<void>;
  onCancelJob: (job: EmbeddingJob) => void;
  onRemoveJob: (job: EmbeddingJob) => void;
  /**
   * Job ids (registry keys / collection names) already surfaced elsewhere
   * (e.g. by the progress dock for a client-initiated embed) — hidden here
   * to avoid showing the same job twice.
   */
  hideJobIds?: (string | null | undefined)[];
}

/**
 * Page-global slim panel listing running + interrupted jobs, driven purely by
 * polled server state — so jobs stay visible after a page reload, regardless
 * of which tab started them. Renders nothing when there are no jobs.
 */
export function ActiveJobsStrip({
  onResumeJob,
  onCancelJob,
  onRemoveJob,
  hideJobIds,
}: ActiveJobsStripProps) {
  const { jobs, refetch } = useEmbeddingJobs(null);

  const visibleJobs = jobs.filter(
    (job) =>
      job.status !== 'completed' &&
      !(hideJobIds ?? []).includes(job.collectionName)
  );

  if (visibleJobs.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Jobs</CardTitle>
            <CardDescription>
              Running and interrupted embedding operations
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {visibleJobs.map((job) => (
          <JobCard
            key={job.collectionName}
            job={job}
            onResume={onResumeJob}
            onCancel={onCancelJob}
            onRemove={onRemoveJob}
          />
        ))}
      </CardContent>
    </Card>
  );
}

interface JobCardProps {
  job: EmbeddingJob;
  onResume?: (job: EmbeddingJob) => void | Promise<void>;
  onCancel?: (job: EmbeddingJob) => void;
  onRemove?: (job: EmbeddingJob) => void;
}

export function JobCard({ job, onResume, onCancel, onRemove }: JobCardProps) {
  const statusColor = {
    running: 'bg-blue-500',
    interrupted: 'bg-yellow-500',
    completed: 'bg-green-500',
  }[job.status];

  const isLlmLabeling = job.jobType === 'llm_labeling';
  const llmConfig = isLlmLabeling ? job.config as { collection_name?: string; llm_provider?: string; llm_model?: string } : null;

  return (
    <div className="border rounded-lg p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-medium font-mono text-sm">
              {llmConfig?.collection_name || job.collectionName}
            </span>
            <Badge variant="outline" className="text-xs">
              <span className={`w-2 h-2 rounded-full ${statusColor} mr-1`} />
              {job.status}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            {isLlmLabeling ? (
              <>LLM Labeling{llmConfig?.llm_provider && ` • ${llmConfig.llm_provider}/${llmConfig.llm_model}`}</>
            ) : (
              <>Source: {job.source}{job.embeddingModel && ` • Model: ${job.embeddingModel}`}</>
            )}
          </p>
        </div>
        {/* Cancel only for non-LLM jobs: backend cancel registration for
            llm_labeling is unverified */}
        {job.status === 'running' && onCancel && !isLlmLabeling && (
          <Button
            size="sm"
            variant="destructive"
            onClick={() => onCancel(job)}
            className="gap-1"
          >
            <Square className="h-3 w-3" />
            Cancel
          </Button>
        )}
        {job.status === 'interrupted' && (
          <div className="flex gap-2">
            {onResume && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onResume(job)}
                className="gap-1"
              >
                <Play className="h-3 w-3" />
                Resume
              </Button>
            )}
            {onRemove && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onRemove(job)}
                className="gap-1 text-muted-foreground"
              >
                <X className="h-3 w-3" />
                Remove
              </Button>
            )}
          </div>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Progress</span>
          <span className="font-medium">{job.percentComplete.toFixed(1)}%</span>
        </div>
        <Progress value={job.percentComplete} className="h-2" />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>
            {job.itemsEmbedded.toLocaleString()} / {job.totalExpected.toLocaleString()} {isLlmLabeling ? 'topics' : 'items'}
          </span>
          <span>
            Batch {job.batchesCompleted} / {job.totalBatches}
          </span>
        </div>
      </div>

      {job.columns && job.columns.length > 0 && (
        <div className="text-xs text-muted-foreground">
          Columns: {job.columns.join(', ')}
        </div>
      )}
    </div>
  );
}
