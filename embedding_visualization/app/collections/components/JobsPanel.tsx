'use client';

import { useQuery } from '@apollo/client/react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/lib/ui-primitives/card';
import { Button } from '@/lib/ui-primitives/button';
import { Progress } from '@/lib/ui-primitives/progress';
import { Badge } from '@/lib/ui-primitives/badge';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { GET_EMBEDDING_JOBS } from '@/lib/graphql/queries';
import type { EmbeddingJob } from '@/lib/graphql/mutations';
import { RefreshCw, Play, Square, X, AlertCircle } from 'lucide-react';

interface JobsQueryData {
  embeddingJobs: EmbeddingJob[];
}

interface JobsPanelProps {
  /** Called when user clicks resume on an interrupted job */
  onResumeJob?: (job: EmbeddingJob) => void;
  /** Called when user clicks cancel on a running job */
  onCancelJob?: (job: EmbeddingJob) => void;
  /** Called when user clicks remove on an interrupted job */
  onRemoveJob?: (job: EmbeddingJob) => void;
  /** Filter jobs by status */
  statusFilter?: 'running' | 'interrupted' | 'completed' | null;
}

/**
 * Displays a list of embedding jobs with their progress.
 * Shows interrupted jobs with resume/remove buttons and running jobs with cancel button.
 */
export function JobsPanel({ onResumeJob, onCancelJob, onRemoveJob, statusFilter = 'interrupted' }: JobsPanelProps) {
  const { data, loading, error, refetch } = useQuery<JobsQueryData>(GET_EMBEDDING_JOBS, {
    variables: statusFilter ? { status: statusFilter } : {},
    fetchPolicy: 'network-only',
    pollInterval: 5000, // Poll every 5 seconds for updates
  });

  const jobs = data?.embeddingJobs ?? [];

  if (loading && jobs.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6 flex items-center justify-center">
          <Spinner className="h-6 w-6" />
          <span className="ml-2 text-muted-foreground">Loading jobs...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-destructive">
        <CardContent className="pt-6">
          <div className="text-destructive flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            Failed to load jobs: {error.message}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (jobs.length === 0) {
    return null; // Don't show panel if no jobs
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>
              {statusFilter === 'interrupted' ? 'Interrupted Jobs' : 'Jobs'}
            </CardTitle>
            <CardDescription>
              {statusFilter === 'interrupted'
                ? 'These jobs were interrupted and can be resumed'
                : 'Current and recent embedding operations'}
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {jobs.map((job) => (
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
  onResume?: (job: EmbeddingJob) => void;
  onCancel?: (job: EmbeddingJob) => void;
  onRemove?: (job: EmbeddingJob) => void;
}

function JobCard({ job, onResume, onCancel, onRemove }: JobCardProps) {
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
        {job.status === 'running' && onCancel && (
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
