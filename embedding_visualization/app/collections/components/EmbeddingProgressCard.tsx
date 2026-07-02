'use client';

import { useEffect, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/lib/ui-primitives/card';
import { Progress } from '@/lib/ui-primitives/progress';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { EMBEDDING_PROGRESS_SUBSCRIPTION } from '@/lib/graphql/queries';
import type { JobProgress } from '@/lib/graphql/mutations';

interface EmbeddingProgressCardProps {
  /** The collection name (job ID) to subscribe to */
  collectionName: string;
  /** Called when embedding completes */
  onComplete?: () => void;
  /** Called when embedding fails */
  onError?: (error: string) => void;
}

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

/**
 * Displays real-time progress during embedding operations.
 * Subscribes to GraphQL progress updates via WebSocket.
 */
export function EmbeddingProgressCard({
  collectionName,
  onComplete,
  onError,
}: EmbeddingProgressCardProps) {
  const [progress, setProgress] = useState<JobProgress | null>(null);

  const { data, error: subscriptionError } = useSubscription<SubscriptionData>(
    EMBEDDING_PROGRESS_SUBSCRIPTION,
    {
      variables: { jobId: collectionName },
      skip: !collectionName,
    }
  );

  // Update progress when subscription data arrives
  useEffect(() => {
    if (data?.embeddingProgress) {
      setProgress(data.embeddingProgress);

      // Handle completion
      if (data.embeddingProgress.status === 'completed') {
        onComplete?.();
      }

      // Handle failure
      if (data.embeddingProgress.status === 'failed' && data.embeddingProgress.error) {
        onError?.(data.embeddingProgress.error);
      }
    }
  }, [data, onComplete, onError]);

  // Handle subscription errors
  useEffect(() => {
    if (subscriptionError) {
      console.error('Subscription error:', subscriptionError);
    }
  }, [subscriptionError]);

  const percentComplete = progress
    ? Math.round((progress.itemsProcessed / progress.totalItems) * 100)
    : 0;

  const statusText = progress?.status === 'completed'
    ? 'Complete!'
    : progress?.status === 'failed'
    ? 'Failed'
    : 'Embedding...';

  return (
    <Card className="border-blue-500/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-lg">
          {progress?.status !== 'completed' && progress?.status !== 'failed' && (
            <Spinner className="h-5 w-5" />
          )}
          {statusText}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Progress value={percentComplete} className="h-3" />

        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Items:</span>{' '}
            <span className="font-medium">
              {progress?.itemsProcessed?.toLocaleString() ?? 0} / {progress?.totalItems?.toLocaleString() ?? '?'}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Progress:</span>{' '}
            <span className="font-medium">{percentComplete}%</span>
          </div>
          <div>
            <span className="text-muted-foreground">Batch:</span>{' '}
            <span className="font-medium">
              {progress?.currentBatch ?? 0} / {progress?.totalBatches ?? '?'}
            </span>
          </div>
          <div>
            <span className="text-muted-foreground">Collection:</span>{' '}
            <span className="font-medium font-mono text-xs">{collectionName}</span>
          </div>
        </div>

        {progress?.error && (
          <div className="text-destructive text-sm">
            <strong>Error:</strong> {progress.error}
          </div>
        )}

        {!progress && !subscriptionError && (
          <p className="text-sm text-muted-foreground">
            Waiting for progress updates...
          </p>
        )}

        {subscriptionError && (
          <p className="text-sm text-muted-foreground">
            Progress subscription unavailable. Embedding is running in the background.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
