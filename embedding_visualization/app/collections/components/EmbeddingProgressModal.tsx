'use client';

import { useEffect, useRef, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { Card, CardContent } from '@/lib/ui-primitives/card';
import { Progress } from '@/lib/ui-primitives/progress';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { EMBEDDING_PROGRESS_SUBSCRIPTION } from '@/lib/graphql/queries';
import type { JobProgress } from '@/lib/graphql/mutations';
import { Square } from 'lucide-react';

interface ProgressModalProps {
  /** Job ID to subscribe to for WebSocket progress updates */
  jobId: string;
  /** Title in modal header (default: jobId) */
  title?: string;
  /** Hint text at bottom (default: "This may take several minutes for large datasets.") */
  subtitle?: string;
  /** Label for the items counter, e.g. "topics" or "items" (default: "items") */
  itemsLabel?: string;
  /** Called when user clicks the Cancel button */
  onCancel?: () => void;
  /** Disable the Cancel button while the mutation is in flight */
  cancelLoading?: boolean;
}

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Compute progress percentage using a blended stage+item approach.
 *
 * Multi-stage operations (totalBatches > 1): base progress from currentBatch/totalBatches,
 * plus fractional item progress within the current stage when available.
 * Single-stage operations: items-only progress.
 */
function computePercent(p: JobProgress): number {
  const hasMeaningfulItems = p.totalItems > 0 && p.itemsProcessed > 0;
  const isMultiStage = p.totalBatches > 1;

  if (isMultiStage) {
    const stageWidth = 100 / p.totalBatches;
    const base = p.currentBatch * stageWidth;
    if (hasMeaningfulItems) {
      // Blend: stage base + fractional item progress within this stage
      const itemFraction = p.itemsProcessed / p.totalItems;
      return Math.min(100, Math.round(base + itemFraction * stageWidth));
    }
    return Math.min(100, Math.round(base));
  }

  if (p.totalItems > 0) {
    return Math.round((p.itemsProcessed / p.totalItems) * 100);
  }

  return 0;
}

/**
 * Modal overlay displaying real-time job progress.
 * Uses the same card-style layout as JobsPanel for consistency.
 * Subscribes to GraphQL progress updates via WebSocket.
 *
 * Supports two progress models:
 * - Stage-based (totalBatches > 1): bar tracks currentBatch/totalBatches,
 *   with sub-stage item progress blended in when available.
 * - Item-based (totalBatches <= 1): bar tracks itemsProcessed/totalItems.
 */
export function ProgressModal({
  jobId,
  title,
  subtitle = 'This may take several minutes for large datasets.',
  itemsLabel = 'items',
  onCancel,
  cancelLoading,
}: ProgressModalProps) {
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const startTimeRef = useRef(Date.now());
  const [elapsed, setElapsed] = useState(0);

  // ETA tracking: record the first meaningful progress update
  const etaBaseRef = useRef<{ time: number; items: number } | null>(null);
  const lastTotalRef = useRef<number>(0);
  const lastItemsRef = useRef<number>(0);
  const [eta, setEta] = useState<number | null>(null);

  const { data, error: subscriptionError } = useSubscription<SubscriptionData>(
    EMBEDDING_PROGRESS_SUBSCRIPTION,
    {
      variables: { jobId },
      skip: !jobId,
    }
  );

  // Update progress when subscription data arrives
  useEffect(() => {
    if (data?.embeddingProgress) {
      const p = data.embeddingProgress;
      setProgress(p);

      // Calculate ETA when we have meaningful item-level progress
      if (p.totalItems > 0 && p.itemsProcessed > 0) {
        // Reset baseline when totalItems changes or itemsProcessed regresses (new phase)
        if (
          p.totalItems !== lastTotalRef.current ||
          p.itemsProcessed < lastItemsRef.current
        ) {
          lastTotalRef.current = p.totalItems;
          etaBaseRef.current = null;
          setEta(null);
        }
        lastItemsRef.current = p.itemsProcessed;

        if (etaBaseRef.current === null) {
          // Record the first progress update as our baseline
          etaBaseRef.current = { time: Date.now(), items: p.itemsProcessed };
        } else if (p.itemsProcessed > etaBaseRef.current.items) {
          // We have at least 2 data points — calculate ETA
          const elapsedSinceBase = Date.now() - etaBaseRef.current.time;
          const itemsSinceBase = p.itemsProcessed - etaBaseRef.current.items;
          const avgTimePerItem = elapsedSinceBase / itemsSinceBase;
          const remaining = avgTimePerItem * (p.totalItems - p.itemsProcessed);
          setEta(remaining);
        }
      }
    }
  }, [data]);

  // Handle subscription errors
  useEffect(() => {
    if (subscriptionError) {
      console.error('Progress subscription error:', subscriptionError);
    }
  }, [subscriptionError]);

  // Elapsed timer
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTimeRef.current);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const percentComplete = progress ? computePercent(progress) : 0;

  // Show progress bar when we have stage-based or item-based progress
  const hasProgress = progress && (
    (progress.totalItems > 0) ||
    (progress.totalBatches > 1 && progress.currentBatch > 0)
  );

  // Determine whether to show item counter (only when items are actually meaningful)
  const showItemCounter = progress && progress.totalItems > 0 && progress.itemsProcessed > 0;

  // Show stage counter for multi-stage operations
  const isMultiStage = progress && progress.totalBatches > 1;

  const statusColor = {
    running: 'bg-blue-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
  }[progress?.status || 'running'];

  const displayTitle = title || jobId;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
      <Card className="w-[650px]">
        <CardContent className="pt-6 space-y-4">
          {/* Header with title and status */}
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="font-medium font-mono text-sm">{displayTitle}</span>
                <Badge variant="outline" className="text-xs">
                  <span className={`w-2 h-2 rounded-full ${statusColor} mr-1`} />
                  {progress?.status || 'initializing'}
                </Badge>
                <span className="text-xs text-muted-foreground font-mono">
                  {formatElapsed(elapsed)}
                  {eta !== null && eta > 0 && ` · ~${formatElapsed(eta)} remaining`}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {onCancel && (!progress || progress.status === 'running') && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={onCancel}
                  disabled={cancelLoading}
                  className="gap-1"
                >
                  <Square className="h-3 w-3" />
                  {cancelLoading ? 'Cancelling...' : 'Cancel'}
                </Button>
              )}
              {(!progress || progress.status === 'running') && (
                <Spinner className="h-5 w-5" />
              )}
            </div>
          </div>

          {/* Status message */}
          {progress?.message && (
            <p className="text-sm text-muted-foreground">
              {progress.message}
            </p>
          )}

          {/* Progress bar and stats */}
          {hasProgress && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-medium">{percentComplete}%</span>
              </div>
              <Progress value={percentComplete} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>
                  {showItemCounter
                    ? `${progress.itemsProcessed.toLocaleString()} / ${progress.totalItems.toLocaleString()} ${itemsLabel}`
                    : isMultiStage
                      ? `Stage ${Math.floor(progress.currentBatch)} / ${progress.totalBatches}`
                      : `0 / ${progress.totalItems.toLocaleString()} ${itemsLabel}`
                  }
                </span>
                {isMultiStage && showItemCounter && (
                  <span>
                    Stage {Math.floor(progress.currentBatch)} / {progress.totalBatches}
                  </span>
                )}
                {!isMultiStage && progress.totalBatches > 0 && (
                  <span>
                    Batch {progress.currentBatch} / {progress.totalBatches}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Initial state (no progress yet) */}
          {!hasProgress && !subscriptionError && (
            <div className="text-center py-4">
              <p className="text-sm text-muted-foreground">
                {progress?.message || `Initializing ${displayTitle.toLowerCase()}...`}
              </p>
            </div>
          )}

          {/* Error state */}
          {subscriptionError && (
            <div className="text-center py-4">
              <p className="text-sm text-destructive">
                Connection error. Progress updates may be delayed.
              </p>
            </div>
          )}

          {/* Helpful note */}
          <p className="text-xs text-muted-foreground text-center">
            {subtitle}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

/** @deprecated Use ProgressModal instead */
export const EmbeddingProgressModal = ProgressModal;
