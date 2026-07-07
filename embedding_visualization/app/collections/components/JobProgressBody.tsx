'use client';

import { Progress } from '@/lib/ui-primitives/progress';
import { Badge } from '@/lib/ui-primitives/badge';
import { Button } from '@/lib/ui-primitives/button';
import { Spinner } from '@/lib/ui-primitives/spinner';
import { Square } from 'lucide-react';
import { formatElapsed } from '../lib/jobProgress';
import { useJobProgress } from '../lib/useJobProgress';

export interface JobProgressBodyProps {
  /** Job ID to subscribe to for WebSocket progress updates */
  jobId: string;
  /** Title in the header (default: jobId) */
  title?: string;
  /** Hint text at the bottom (pass undefined to hide) */
  subtitle?: string;
  /** Label for the items counter, e.g. "topics" or "items" (default: "items") */
  itemsLabel?: string;
  /** Called when user clicks the Cancel button */
  onCancel?: () => void;
  /** Disable the Cancel button while the mutation is in flight */
  cancelLoading?: boolean;
}

/**
 * Layout-agnostic live progress display for a job: header with status badge,
 * elapsed/ETA, optional cancel, progress bar, and counters. Wrapped by
 * ProgressModal (centered blocking overlay).
 *
 * Supports two progress models:
 * - Stage-based (totalBatches > 1): bar tracks currentBatch/totalBatches,
 *   with sub-stage item progress blended in when available.
 * - Item-based (totalBatches <= 1): bar tracks itemsProcessed/totalItems.
 */
export function JobProgressBody({
  jobId,
  title,
  subtitle,
  itemsLabel = 'items',
  onCancel,
  cancelLoading,
}: JobProgressBodyProps) {
  const {
    progress,
    percent,
    elapsedMs,
    etaMs,
    subscriptionError,
    hasProgress,
    showItemCounter,
    isMultiStage,
  } = useJobProgress(jobId);

  const statusColor = {
    running: 'bg-blue-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
  }[progress?.status || 'running'];

  const displayTitle = title || jobId;

  return (
    <div className="space-y-4">
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
              {formatElapsed(elapsedMs)}
              {etaMs !== null && etaMs > 0 && ` · ~${formatElapsed(etaMs)} remaining`}
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
      {hasProgress && progress && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">{percent}%</span>
          </div>
          <Progress value={percent} className="h-2" />
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
      {subtitle && (
        <p className="text-xs text-muted-foreground text-center">
          {subtitle}
        </p>
      )}
    </div>
  );
}
