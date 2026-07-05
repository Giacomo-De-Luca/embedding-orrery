'use client';

import { useEffect, useRef, useState } from 'react';
import { useSubscription } from '@apollo/client/react';
import { EMBEDDING_PROGRESS_SUBSCRIPTION } from '@/lib/graphql/queries';
import type { JobProgress } from '@/lib/graphql/mutations';
import { computePercent, nextEtaState, type EtaState } from './jobProgress';

interface SubscriptionData {
  embeddingProgress: JobProgress;
}

export interface JobProgressState {
  progress: JobProgress | null;
  /** Blended stage+item percentage (0-100) */
  percent: number;
  /** Milliseconds since this jobId started being observed */
  elapsedMs: number;
  /** Estimated remaining milliseconds, or null before two data points exist */
  etaMs: number | null;
  subscriptionError: Error | undefined;
  /** Whether there is stage- or item-based progress worth rendering a bar for */
  hasProgress: boolean;
  /** Whether the item counter is meaningful */
  showItemCounter: boolean;
  isMultiStage: boolean;
}

/**
 * Subscribes to WebSocket progress for a job and derives elapsed time, ETA,
 * and percentage. Layout-agnostic — used by the modal, the dock, and any
 * future progress surface. Resets its timers when jobId changes so a
 * persistently mounted consumer can track successive jobs.
 */
export function useJobProgress(jobId: string | null): JobProgressState {
  const [progress, setProgress] = useState<JobProgress | null>(null);
  const startTimeRef = useRef(Date.now());
  const [elapsedMs, setElapsedMs] = useState(0);
  const etaRef = useRef<EtaState | null>(null);
  const [etaMs, setEtaMs] = useState<number | null>(null);

  const { data, error: subscriptionError } = useSubscription<SubscriptionData>(
    EMBEDDING_PROGRESS_SUBSCRIPTION,
    {
      variables: { jobId },
      skip: !jobId,
    }
  );

  // Reset when tracking a different job (consumer may stay mounted)
  useEffect(() => {
    setProgress(null);
    startTimeRef.current = Date.now();
    setElapsedMs(0);
    etaRef.current = null;
    setEtaMs(null);
  }, [jobId]);

  // Fold subscription updates into progress + ETA state
  useEffect(() => {
    if (data?.embeddingProgress) {
      const p = data.embeddingProgress;
      setProgress(p);
      etaRef.current = nextEtaState(etaRef.current, p, Date.now());
      setEtaMs(etaRef.current?.etaMs ?? null);
    }
  }, [data]);

  useEffect(() => {
    if (subscriptionError) {
      console.error('Progress subscription error:', subscriptionError);
    }
  }, [subscriptionError]);

  // Elapsed timer
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const percent = progress ? computePercent(progress) : 0;
  const hasProgress =
    !!progress &&
    (progress.totalItems > 0 || (progress.totalBatches > 1 && progress.currentBatch > 0));
  const showItemCounter = !!progress && progress.totalItems > 0 && progress.itemsProcessed > 0;
  const isMultiStage = !!progress && progress.totalBatches > 1;

  return {
    progress,
    percent,
    elapsedMs,
    etaMs,
    subscriptionError,
    hasProgress,
    showItemCounter,
    isMultiStage,
  };
}
