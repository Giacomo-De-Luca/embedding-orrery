'use client';

/**
 * Topic-quality scoring hook (mirrors the useProbes shape rather than growing
 * useEmbedDataset further). Completion is driven by the awaited mutation
 * promise; the `JobIds.topicEvaluation` progress subscription is display-only
 * and lives in the consuming component's ProgressModal.
 */

import { useCallback, useState } from 'react';
import { useMutation } from '@apollo/client/react';
import { toast } from 'sonner';
import { EVALUATE_TOPICS } from '../graphql/queries';
import type { EvaluateTopicsInput, EvaluateTopicsResult } from '../graphql/mutations';
import { announceJobOutcome } from '../utils/jobCancellation';

export interface UseTopicQualityReturn {
  evaluateTopics: (input: EvaluateTopicsInput) => Promise<EvaluateTopicsResult | null>;
  loading: boolean;
  lastResult: EvaluateTopicsResult | null;
  error: string | null;
  clearError: () => void;
}

export function useTopicQuality(): UseTopicQualityReturn {
  const [mutate, { loading }] = useMutation<{ evaluateTopics: EvaluateTopicsResult }>(
    EVALUATE_TOPICS
  );
  const [lastResult, setLastResult] = useState<EvaluateTopicsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const evaluateTopics = useCallback(
    async (input: EvaluateTopicsInput): Promise<EvaluateTopicsResult | null> => {
      setError(null);
      try {
        const { data, error: gqlError } = await mutate({
          variables: { input },
          // C_v on large collections can take minutes.
          context: { fetchOptions: { timeout: 600000 } },
        });
        if (gqlError) {
          setError(gqlError.message);
          toast.error(gqlError.message);
          return null;
        }
        const result = data?.evaluateTopics ?? null;
        const outcome = announceJobOutcome(result, {
          cancelled: () => `Topic scoring for "${input.collectionName}" cancelled.`,
          failed: (r) => `Topic scoring failed: ${r.error}`,
          succeeded: (r) =>
            `Scored ${r.level} quality for "${r.collectionName}" in ${r.durationSeconds.toFixed(1)}s`,
        }, toast);
        // A cancel persisted nothing: keep showing the stored scores
        if (outcome === 'cancelled') return result;
        if (outcome === 'failed') setError(result!.error!);
        setLastResult(result);
        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to score topic quality';
        setError(message);
        toast.error(message);
        return null;
      }
    },
    [mutate]
  );

  return { evaluateTopics, loading, lastResult, error, clearError: () => setError(null) };
}
