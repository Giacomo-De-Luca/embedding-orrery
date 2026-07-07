'use client';

/**
 * Topic-quality scoring hook (mirrors the useProbes shape rather than growing
 * useEmbedDataset further). Completion is driven by the awaited mutation
 * promise; the `{collection}_evaluate` progress subscription is display-only
 * and lives in the consuming component's ProgressModal.
 */

import { useCallback, useState } from 'react';
import { useMutation } from '@apollo/client/react';
import { toast } from 'sonner';
import { EVALUATE_TOPICS } from '../graphql/queries';
import type { EvaluateTopicsInput, EvaluateTopicsResult } from '../graphql/mutations';

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
        if (result?.error) {
          setError(result.error);
          toast.error(`Topic scoring failed: ${result.error}`);
        } else if (result) {
          toast.success(
            `Scored ${result.level} quality for "${result.collectionName}" in ${result.durationSeconds.toFixed(1)}s`
          );
        }
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
