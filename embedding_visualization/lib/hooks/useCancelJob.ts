'use client';

import { useCallback } from 'react';
import { useMutation } from '@apollo/client/react';
import { CANCEL_JOB } from '../graphql/mutations';

/**
 * Request cancellation of a running backend job by its progress job id.
 * Resolves true when the backend found and signalled the job; the job then
 * stops at its next batch, so the caller keeps showing progress until the
 * job's own mutation resolves.
 */
export function useCancelJob(): (jobId: string) => Promise<boolean> {
  const [cancelJobMutation] = useMutation<{ cancelJob: boolean }, { jobId: string }>(CANCEL_JOB);

  return useCallback(async (jobId: string): Promise<boolean> => {
    try {
      const { data } = await cancelJobMutation({ variables: { jobId } });
      return data?.cancelJob ?? false;
    } catch {
      return false;
    }
  }, [cancelJobMutation]);
}
