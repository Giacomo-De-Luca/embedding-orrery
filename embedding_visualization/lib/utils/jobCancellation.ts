/**
 * Cancelling long-running backend jobs (embedding, LLM topic labeling, SAE
 * document activations) from their progress modal. The backend stops a
 * cancelled job at its next batch / topic / document, keeps the work done so
 * far, and reports JOB_CANCELLED_MESSAGE as the job's error.
 */

/** Error a cancelled job reports. Mirrors `CANCELLED_MESSAGE` in
 * interpretability_backend/backend/services/cancel_registry.py. */
export const JOB_CANCELLED_MESSAGE = 'Cancelled by user';

/** True when a job result's error means the user cancelled it (not a failure). */
export function isJobCancelled(error: string | null | undefined): boolean {
  return error === JOB_CANCELLED_MESSAGE;
}

/**
 * Cancel button lifecycle in a progress modal: `requesting` while the cancel
 * mutation is in flight, then `requested` until the job actually stops
 * (it finishes its current batch first). A refused request — the job is no
 * longer running — returns to `idle`.
 */
export type CancelState = 'idle' | 'requesting' | 'requested';

export interface CancelButtonView {
  visible: boolean;
  disabled: boolean;
  label: string;
}

/** What the progress modal's Cancel button shows for a cancel state and job
 * status (`undefined` = no progress event received yet). */
export function cancelButtonView(
  state: CancelState,
  jobStatus: string | undefined,
): CancelButtonView {
  const pending = state !== 'idle';
  return {
    visible: jobStatus === undefined || jobStatus === 'running',
    disabled: pending,
    label: pending ? 'Cancelling…' : 'Cancel',
  };
}
