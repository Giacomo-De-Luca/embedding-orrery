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

export type JobOutcome = 'succeeded' | 'cancelled' | 'failed';

/** How a finished job ended, read from its result's error (null: no result). */
export function jobOutcome(result: { error?: string | null } | null | undefined): JobOutcome | null {
  if (!result) return null;
  if (!result.error) return 'succeeded';
  return isJobCancelled(result.error) ? 'cancelled' : 'failed';
}

/** Where outcome messages go: sonner's `toast` in the app. */
export interface OutcomeNotifier {
  info(message: string): unknown;
  error(message: string): unknown;
  success(message: string): unknown;
}

/**
 * Announce a finished job: a cancel as info, a failure as an error, a
 * success as success. Returns the outcome so the caller can update its own
 * state (e.g. keep the previous result on a cancel).
 */
export function announceJobOutcome<R extends { error?: string | null }>(
  result: R | null | undefined,
  messages: Record<JobOutcome, (result: R) => string>,
  notifier: OutcomeNotifier,
): JobOutcome | null {
  const outcome = jobOutcome(result);
  if (!outcome || !result) return null;
  const message = messages[outcome](result);
  if (outcome === 'cancelled') notifier.info(message);
  else if (outcome === 'failed') notifier.error(message);
  else notifier.success(message);
  return outcome;
}

/**
 * Cancel button lifecycle in a progress modal: `requesting` while the cancel
 * mutation is in flight, then `requested` until the job actually stops (at
 * its next checkpoint, or when its worker process is killed). A refused
 * request — the job is no longer running — returns to `idle`.
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
