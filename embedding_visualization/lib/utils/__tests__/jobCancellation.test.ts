import { describe, it, expect } from 'vitest';
import {
  JOB_CANCELLED_MESSAGE,
  cancelButtonView,
  isJobCancelled,
  jobOutcome,
} from '../jobCancellation';

describe('jobOutcome', () => {
  it('classifies a job result by its error', () => {
    expect(jobOutcome({ error: null })).toBe('succeeded');
    expect(jobOutcome({})).toBe('succeeded');
    expect(jobOutcome({ error: JOB_CANCELLED_MESSAGE })).toBe('cancelled');
    expect(jobOutcome({ error: 'boom' })).toBe('failed');
  });

  it('is null when there is no result', () => {
    expect(jobOutcome(null)).toBeNull();
    expect(jobOutcome(undefined)).toBeNull();
  });
});

describe('isJobCancelled', () => {
  it('matches the backend cancellation error exactly', () => {
    expect(isJobCancelled(JOB_CANCELLED_MESSAGE)).toBe(true);
    expect(isJobCancelled('Cancelled by user')).toBe(true);
  });

  it('is false for real errors and for no error', () => {
    expect(isJobCancelled('Model load failed')).toBe(false);
    expect(isJobCancelled(null)).toBe(false);
    expect(isJobCancelled(undefined)).toBe(false);
    expect(isJobCancelled('')).toBe(false);
  });
});

describe('cancelButtonView', () => {
  it('offers Cancel before the first progress event and while running', () => {
    expect(cancelButtonView('idle', undefined)).toEqual({ visible: true, disabled: false, label: 'Cancel' });
    expect(cancelButtonView('idle', 'running')).toEqual({ visible: true, disabled: false, label: 'Cancel' });
  });

  it('stays disabled on "Cancelling…" from the click until the job stops', () => {
    for (const state of ['requesting', 'requested'] as const) {
      expect(cancelButtonView(state, 'running')).toEqual({
        visible: true,
        disabled: true,
        label: 'Cancelling…',
      });
    }
  });

  it('hides once the job has finished, failed or been cancelled', () => {
    expect(cancelButtonView('idle', 'completed').visible).toBe(false);
    expect(cancelButtonView('requested', 'failed').visible).toBe(false);
  });
});
