import { describe, it, expect } from 'vitest';
import {
  JOB_CANCELLED_MESSAGE,
  announceJobOutcome,
  cancelButtonView,
  isJobCancelled,
  jobOutcome,
} from '../jobCancellation';

describe('announceJobOutcome', () => {
  const messages = {
    cancelled: (r: { error?: string | null }) => `cancelled (${r.error})`,
    failed: (r: { error?: string | null }) => `failed: ${r.error}`,
    succeeded: () => 'done',
  };

  function notifier() {
    const calls: string[] = [];
    return {
      calls,
      info: (m: string) => calls.push(`info ${m}`),
      error: (m: string) => calls.push(`error ${m}`),
      success: (m: string) => calls.push(`success ${m}`),
    };
  }

  it('routes each outcome to its notifier and returns it', () => {
    const n = notifier();
    expect(announceJobOutcome({ error: null }, messages, n)).toBe('succeeded');
    expect(announceJobOutcome({ error: 'boom' }, messages, n)).toBe('failed');
    expect(announceJobOutcome({ error: JOB_CANCELLED_MESSAGE }, messages, n)).toBe('cancelled');
    expect(n.calls).toEqual([
      'success done',
      'error failed: boom',
      `info cancelled (${JOB_CANCELLED_MESSAGE})`,
    ]);
  });

  it('announces nothing without a result', () => {
    const n = notifier();
    expect(announceJobOutcome(null, messages, n)).toBeNull();
    expect(n.calls).toEqual([]);
  });
});

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
