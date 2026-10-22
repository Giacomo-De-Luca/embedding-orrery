import { describe, expect, it } from 'vitest';

import type { MediaEmbeddingSettings } from '@/lib/graphql/mutations';
import { MediaOptionsBudget } from '../mediaOptions';

/** The shipped config/media_embedding.json, as the mediaEmbeddingSettings query serves it. */
const SETTINGS: MediaEmbeddingSettings = {
  defaults: {
    imageMaxSoftTokens: 280,
    videoFps: 1,
    videoMaxFrames: 32,
    videoMaxSoftTokens: 140,
    maxAudioSeconds: 120,
  },
  softTokenBudgets: [70, 140, 280, 560, 1120],
  videoFpsLimit: 4,
  videoFramesLimit: 128,
  audioSecondsLimit: 327,
  contextTokens: 8192,
  reservedTokens: 512,
  audioTokensPerSecond: 25,
  defaultModel: 'google/embeddinggemma-2',
};

const budget = new MediaOptionsBudget(SETTINGS);

describe('MediaOptionsBudget defaults and overrides', () => {
  it('starts from the defaults the query serves', () => {
    expect(budget.effective()).toEqual(SETTINGS.defaults);
    expect(budget.effective({ videoFps: 2, videoMaxFrames: null })).toEqual({ ...SETTINGS.defaults, videoFps: 2 });
  });

  it('keeps only values that differ from the defaults', () => {
    const changed = budget.withOption({}, 'videoMaxFrames', 16);
    expect(changed).toEqual({ videoMaxFrames: 16 });
    expect(budget.withOption(changed, 'videoMaxFrames', 32)).toEqual({});
    expect(budget.withOption(changed, 'imageMaxSoftTokens', 560)).toEqual({ videoMaxFrames: 16, imageMaxSoftTokens: 560 });
  });

  it('sends only the changed options of the selected kinds, or nothing', () => {
    const overrides = { imageMaxSoftTokens: 560, videoMaxFrames: 16, maxAudioSeconds: 120 };
    expect(budget.request(overrides, { image: 1 })).toEqual({ imageMaxSoftTokens: 560 });
    expect(budget.request(overrides, { video: 1, audio: 2 })).toEqual({ videoMaxFrames: 16 });
    expect(budget.request({}, { image: 1 })).toBeUndefined();
    expect(budget.request(overrides, {})).toBeUndefined();
  });
});

describe('MediaOptionsBudget.estimateTokens (backend MediaEmbeddingSettings.estimate_tokens)', () => {
  const defaults = budget.effective();

  it('adds each media column to the reserved text allowance', () => {
    expect(budget.estimateTokens(defaults, {})).toBe(512);
    expect(budget.estimateTokens(defaults, { image: 2 })).toBe(512 + 2 * 280);
    expect(budget.estimateTokens(defaults, { video: 1 })).toBe(512 + 32 * 140);
    expect(budget.estimateTokens(defaults, { audio: 1 })).toBe(512 + 120 * 25);
    expect(budget.estimateTokens(defaults, { image: 1, video: 1, audio: 1 })).toBe(8272);
  });

  it('rounds the audio tokens up like the backend', () => {
    expect(budget.estimateTokens({ ...defaults, maxAudioSeconds: 10.01 }, { audio: 1 })).toBe(512 + 251);
  });
});

describe('MediaOptionsBudget.issues', () => {
  const defaults = budget.effective();

  it('passes rows that fit the context', () => {
    expect(budget.issues(defaults, { image: 1, video: 1 })).toEqual([]);
  });

  it('rejects a row that could exceed the context, with the estimate', () => {
    expect(budget.issues(defaults, { image: 1, video: 1, audio: 1 })).toEqual([
      "A row could need about 8272 tokens, more than the model's 8192-token context. " +
        'Lower the video frames, the tokens per image or frame, or the audio seconds.',
    ]);
  });

  it('checks the options of the selected kinds against the limits', () => {
    expect(budget.issues({ ...defaults, imageMaxSoftTokens: 300 }, { image: 1 })).toEqual([
      'Tokens per image must be one of 70, 140, 280, 560, 1120',
    ]);
    expect(budget.issues({ ...defaults, videoFps: 5, videoMaxFrames: 1.5 }, { video: 1 })).toEqual([
      'Frames per second must be above 0 and at most 4',
      'Max frames must be a whole number from 1 to 128',
    ]);
    expect(budget.issues({ ...defaults, maxAudioSeconds: 0 }, { audio: 1 })).toEqual([
      'Audio length must be above 0 and at most 327 seconds',
    ]);
    // Options of kinds the job does not embed are never sent, so never checked.
    expect(budget.issues({ ...defaults, maxAudioSeconds: 0 }, { image: 1 })).toEqual([]);
  });
});
