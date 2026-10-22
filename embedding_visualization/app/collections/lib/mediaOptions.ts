/**
 * Media options of the Image / Audio / Video / Mixed embed form. The
 * `mediaEmbeddingSettings` query serves their defaults and limits
 * (config/media_embedding.json); a job sends only the options it changes, for
 * the kinds it embeds. Pure (no React).
 *
 * The limits and the token estimate mirror the backend's
 * `MediaEmbeddingSettings` (`_validated`, `estimate_tokens`, `check_budget` in
 * interpretability_backend/backend/utils/media_settings.py): EmbeddingGemma 2
 * reads every modality of a row in one context window.
 */
import type { MediaEmbeddingSettings, MediaOptions, MediaOptionsInput } from '@/lib/graphql/mutations';
import type { MediaKind, MediaKindCounts } from './mediaColumns';

export type MediaOptionKey = keyof MediaOptions;

/** The media kind each option applies to. */
export const MEDIA_OPTION_KINDS: Record<MediaOptionKey, MediaKind> = {
  imageMaxSoftTokens: 'image',
  videoFps: 'video',
  videoMaxFrames: 'video',
  videoMaxSoftTokens: 'video',
  maxAudioSeconds: 'audio',
};

const OPTION_KEYS = Object.keys(MEDIA_OPTION_KINDS) as MediaOptionKey[];

const isWhole = (value: number) => Number.isInteger(value);

export class MediaOptionsBudget {
  constructor(readonly settings: MediaEmbeddingSettings) {}

  /** The options in effect: the defaults with the overrides applied (null keeps the default). */
  effective(overrides?: MediaOptionsInput | null): MediaOptions {
    const options = { ...this.settings.defaults };
    for (const key of OPTION_KEYS) {
      const value = overrides?.[key];
      if (value !== null && value !== undefined) options[key] = value;
    }
    return options;
  }

  /** The overrides after setting one option; a value equal to its default is dropped. */
  withOption(overrides: MediaOptionsInput, key: MediaOptionKey, value: number): MediaOptionsInput {
    const next = { ...overrides };
    if (value === this.settings.defaults[key]) delete next[key];
    else next[key] = value;
    return next;
  }

  /**
   * What a job sends: the changed options of the kinds it embeds, or nothing
   * (the backend then uses its defaults).
   */
  request(overrides: MediaOptionsInput, kinds: MediaKindCounts): MediaOptionsInput | undefined {
    const request: MediaOptionsInput = {};
    for (const key of OPTION_KEYS) {
      const value = overrides[key];
      if (!kinds[MEDIA_OPTION_KINDS[key]] || value === null || value === undefined) continue;
      if (value !== this.settings.defaults[key]) request[key] = value;
    }
    return Object.keys(request).length > 0 ? request : undefined;
  }

  /** Worst-case tokens of one row with `kinds` media columns of each kind. */
  estimateTokens(options: MediaOptions, kinds: MediaKindCounts): number {
    const s = this.settings;
    const audioTokens = Math.ceil(options.maxAudioSeconds * s.audioTokensPerSecond);
    return (
      s.reservedTokens +
      (kinds.image ?? 0) * options.imageMaxSoftTokens +
      (kinds.video ?? 0) * options.videoMaxFrames * options.videoMaxSoftTokens +
      (kinds.audio ?? 0) * audioTokens
    );
  }

  /** Problems the backend would reject the job for: options out of range, a row over the context. */
  issues(options: MediaOptions, kinds: MediaKindCounts): string[] {
    const s = this.settings;
    const issues: string[] = [];
    const budgets = s.softTokenBudgets.join(', ');
    if (kinds.image && !s.softTokenBudgets.includes(options.imageMaxSoftTokens)) {
      issues.push(`Tokens per image must be one of ${budgets}`);
    }
    if (kinds.video) {
      if (!(options.videoFps > 0 && options.videoFps <= s.videoFpsLimit)) {
        issues.push(`Frames per second must be above 0 and at most ${s.videoFpsLimit}`);
      }
      if (!isWhole(options.videoMaxFrames) || options.videoMaxFrames < 1 || options.videoMaxFrames > s.videoFramesLimit) {
        issues.push(`Max frames must be a whole number from 1 to ${s.videoFramesLimit}`);
      }
      if (!s.softTokenBudgets.includes(options.videoMaxSoftTokens)) {
        issues.push(`Tokens per frame must be one of ${budgets}`);
      }
    }
    if (kinds.audio && !(options.maxAudioSeconds > 0 && options.maxAudioSeconds <= s.audioSecondsLimit)) {
      issues.push(`Audio length must be above 0 and at most ${s.audioSecondsLimit} seconds`);
    }
    const estimate = this.estimateTokens(options, kinds);
    if (estimate > s.contextTokens) {
      issues.push(
        `A row could need about ${estimate} tokens, more than the model's ${s.contextTokens}-token context. ` +
          'Lower the video frames, the tokens per image or frame, or the audio seconds.',
      );
    }
    return issues;
  }
}
