/**
 * Tests for the pure model-id → HF checkpoint derivation (modelCheckpoints.ts,
 * re-exported by modelLoader.ts). Mirrors the backend's
 * services/model_registry.py: known model ids resolve via the record, unknown
 * gemma-style ids fall back to the legacy string rule.
 */
import { describe, it, expect } from 'vitest';

import { modelIdToCheckpoint, isModelMatch, isQwenModel, modelDisplayName } from '../modelCheckpoints';

describe('modelIdToCheckpoint', () => {
  it('maps gemma it ids to google checkpoints', () => {
    expect(modelIdToCheckpoint('gemma-3-4b-it')).toBe('google/gemma-3-4b-it');
  });

  it('appends -pt for gemma base ids (backend _normalize_checkpoint parity)', () => {
    expect(modelIdToCheckpoint('gemma-3-1b')).toBe('google/gemma-3-1b-pt');
  });

  it('maps qwen-scope model ids via the registry record', () => {
    // The model_id names the SAE training provenance (Base); the checkpoint
    // is the instruct model we chat with — deliberate, see the Phase-1 plan.
    expect(modelIdToCheckpoint('qwen3-1.7B-base')).toBe('Qwen/Qwen3-1.7B');
  });

  it('strips an org prefix before applying the gemma fallback', () => {
    expect(modelIdToCheckpoint('google/gemma-3-4b-it')).toBe('google/gemma-3-4b-it');
  });
});

describe('isModelMatch', () => {
  it('matches identical checkpoints', () => {
    expect(isModelMatch('google/gemma-3-4b-it', 'google/gemma-3-4b-it')).toBe(true);
  });

  it('matches when only one side carries the org prefix', () => {
    expect(isModelMatch('gemma-3-1b-pt', 'google/gemma-3-1b-pt')).toBe(true);
  });

  it('round-trips a qwen id through the registry', () => {
    expect(isModelMatch('Qwen/Qwen3-1.7B', modelIdToCheckpoint('qwen3-1.7B-base'))).toBe(true);
  });

  it('rejects null / mismatched models', () => {
    expect(isModelMatch(null, 'Qwen/Qwen3-1.7B')).toBe(false);
    expect(isModelMatch('google/gemma-3-4b-it', 'Qwen/Qwen3-1.7B')).toBe(false);
  });
});

describe('isQwenModel / modelDisplayName', () => {
  it('detects the qwen family case-insensitively', () => {
    expect(isQwenModel('qwen3-1.7B-base')).toBe(true);
    expect(isQwenModel('Qwen3.5-2B-base')).toBe(true);
    expect(isQwenModel('gemma-3-4b-it')).toBe(false);
    expect(isQwenModel(null)).toBe(false);
  });

  it('derives display names per family', () => {
    expect(modelDisplayName('qwen3-1.7B-base')).toBe('Qwen');
    expect(modelDisplayName('gemma-3-4b-it')).toBe('Gemma');
    expect(modelDisplayName(null)).toBe('Gemma');
    expect(modelDisplayName('other-model')).toBe('other-model');
  });
});
