/**
 * Tests for the pure helpers in useSteeringChat.ts.
 *
 * These cover the strength-0 filter behaviour: auto-loaded presets sitting at
 * strength 0 must not (a) trigger a chat reset on appearance/removal, or
 * (b) be shipped to the backend (which would waste SAE loads on no-ops).
 */

import { describe, it, expect } from 'vitest';
import {
  activeSteeringFeatures,
  configKey,
  buildSteeringInputs,
  buildStreamInput,
  parseMessageParts,
} from '../useSteeringChat';
import type { SteeringConfig, SteeringFeature } from '@/lib/types/types';

const SAE_PRESET: SteeringFeature = {
  modelId: 'gemma-3-4b-it',
  saeId: '9-gemmascope-2-res-16k',
  layerIndex: 9,
  featureIndex: 197,
  strength: 0,
  hookType: 'RESID_POST',
  width: '16k',
};

const REFUSAL_PRESET: SteeringFeature = {
  modelId: 'gemma-3-4b-it',
  saeId: '',
  layerIndex: 14,
  featureIndex: 0,
  strength: 0,
  directionName: 'refusal',
};

const POETRY_PRESET: SteeringFeature = {
  modelId: 'gemma-3-4b-it',
  saeId: '',
  layerIndex: 11,
  featureIndex: 0,
  strength: 0,
  directionName: 'poetry',
};

describe('activeSteeringFeatures', () => {
  it('filters strength-0 entries and keeps the rest', () => {
    const features = [SAE_PRESET, { ...REFUSAL_PRESET, strength: -1 }, POETRY_PRESET];
    const active = activeSteeringFeatures(features);
    expect(active).toHaveLength(1);
    expect(active[0].directionName).toBe('refusal');
  });

  it('returns empty for an all-inert preset bundle', () => {
    expect(activeSteeringFeatures([SAE_PRESET, REFUSAL_PRESET, POETRY_PRESET])).toEqual([]);
  });
});

describe('buildSteeringInputs', () => {
  it('returns null when all features sit at strength 0', () => {
    const config: SteeringConfig = { features: [SAE_PRESET, REFUSAL_PRESET, POETRY_PRESET] };
    expect(buildSteeringInputs(config)).toBeNull();
  });

  it('returns null for the empty baseline config (unsteered generation)', () => {
    // The compare-mode baseline thread runs on { features: [] }; the backend
    // treats null steering as an unsteered generation.
    expect(buildSteeringInputs({ features: [] })).toBeNull();
  });

  it('includes only non-zero entries when one preset is activated', () => {
    const config: SteeringConfig = {
      features: [SAE_PRESET, { ...REFUSAL_PRESET, strength: -1 }, POETRY_PRESET],
    };
    const inputs = buildSteeringInputs(config);
    expect(inputs).not.toBeNull();
    expect(inputs).toHaveLength(1);
    expect(inputs![0].directionName).toBe('refusal');
    expect(inputs![0].strength).toBe(-1);
  });

  it('passes directionName through to the GraphQL payload', () => {
    const config: SteeringConfig = {
      features: [{ ...POETRY_PRESET, strength: 1.5 }],
    };
    const inputs = buildSteeringInputs(config);
    expect(inputs![0].directionName).toBe('poetry');
  });

  it('emits null directionName for SAE-feature entries', () => {
    const config: SteeringConfig = {
      features: [{ ...SAE_PRESET, strength: 800 }],
    };
    const inputs = buildSteeringInputs(config);
    expect(inputs![0].directionName).toBeNull();
    expect(inputs![0].featureIndex).toBe(197);
  });
});

describe('buildStreamInput', () => {
  const TURNS = [{ role: 'user', content: 'hi' }];

  it('threads a concrete seed into the payload for reproducible sampling', () => {
    const input = buildStreamInput(TURNS, null, 256, 0.7, 42);
    expect(input.seed).toBe(42);
    expect(input.outputLen).toBe(256);
    expect(input.temperature).toBe(0.7);
    expect(input.steering).toBeNull();
  });

  it('coalesces an omitted seed to null (non-deterministic)', () => {
    expect(buildStreamInput(TURNS, null, 256, 0.7).seed).toBeNull();
  });

  it('passes an explicit null seed through (unseeded default in ChatPanel)', () => {
    expect(buildStreamInput(TURNS, null, 256, 0.7, null).seed).toBeNull();
  });

  it('a steered and a baseline call with the same seed share the seed', () => {
    const steering = buildSteeringInputs({ features: [{ ...SAE_PRESET, strength: 800 }] });
    const steered = buildStreamInput(TURNS, steering, 128, 0.8, 7);
    const baseline = buildStreamInput(TURNS, null, 128, 0.8, 7);
    expect(steered.seed).toBe(baseline.seed);
    expect(steered.steering).not.toBeNull();
    expect(baseline.steering).toBeNull();
  });

  it('defaults enableThinking to false (thinking suppressed)', () => {
    expect(buildStreamInput(TURNS, null, 256, 0.7).enableThinking).toBe(false);
  });

  it('passes enableThinking through when toggled on (Qwen thinking mode)', () => {
    expect(buildStreamInput(TURNS, null, 256, 0.7, null, true).enableThinking).toBe(true);
  });
});

describe('configKey', () => {
  it('returns the same key when only strength-0 entries are added or removed', () => {
    const empty: SteeringConfig = { features: [] };
    const withPresets: SteeringConfig = { features: [SAE_PRESET, REFUSAL_PRESET, POETRY_PRESET] };
    expect(configKey(empty)).toBe(configKey(withPresets));
  });

  it('changes the key when a preset is dialled in', () => {
    const before: SteeringConfig = { features: [REFUSAL_PRESET] };
    const after: SteeringConfig = { features: [{ ...REFUSAL_PRESET, strength: -1 }] };
    expect(configKey(before)).not.toBe(configKey(after));
  });

  it('is stable under insertion order for direction presets', () => {
    const a: SteeringConfig = {
      features: [
        { ...REFUSAL_PRESET, strength: -1 },
        { ...POETRY_PRESET, strength: 1 },
      ],
    };
    const b: SteeringConfig = {
      features: [
        { ...POETRY_PRESET, strength: 1 },
        { ...REFUSAL_PRESET, strength: -1 },
      ],
    };
    expect(configKey(a)).toBe(configKey(b));
  });
});

describe('parseMessageParts', () => {
  it('returns a single text part when there is no <think> block (Gemma path)', () => {
    expect(parseMessageParts('Hello, world.')).toEqual([{ type: 'text', text: 'Hello, world.' }]);
  });

  it('returns [] for empty content (streaming placeholder)', () => {
    expect(parseMessageParts('')).toEqual([]);
  });

  it('marks an unclosed <think> block as streaming reasoning', () => {
    expect(parseMessageParts('<think>Let me work this out')).toEqual([
      { type: 'reasoning', text: 'Let me work this out', state: 'streaming' },
    ]);
  });

  it('splits a closed <think> block and the answer into two parts', () => {
    const parts = parseMessageParts('<think>17 + 25 = 42</think>\n\nThe answer is 42.');
    expect(parts).toEqual([
      { type: 'reasoning', text: '17 + 25 = 42', state: 'done' },
      { type: 'text', text: '\n\nThe answer is 42.' },
    ]);
  });

  it('closes an open reasoning block when finalized (done=true)', () => {
    expect(parseMessageParts('<think>cut off mid-thought', true)).toEqual([
      { type: 'reasoning', text: 'cut off mid-thought', state: 'done' },
    ]);
  });

  it('keeps leading answer text before a think block', () => {
    const parts = parseMessageParts('Sure.<think>reasoning</think>Done.');
    expect(parts).toEqual([
      { type: 'text', text: 'Sure.' },
      { type: 'reasoning', text: 'reasoning', state: 'done' },
      { type: 'text', text: 'Done.' },
    ]);
  });
});

describe('parseMessageParts — empty think block', () => {
  it('drops an empty closed <think></think> and keeps only the answer', () => {
    expect(parseMessageParts('<think>  </think>  Hi there!')).toEqual([
      { type: 'text', text: '  Hi there!' },
    ]);
  });
});
