/**
 * Tests for the pure helpers in useSteeringChat.ts.
 *
 * These cover the strength-0 filter behaviour: auto-loaded presets sitting at
 * strength 0 must not (a) trigger a chat reset on appearance/removal, or
 * (b) be shipped to the backend (which would waste SAE loads on no-ops).
 */

import { describe, it, expect } from 'vitest';
import { configKey, buildSteeringInputs } from '../useSteeringChat';
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

describe('buildSteeringInputs', () => {
  it('returns null when all features sit at strength 0', () => {
    const config: SteeringConfig = { features: [SAE_PRESET, REFUSAL_PRESET, POETRY_PRESET] };
    expect(buildSteeringInputs(config)).toBeNull();
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
