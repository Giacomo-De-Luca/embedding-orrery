/**
 * Tests for the steering preset registry.
 *
 * Presets are auto-loaded by ChatPanel on mount when the steering config is
 * empty and the model matches. They are shipped at strength 0 — the user
 * activates them by adjusting the slider.
 */

import { describe, it, expect } from 'vitest';
import { STEERING_PRESETS } from '../steeringPresets';
import { steeringFeatureKey } from '@/lib/stores/useModelIdentityStore';

describe('STEERING_PRESETS', () => {
  it('exposes a five-entry preset bundle for gemma-3-4b-it', () => {
    const presets = STEERING_PRESETS['gemma-3-4b-it'];
    expect(presets).toBeDefined();
    expect(presets).toHaveLength(5);
  });

  it('ships all presets at strength 0', () => {
    for (const p of STEERING_PRESETS['gemma-3-4b-it']) {
      expect(p.strength).toBe(0);
    }
  });

  it('contains three SAE features for layer 9 (16k residual)', () => {
    const sae = STEERING_PRESETS['gemma-3-4b-it'].filter((p) => !p.directionName);
    expect(sae).toHaveLength(3);
    for (const f of sae) {
      expect(f.modelId).toBe('gemma-3-4b-it');
      expect(f.saeId).toBe('9-gemmascope-2-res-16k');
      expect(f.layerIndex).toBe(9);
      expect(f.hookType).toBe('RESID_POST');
      expect(f.width).toBe('16k');
    }
    const indices = sae.map((f) => f.featureIndex).sort((a, b) => a - b);
    expect(indices).toEqual([197, 3289, 4963]);
  });

  it('contains two direction presets (refusal, poetry)', () => {
    const directions = STEERING_PRESETS['gemma-3-4b-it'].filter((p) => p.directionName);
    const names = directions.map((p) => p.directionName).sort();
    expect(names).toEqual(['poetry', 'refusal']);
    // Direction presets carry no SAE coordinates
    for (const d of directions) {
      expect(d.saeId).toBe('');
    }
  });

  it('produces distinct identity keys for SAE features vs directions', () => {
    const keys = STEERING_PRESETS['gemma-3-4b-it'].map(steeringFeatureKey);
    expect(new Set(keys).size).toBe(keys.length);
    // Direction keys use the dedicated namespace
    expect(keys).toContain('gemma-3-4b-it::direction::refusal');
    expect(keys).toContain('gemma-3-4b-it::direction::poetry');
  });
});
