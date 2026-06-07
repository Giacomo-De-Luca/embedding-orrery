/**
 * Steering preset registry — model-specific demo bundles.
 *
 * Each entry is auto-loaded into the steering config when the chat
 * interface mounts and the user has the matching model selected with
 * no features currently configured. Presets ship at strength 0 (inert);
 * the user activates them by dialling the strength slider.
 *
 * Two kinds of preset:
 *   - SAE features: identified by (modelId, saeId, featureIndex)
 *   - Pre-extracted directions: identified by ``directionName``. These
 *     resolve server-side via DIRECTION_REGISTRY in interpret_service.py
 *     (layer + .pt file baked in there). The SAE-related fields here
 *     are placeholders that the backend ignores.
 */

import type { SteeringFeature } from '@/lib/types/types';

export const STEERING_PRESETS: Record<string, SteeringFeature[]> = {
  'gemma-3-4b-it': [
    // Three SAE features at layer 9, 16k width, residual.
    {
      modelId: 'gemma-3-4b-it',
      saeId: '9-gemmascope-2-res-16k',
      layerIndex: 9,
      featureIndex: 197,
      strength: 0,
      label: 'Religion & spirituality',
      hookType: 'RESID_POST',
      width: '16k',
    },
    {
      modelId: 'gemma-3-4b-it',
      saeId: '9-gemmascope-2-res-16k',
      layerIndex: 9,
      featureIndex: 3289,
      strength: 0,
      label: 'Poetry',
      hookType: 'RESID_POST',
      width: '16k',
    },
    {
      modelId: 'gemma-3-4b-it',
      saeId: '9-gemmascope-2-res-16k',
      layerIndex: 9,
      featureIndex: 4963,
      strength: 0,
      label: 'Sexually explicit',
      hookType: 'RESID_POST',
      width: '16k',
    },
    // Pre-extracted directions. Application layer is baked into the
    // backend's DIRECTION_REGISTRY; we keep layerIndex here for UI
    // display only.
    {
      modelId: 'gemma-3-4b-it',
      saeId: '',
      layerIndex: 14,
      featureIndex: 0,
      strength: 0,
      label: 'Refusal',
      directionName: 'refusal',
    },
  ],
};
