/**
 * Pure model-id → HuggingFace-checkpoint derivation.
 *
 * Mirror of the backend registry (`backend/services/model_registry.py`):
 * known model ids resolve via the record; unregistered gemma-style ids fall
 * back to the legacy string rule. Keep the two in sync.
 *
 * Split out of modelLoader.ts so it stays importable (and unit-testable)
 * without dragging in the Apollo client.
 */

export const MODEL_CHECKPOINTS: Record<string, string> = {
  // The model_id names the SAE training provenance (Base); the checkpoint is
  // the instruct model we chat with — deliberate, see the Phase-1 qwen plan.
  'qwen3-1.7B-base': 'Qwen/Qwen3-1.7B',
};

/**
 * Construct the HF checkpoint string for a stored model ID.
 *
 * Gemma fallback rule matches the backend's `_normalize_checkpoint` (appends
 * the `-pt` variant when missing) so `ensureModelLoaded` comparisons work:
 *   "gemma-3-4b-it"  → "google/gemma-3-4b-it"
 *   "gemma-3-1b"     → "google/gemma-3-1b-pt"
 */
export function modelIdToCheckpoint(modelId: string): string {
  const registered = MODEL_CHECKPOINTS[modelId];
  if (registered) return registered;
  // Strip any org prefix that might already be present
  const name = modelId.includes('/') ? modelId.split('/').pop()! : modelId;
  // Parse: strip "gemma-3-" prefix, then split on first "-" for size/variant
  const stripped = name.startsWith('gemma-3-') ? name.slice('gemma-3-'.length) : name;
  const dashIdx = stripped.indexOf('-');
  const variant = dashIdx >= 0 ? stripped.slice(dashIdx + 1) : 'pt';
  const size = dashIdx >= 0 ? stripped.slice(0, dashIdx) : stripped;
  const canonical = variant ? `gemma-3-${size}-${variant}` : `gemma-3-${size}-pt`;
  return `google/${canonical}`;
}

/**
 * Check if a loaded model name matches a requested checkpoint.
 * Handles normalization differences (e.g. backend may store "google/gemma-3-1b-pt"
 * while frontend sends "google/gemma-3-1b").
 */
export function isModelMatch(loadedName: string | null | undefined, checkpoint: string): boolean {
  if (!loadedName) return false;
  if (loadedName === checkpoint) return true;
  // Normalize both sides and compare
  const loadedBase = loadedName.includes('/') ? loadedName.split('/').pop()! : loadedName;
  const checkBase = checkpoint.includes('/') ? checkpoint.split('/').pop()! : checkpoint;
  return loadedBase === checkBase;
}

/** Family test for qwen-gated UI (thinking toggle, future per-family ranges). */
export function isQwenModel(modelId: string | null | undefined): boolean {
  return modelId?.toLowerCase().startsWith('qwen') ?? false;
}

/** Short human-readable model family name for display copy. */
export function modelDisplayName(modelId: string | null | undefined): string {
  if (!modelId) return 'Gemma';
  if (isQwenModel(modelId)) return 'Qwen';
  if (modelId.toLowerCase().startsWith('gemma')) return 'Gemma';
  return modelId;
}
