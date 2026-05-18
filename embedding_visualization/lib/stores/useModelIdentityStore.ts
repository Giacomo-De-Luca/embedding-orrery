import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { SteeringConfig, SteeringFeature } from '../types/types';
import type { ParsedSaeId } from '../utils/saeCollections';
import { parseSaeId } from '../utils/saeCollections';
import { modelIdToCheckpoint } from '../utils/modelLoader';

// ---------------------------------------------------------------------------
// Steering feature key (moved here as single source of truth)
// ---------------------------------------------------------------------------

/** Composite identity key for a steering feature. */
export function steeringFeatureKey(
  f: SteeringFeature | { modelId: string; saeId: string; featureIndex: number },
): string {
  return `${f.modelId}::${f.saeId}::${f.featureIndex}`;
}

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface ModelIdentityState {
  // Primary identity (set by selectors / URL via bridge effect)
  modelId: string | null;
  saeId: string | null;

  // Cached parse of saeId (auto-derived via subscription, never set directly)
  parsedSae: ParsedSaeId | null;

  // Derived from modelId (auto-derived via subscription)
  checkpoint: string | null;

  // Backend model status (synced from MODEL_STATUS polling)
  backendLoaded: boolean;
  backendModelName: string | null;
  backendDevice: string | null;
  backendVariant: string | null;
  backendModelSize: string | null;

  // Steering config (single source of truth, moved from page.tsx useState)
  steeringConfig: SteeringConfig;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export interface ModelIdentityActions {
  /** Set the resolved model/SAE identity (called by useSaeSelectors bridge). */
  setIdentity: (modelId: string | null, saeId: string | null) => void;

  /** Sync backend model status from MODEL_STATUS query result. */
  syncBackendStatus: (status: {
    loaded: boolean;
    modelName: string | null;
    device: string | null;
    variant?: string | null;
    modelSize?: string | null;
  }) => void;

  // Steering config mutations
  setSteeringConfig: (config: SteeringConfig) => void;
  addSteeringFeature: (feature: SteeringFeature) => void;
  removeSteeringFeature: (key: string) => void;
  updateSteeringStrength: (key: string, strength: number) => void;
}

export type ModelIdentityStore = ModelIdentityState & ModelIdentityActions;

// ---------------------------------------------------------------------------
// Derived state helpers
// ---------------------------------------------------------------------------

function deriveFromSaeId(saeId: string | null): {
  parsedSae: ParsedSaeId | null;
} {
  return { parsedSae: saeId ? parseSaeId(saeId) : null };
}

function deriveCheckpoint(modelId: string | null): string | null {
  return modelId ? modelIdToCheckpoint(modelId) : null;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useModelIdentityStore = create<ModelIdentityStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    modelId: null,
    saeId: null,
    parsedSae: null,
    checkpoint: null,
    backendLoaded: false,
    backendModelName: null,
    backendDevice: null,
    backendVariant: null,
    backendModelSize: null,
    steeringConfig: { features: [] },

    // ---- Actions ----

    setIdentity: (modelId, saeId) => {
      const prev = get();
      if (prev.modelId === modelId && prev.saeId === saeId) return;

      const parsed = deriveFromSaeId(saeId);
      const checkpoint = deriveCheckpoint(modelId);

      set({ modelId, saeId, ...parsed, checkpoint });

      // Sync steering config to new identity when it changes
      const { steeringConfig } = get();
      if (modelId && saeId && parsed.parsedSae) {
        const currentModelId = steeringConfig.features[0]?.modelId;
        const currentSaeId = steeringConfig.features[0]?.saeId;
        if (currentModelId !== modelId || currentSaeId !== saeId) {
          // Model changed: update all features to use new identity
          if (steeringConfig.features.length > 0 && currentModelId === modelId) {
            // Same model, different SAE: update saeId and parsed fields
            set({
              steeringConfig: {
                features: steeringConfig.features.map((f) => ({
                  ...f,
                  saeId,
                  layerIndex: parsed.parsedSae!.layerIndex,
                  hookType: parsed.parsedSae!.hookType,
                  width: parsed.parsedSae!.width,
                })),
              },
            });
          } else {
            // Different model: reset to empty steering config
            set({ steeringConfig: { features: [] } });
          }
        }
      }
    },

    syncBackendStatus: (status) => {
      set({
        backendLoaded: status.loaded,
        backendModelName: status.modelName,
        backendDevice: status.device,
        backendVariant: status.variant ?? null,
        backendModelSize: status.modelSize ?? null,
      });
    },

    setSteeringConfig: (config) => set({ steeringConfig: config }),

    addSteeringFeature: (feature) => {
      const key = steeringFeatureKey(feature);
      set((prev) => ({
        steeringConfig: {
          features: [
            ...prev.steeringConfig.features.filter(
              (x) => steeringFeatureKey(x) !== key,
            ),
            feature,
          ],
        },
      }));
    },

    removeSteeringFeature: (key) => {
      set((prev) => ({
        steeringConfig: {
          features: prev.steeringConfig.features.filter(
            (x) => steeringFeatureKey(x) !== key,
          ),
        },
      }));
    },

    updateSteeringStrength: (key, strength) => {
      set((prev) => ({
        steeringConfig: {
          features: prev.steeringConfig.features.map((f) =>
            steeringFeatureKey(f) === key ? { ...f, strength } : f,
          ),
        },
      }));
    },
  })),
);

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

export const selectModelId = (s: ModelIdentityStore) => s.modelId;
export const selectSaeId = (s: ModelIdentityStore) => s.saeId;
export const selectParsedSae = (s: ModelIdentityStore) => s.parsedSae;
export const selectCheckpoint = (s: ModelIdentityStore) => s.checkpoint;
export const selectSteeringConfig = (s: ModelIdentityStore) => s.steeringConfig;
export const selectBackendVariant = (s: ModelIdentityStore) => s.backendVariant;
export const selectBackendLoaded = (s: ModelIdentityStore) => s.backendLoaded;
