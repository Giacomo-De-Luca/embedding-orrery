import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { SteeringConfig, SteeringFeature } from '../types/types';
import type { ParsedSaeId } from '../utils/saeCollections';
import { parseSaeId } from '../utils/saeCollections';
import { modelIdToCheckpoint } from '../utils/modelLoader';

// ---------------------------------------------------------------------------
// Steering feature key (single source of truth across the app)
// ---------------------------------------------------------------------------

/** Composite identity key for a steering feature.
 *
 * Direction-vector presets (``directionName`` set) are keyed independently
 * of the SAE coordinate, so adding a direction never collides with — or
 * is wiped by — SAE-feature presets at any layer/saeId. */
export function steeringFeatureKey(
  f:
    | SteeringFeature
    | { modelId: string; saeId: string; featureIndex: number; directionName?: string },
): string {
  if (f.directionName) {
    return `${f.modelId}::direction::${f.directionName}`;
  }
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
  /** Set the resolved model/SAE identity (called by useSaeSelection bridge). */
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

      // Sync steering config to new identity when it changes.
      // Three cases:
      //   (a) No features → nothing to sync
      //   (b) Same model, different SAE → update SAE-derived fields on SAE features
      //       only; direction presets are SAE-agnostic and are left untouched
      //   (c) Different model → clear features (they reference the wrong model).
      //       Checked independently of saeId — multi-SAE selections pass
      //       saeId=null but still switch models.
      const { steeringConfig } = get();
      if (steeringConfig.features.length === 0) return;
      const currentModelId = steeringConfig.features[0]?.modelId;
      if (modelId && currentModelId !== modelId) {
        // Case (c): different model
        set({ steeringConfig: { features: [] } });
        return;
      }
      if (modelId && saeId && parsed.parsedSae) {
        const currentSaeId = steeringConfig.features[0]?.saeId;
        if (currentSaeId !== saeId) {
          // Case (b): same model, different SAE — remap SAE features only
          set({
            steeringConfig: {
              features: steeringConfig.features.map((f) =>
                f.directionName
                  ? f
                  : {
                      ...f,
                      saeId,
                      layerIndex: parsed.parsedSae!.layerIndex,
                      hookType: parsed.parsedSae!.hookType,
                      width: parsed.parsedSae!.width,
                    },
              ),
            },
          });
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
