/**
 * Single source of truth for SAE collection <-> model/layer mappings.
 * Add new entries here when ingesting SAE data for additional models.
 */

interface SaeIdentifier {
  modelId: string;
  saeId: string;
}

interface SaeEntry {
  collectionName: string;
  modelId: string;
  saeId: string;
  /** ChromaDB collection with explanation-embedding vectors for semantic search */
  embeddedCollectionName?: string;
}

const SAE_ENTRIES: SaeEntry[] = [
  {
    collectionName: 'gemma_9_16k',
    modelId: 'gemma-3-4b-it',
    saeId: '9-gemmascope-2-res-16k',
    embeddedCollectionName: 'Gemma_9_16k_embedded',
  },
  {
    collectionName: 'Gemma_9_16k_embedded',
    modelId: 'gemma-3-4b-it',
    saeId: '9-gemmascope-2-res-16k',
  },
];

/** Collection name -> SAE model/layer (used by visualization to detect SAE collections) */
export const COLLECTION_TO_SAE: Record<string, SaeIdentifier> = Object.fromEntries(
  SAE_ENTRIES.map((e) => [e.collectionName, { modelId: e.modelId, saeId: e.saeId }]),
);

/** "modelId::saeId" -> collection name (used by features page to link back to visualization) */
export const SAE_TO_COLLECTION: Record<string, string> = Object.fromEntries(
  SAE_ENTRIES.map((e) => [`${e.modelId}::${e.saeId}`, e.collectionName]),
);

/** Check if a collection name is an SAE collection */
export function getSaeInfo(collectionName: string | null): SaeIdentifier | null {
  if (!collectionName) return null;
  return COLLECTION_TO_SAE[collectionName] ?? null;
}

/**
 * Get the ChromaDB collection name that has explanation-embedding vectors
 * for semantic search. Returns null if no embedded collection exists.
 * @param modelSaeKey — "modelId::saeId" format
 */
export function getSemanticCollectionName(modelSaeKey: string): string | null {
  const [modelId, saeId] = modelSaeKey.split('::');
  const entry = SAE_ENTRIES.find(
    (e) => e.modelId === modelId && e.saeId === saeId && e.embeddedCollectionName,
  );
  return entry?.embeddedCollectionName ?? null;
}

/** The metadata field on SAE collection items that holds the feature index */
export const SAE_FEATURE_INDEX_FIELD = 'index';

// ── saeId parsing ──────────────────────────────────────────────────

export type HookType = 'RESID_POST' | 'MLP_OUT' | 'ATTN_OUT';

export interface ParsedSaeId {
  layerIndex: number;
  hookType: HookType;
  width: string;
}

/** Inverts the backend's HOOK_TO_NEURONPEDIA mapping (interpret/sae/source_ids.py). */
const NEURONPEDIA_TO_HOOK: Record<string, HookType> = {
  res: 'RESID_POST',
  mlp: 'MLP_OUT',
  att: 'ATTN_OUT',
};

/**
 * Parse a Neuronpedia-format saeId into its structured components.
 * Format: "{layer}-gemmascope-{version}-{hookAbbrev}-{width}"
 * Example: "9-gemmascope-2-res-16k" → { layerIndex: 9, hookType: 'RESID_POST', width: '16k' }
 */
export function parseSaeId(saeId: string): ParsedSaeId {
  const parts = saeId.split('-');
  const layerIndex = parseInt(parts[0], 10) || 0;
  const hookAbbrev = parts[3] ?? 'res';
  const width = parts[4] ?? '16k';
  const hookType = NEURONPEDIA_TO_HOOK[hookAbbrev] ?? 'RESID_POST';
  return { layerIndex, hookType, width };
}
