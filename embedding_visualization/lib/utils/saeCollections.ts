/**
 * SAE collection <-> model/layer mappings.
 *
 * Primary lookup: `getSaeInfoFromMetadata()` reads `sae_model_id` / `sae_id`
 * from the collection's DuckDB metadata (set via Collection Manager UI).
 *
 * Fallback: `getSaeInfo()` checks the hardcoded `SAE_ENTRIES` for legacy
 * collections that predate the dynamic metadata approach.
 */

export interface SaeIdentifier {
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
  // Label collections whose vectors ARE the label embeddings — they serve as
  // their own semantic-search target (verified: items carry metadata.index).
  {
    collectionName: 'sae_4b_22_res_16k_labels',
    modelId: 'gemma-3-4b-it',
    saeId: '22-gemmascope-2-res-16k',
    embeddedCollectionName: 'sae_4b_22_res_16k_labels',
  },
  {
    collectionName: 'sae_1b_pt_22_res_16k_labels',
    modelId: 'gemma-3-1b',
    saeId: '22-gemmascope-2-res-16k',
    embeddedCollectionName: 'sae_1b_pt_22_res_16k_labels',
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

/**
 * Read SAE info from collection metadata (dynamic, preferred).
 * Returns null if the metadata doesn't contain SAE linkage fields.
 */
export function getSaeInfoFromMetadata(
  metadata: { sae_model_id?: string | null; sae_id?: string | null } | null | undefined,
): SaeIdentifier | null {
  if (!metadata) return null;
  const { sae_model_id, sae_id } = metadata;
  if (typeof sae_model_id === 'string' && typeof sae_id === 'string') {
    return { modelId: sae_model_id, saeId: sae_id };
  }
  return null;
}

/** Check if a collection name is an SAE collection (hardcoded fallback) */
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

/**
 * Get all embedded collection names matching a set of (modelId, saeId) pairs.
 * Used for cross-SAE semantic search fan-out.
 */
export function getSemanticCollections(
  saePairs: Array<{ modelId: string; saeId: string }>,
): Array<{ modelId: string; saeId: string; collectionName: string }> {
  return saePairs
    .map(({ modelId, saeId }) => {
      const entry = SAE_ENTRIES.find(
        (e) => e.modelId === modelId && e.saeId === saeId && e.embeddedCollectionName,
      );
      return entry
        ? { modelId, saeId, collectionName: entry.embeddedCollectionName! }
        : null;
    })
    .filter((x): x is { modelId: string; saeId: string; collectionName: string } => x !== null);
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

/** Short abbreviation for each hook type (matches Neuronpedia convention). */
export const HOOK_TYPE_SHORT: Record<HookType, string> = {
  RESID_POST: 'res',
  MLP_OUT: 'mlp',
  ATTN_OUT: 'att',
};

/** Human-readable display name for each hook type. */
export const HOOK_TYPE_DISPLAY: Record<HookType, string> = {
  RESID_POST: 'Residual',
  MLP_OUT: 'MLP',
  ATTN_OUT: 'Attention',
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

/**
 * Build a Neuronpedia-format saeId from structured components — the inverse
 * of parseSaeId. Matches the backend's `neuronpedia_source_id` derivation
 * (interpret/sae/source_ids.py).
 */
export function buildSaeId(layerIndex: number, hookType: HookType, width: string): string {
  return `${layerIndex}-gemmascope-2-${HOOK_TYPE_SHORT[hookType]}-${width}`;
}
