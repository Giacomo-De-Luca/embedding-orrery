/** Pure helpers for the Manage tab's collection list. */

export interface FilterableCollection {
  name: string;
  embeddingModel?: string | null;
}

/** Case-insensitive substring filter on collection name and embedding model. */
export function filterCollections<T extends FilterableCollection>(
  collections: T[],
  query: string
): T[] {
  const q = query.trim().toLowerCase();
  if (!q) return collections;
  return collections.filter(
    (c) =>
      c.name.toLowerCase().includes(q) ||
      (c.embeddingModel ?? '').toLowerCase().includes(q)
  );
}

const PROVIDER_SHORT_LABELS: Record<string, string> = {
  SENTENCE_TRANSFORMERS: 'ST',
  OPENAI: 'OpenAI',
  COHERE: 'Cohere',
  OLLAMA: 'Ollama',
  HUGGINGFACE_API: 'HF API',
  GEMINI: 'Gemini',
  QWEN: 'Qwen',
  BGE: 'BGE',
};

/** Compact provider label for list badges; falls back to the raw value. */
export function providerShortLabel(provider: string | null | undefined): string | null {
  if (!provider) return null;
  return PROVIDER_SHORT_LABELS[provider.toUpperCase()] ?? provider;
}

/** Compact item-count label, e.g. 954 → "954", 153_000 → "153k". */
export function formatItemCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1_000)}k`;
  return n.toLocaleString();
}
