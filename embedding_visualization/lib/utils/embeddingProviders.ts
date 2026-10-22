import type { EmbeddingProvider } from '../graphql/mutations';

export interface ProviderConfig {
  defaultModel: string;
  description: string;
  /** Runs inside the backend process (torch): honours the length cap, and the
   * batch size is the number of texts per forward pass. Mirrors the backend's
   * `LOCAL_PROVIDER_VALUES`. */
  local: boolean;
  /** Form default: small for local models (GPU memory grows with the batch),
   * 100 texts per request for remote APIs. */
  defaultBatchSize: number;
}

export const EMBEDDING_PROVIDERS: Record<EmbeddingProvider, ProviderConfig> = {
  SENTENCE_TRANSFORMERS: {
    defaultModel: 'all-MiniLM-L6-v2',
    description: 'Local (no API key)',
    local: true,
    defaultBatchSize: 32,
  },
  OPENAI: {
    defaultModel: 'text-embedding-3-small',
    description: 'Requires CHROMA_OPENAI_API_KEY',
    local: false,
    defaultBatchSize: 100,
  },
  COHERE: {
    defaultModel: 'embed-english-v3.0',
    description: 'Requires CHROMA_COHERE_API_KEY',
    local: false,
    defaultBatchSize: 100,
  },
  OLLAMA: {
    defaultModel: 'nomic-embed-text',
    description: 'Local Ollama server',
    local: false,
    defaultBatchSize: 100,
  },
  GEMINI: {
    defaultModel: 'gemini-embedding-001',
    description: 'Requires GEMINI_API_KEY',
    local: false,
    defaultBatchSize: 100,
  },
  BGE: {
    defaultModel: 'BAAI/bge-m3',
    description: 'Local BGE-M3 model (no API key)',
    local: true,
    defaultBatchSize: 12,
  },
  QWEN: {
    defaultModel: 'Qwen/Qwen3-Embedding-0.6B',
    description: 'Local Qwen3 model (no API key)',
    local: true,
    defaultBatchSize: 32,
  },
  HUGGINGFACE_API: {
    defaultModel: 'sentence-transformers/all-MiniLM-L6-v2',
    description: 'Requires CHROMA_HUGGINGFACE_API_KEY',
    local: false,
    defaultBatchSize: 100,
  },
};

/** Whether the provider's model runs inside the backend (see `ProviderConfig.local`). */
export function isLocalProvider(provider: EmbeddingProvider): boolean {
  return EMBEDDING_PROVIDERS[provider].local;
}
