import type { EmbeddingProvider } from '../graphql/mutations';

export interface ProviderConfig {
  defaultModel: string;
  description: string;
}

export const EMBEDDING_PROVIDERS: Record<EmbeddingProvider, ProviderConfig> = {
  SENTENCE_TRANSFORMERS: {
    defaultModel: 'all-MiniLM-L6-v2',
    description: 'Local (no API key)',
  },
  OPENAI: {
    defaultModel: 'text-embedding-3-small',
    description: 'Requires CHROMA_OPENAI_API_KEY',
  },
  COHERE: {
    defaultModel: 'embed-english-v3.0',
    description: 'Requires CHROMA_COHERE_API_KEY',
  },
  OLLAMA: {
    defaultModel: 'nomic-embed-text',
    description: 'Local Ollama server',
  },
  GEMINI: {
    defaultModel: 'gemini-embedding-001',
    description: 'Requires GEMINI_API_KEY',
  },
  BGE: {
    defaultModel: 'BAAI/bge-m3',
    description: 'Local BGE-M3 model (no API key)',
  },
  QWEN: {
    defaultModel: 'Qwen/Qwen3-Embedding-0.6B',
    description: 'Local Qwen3 model (no API key)',
  },
  HUGGINGFACE_API: {
    defaultModel: 'sentence-transformers/all-MiniLM-L6-v2',
    description: 'Requires CHROMA_HUGGINGFACE_API_KEY',
  },
};
