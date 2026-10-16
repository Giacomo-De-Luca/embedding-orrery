/**
 * Query-prompt resolution shared by every text-query search (semantic point
 * search and semantic topic search).
 */

/**
 * Map document-time prompt names to query-time equivalents.
 * For example, if documents were embedded with "Retrieval-document",
 * queries should use "Retrieval-query".
 */
const DOCUMENT_TO_QUERY_PROMPT_MAP: Record<string, string> = {
  'Retrieval-document': 'Retrieval-query',
};

/**
 * Get the appropriate query prompt name based on document prompt name.
 */
export function getQueryPromptName(documentPromptName: string | null | undefined): string | null {
  if (!documentPromptName) return null;
  return DOCUMENT_TO_QUERY_PROMPT_MAP[documentPromptName] || documentPromptName;
}

/**
 * The prompt to embed a query with.
 *
 * @param queryPromptName null = none, 'auto' = derive from the collection's
 *   document prompt, anything else = explicit prompt.
 * @param embeddingPromptName the prompt the collection's documents were embedded with.
 */
export function resolveQueryPrompt(
  queryPromptName: string | null | undefined,
  embeddingPromptName: string | null | undefined,
): string | null {
  if (queryPromptName === 'auto') return getQueryPromptName(embeddingPromptName);
  return queryPromptName || null;
}
